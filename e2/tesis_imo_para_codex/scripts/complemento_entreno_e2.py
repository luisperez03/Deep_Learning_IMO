"""Complemento de entrenamiento del E2: lo que exige volver a entrenar.

Bloques (config/complemento_entreno_e2.json):
  sobreajuste  sobreajustar un lote fijo de 4 imágenes, las 10 redes (sanidad de implementación)
  ablacion     TransUNet sin Transformer, sin conexiones de salto y sin término Dice, 5 semillas
  permutacion  máscaras de train barajadas entre imágenes; pipeline completo; p de permutación
  hpo          búsqueda aleatoria con la misma secuencia de intentos para las 10 redes
  piloto       primera(s) época(s) de tareas reales para medir tiempo; se reutilizan al reanudar
  estimar      horas de GPU por bloque a partir del piloto (marcadas como estimadas)

Reutiliza el protocolo de experimento_e2_colab.py: misma pérdida, mismo orden de lotes por
semilla, misma validación por grupos, mismas semillas y checkpoints atómicos por época. Solo
train/val: el test no se carga. Todo lo que produce es exploratorio, porque el test del
benchmark ya fue observado.

Dos diferencias con train_model, ambas sin efecto numérico buscado:
  * caché en RAM de los ejemplos ya preprocesados (mismos arreglos que load_example);
  * cada tarea declara qué conserva: 'best' guarda el mejor checkpoint y borra last.pt al
    terminar; 'ninguno' guarda solo JSON (permutaciones e HPO no necesitan pesos).
"""
import argparse
import copy
import importlib
import importlib.metadata
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experimento_e2_colab as E
from dataset_ready import load_example as _load_example_original
from dataset_ready import metrics, partition
from modelos_e2 import MODELS_E2, SOURCES, main_logits, official_package, resize

PROJECT = E.PROJECT
CONFIG = PROJECT/'config/complemento_entreno_e2.json'
PROTOCOLO = PROJECT/'config/experimento_e2.json'
TEST_CSV = 'test.csv'


# ---------------------------------------------------------------- datos y caché
_CACHE = {}
_USAR_CACHE = True


def ejemplo(row):
    """(x, y, valid) de load_example, memorizado por id. Los arreglos no se modifican."""
    if not _USAR_CACHE:
        return _load_example_original(row)
    key = row['id']
    if key not in _CACHE:
        x, y, v = _load_example_original(row)
        for a in (x, y, v):
            a.setflags(write=False)
        _CACHE[key] = (x, y, v)
    return _CACHE[key]


def _load_example_compatible(row, data_root=None):
    if data_root is not None:
        return _load_example_original(row, data_root)
    return ejemplo(row)


# evaluate_validation y batch_tensors de experimento_e2_colab leen con esta función.
E.load_example = _load_example_compatible


def lote(rows, device, fuente=None):
    """Como E.batch_tensors; con 'fuente' la máscara sale de otra imagen de train."""
    fuente = fuente or {}
    xs, ys, vs = [], [], []
    for row in rows:
        x, y, v = ejemplo(row)
        if row['id'] in fuente:
            y = ejemplo(fuente[row['id']])[1]
        xs.append(x); ys.append(y); vs.append(v)
    x = torch.from_numpy(np.stack(xs)).to(device)
    y = torch.from_numpy(np.stack(ys).astype('float32'))[:, None].to(device)
    v = torch.from_numpy(np.stack(vs))[:, None].to(device)
    return x, y, v


def reubicar(rows, raiz_imagenes=None, raiz_mascaras=None):
    """Solo para pruebas locales: rutas absolutas a imágenes y máscaras fuera del paquete."""
    out = []
    for row in rows:
        row = dict(row)
        if raiz_imagenes:
            row['image_path'] = str((Path(raiz_imagenes)/row['image_path']).resolve())
        if raiz_mascaras:
            row['binary_mask_path'] = str((Path(raiz_mascaras)/Path(row['binary_mask_path']).name).resolve())
        out.append(row)
    return out


def verificar_filas(rows):
    for row in rows:
        for path, esperado in [(PROJECT.parent/row['image_path'], row['image_sha256']),
                               (PROJECT/row['binary_mask_path'], row['binary_mask_sha256'])]:
            if E.sha(path) != esperado:
                raise RuntimeError(f'Datos modificados: {path}')


def filas(args):
    train, val = partition('train'), partition('val')
    if args.local:
        train = reubicar(train, args.raiz_imagenes, args.raiz_mascaras)
        val = reubicar(val, args.raiz_imagenes, args.raiz_mascaras)
    if args.limite_train:
        train = train[:args.limite_train]
    if args.limite_val:
        val = val[:args.limite_val]
    if len(train) < 2 or not val:
        raise ValueError('Se necesitan al menos 2 imágenes de train y 1 de validación')
    return train, val


# ---------------------------------------------------------------- entorno e identidad
def entorno():
    try:
        return E.environment()
    except importlib.metadata.PackageNotFoundError as error:
        # Solo en pruebas locales: el entorno de Colab tiene todos los paquetes.
        return {'incompleto': f'falta {error}', 'torch': torch.__version__,
                'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}


def identidad(args, train, val):
    E.verify_sources()
    if args.local:
        verificar_filas(train + val)
        datos = 'local:' + E.digest(sorted((r['id'], r['image_sha256'], r['binary_mask_sha256'])
                                           for r in train + val))
    else:
        datos = E.verify_data()
    return {'source': E.source_signature(), 'data': datos}


# ---------------------------------------------------------------- modelos de la ablación
class TransUNetVariante(nn.Module):
    """Mismo constructor que TransUNetAdapter (modelos_e2) con dos perillas del código oficial."""

    def __init__(self, num_layers=12, n_skip=3):
        super().__init__()
        pkg = official_package('e2_transunet', SOURCES/'transunet/networks')
        module = importlib.import_module(pkg + '.vit_seg_modeling')
        config = copy.deepcopy(module.CONFIGS['R50-ViT-B_16'])
        config.n_classes, config.n_skip = 1, n_skip
        config.patches.grid = (16, 16)
        config.transformer.num_layers = num_layers
        self.model = module.VisionTransformer(config, img_size=256, num_classes=1)

    def forward(self, x):
        return resize(self.model(x), x.shape[-2:])


def construir(tarea):
    variante = tarea.get('variante')
    if variante in (None, 'completo', 'sin_dice'):
        return MODELS_E2[tarea['modelo']]()
    if variante == 'sin_transformer':
        return TransUNetVariante(num_layers=0)
    if variante == 'sin_saltos':
        return TransUNetVariante(n_skip=0)
    raise ValueError(f'Variante desconocida: {variante}')


def comprobar_variante(tarea, model):
    """La variante quita de verdad el componente; si no, se detiene."""
    variante = tarea.get('variante')
    if variante is None:
        return
    core = model.model
    if variante == 'sin_transformer':
        assert len(core.transformer.encoder.layer) == 0
        assert core.config.n_skip == 3
    elif variante == 'sin_saltos':
        assert len(core.transformer.encoder.layer) == 12
        assert all(block.conv1[0].in_channels == ch for block, ch in
                   zip(core.decoder.blocks, [512, 256, 128, 64]))
    else:
        assert len(core.transformer.encoder.layer) == 12 and core.config.n_skip == 3


# ---------------------------------------------------------------- pérdida
def perdida(salida, y, v, modo, proto):
    if modo == 'bce_dice':
        return E.compute_loss(salida, y, v, proto)
    if modo == 'bce':
        if isinstance(salida, dict):
            raise ValueError('BCE sola definida solo para salidas simples')
        valores = []
        for z, t, m in zip(salida, y, v):
            if not m.any():
                raise ValueError('Imagen sin píxeles válidos')
            valores.append(F.binary_cross_entropy_with_logits(z[m], t[m]))
        return torch.stack(valores).mean()
    raise ValueError(f'Pérdida desconocida: {modo}')


# ---------------------------------------------------------------- permutación
def mapa_permutacion(train, spec):
    """Permutación de las máscaras de train; misma para cualquier sesión con el mismo (semilla, b)."""
    if not spec:
        return {}, None
    orden = np.random.default_rng([spec['semilla'], spec['b']]).permutation(len(train))
    fuente = {train[i]['id']: train[j] for i, j in enumerate(orden)}
    info = {'puntos_fijos': int((orden == np.arange(len(train))).sum()),
            'sha256_orden': E.digest([train[j]['id'] for j in orden])}
    return fuente, info


# ---------------------------------------------------------------- entrenamiento por épocas
def entrenar(tarea, salida, proto, ident, device, train, val, detener_tras=None):
    """Copia de E.train_model generalizada a variante, pérdida, permutación y presupuesto."""
    E.setup_seed(tarea['semilla'])
    dest = Path(salida)/tarea['id']
    dest.mkdir(parents=True, exist_ok=True)
    spec = {'tarea': tarea, 'protocolo': {k: proto[k] for k in
            ('batch_size', 'size', 'threshold', 'pidnet_aux_weight', 'pidnet_boundary_weight')},
            'identidad': ident, 'entorno': entorno(),
            'filas': {'train': len(train), 'val': len(val)}}
    huella = E.digest(spec)
    cfg_file = dest/'config.json'
    if cfg_file.exists():
        if json.loads(cfg_file.read_text(encoding='utf-8'))['fingerprint'] != huella:
            raise RuntimeError(f'Corrida incompatible: {dest}. Usa una carpeta de experimento nueva.')
    else:
        E.atomic_json(cfg_file, {'fingerprint': huella, **spec})
    res_file = dest/'result.json'
    if res_file.exists():
        result = json.loads(res_file.read_text(encoding='utf-8'))
        if result.get('fingerprint') != huella or not result.get('completed'):
            raise RuntimeError(f'Resultado incompatible: {dest}')
        if tarea['conservar'] == 'best' and E.sha(dest/result['best_file']) != result['best_sha256']:
            raise RuntimeError(f'Checkpoint completado corrupto o modificado: {dest}')
        return result
    model = construir(tarea).to(device)
    comprobar_variante(tarea, model)
    optimizer = E.make_optimizer(model, tarea['optim'])
    fuente, info_perm = mapa_permutacion(train, tarea.get('permutacion'))
    last = dest/'last.pt'
    start, best, stale, history, best_file = 0, -1., 0, [], None
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)  # solo archivos propios
        if saved['fingerprint'] != huella:
            raise RuntimeError('Checkpoint incompatible')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start, best, stale = saved['epoch']+1, saved['best'], saved['stale']
        history, best_file = saved['history'], saved['best_file']
        E.restore_rng(saved['rng'])
    t0 = time.perf_counter()
    for epoch in range(start, tarea['epocas']):
        if stale >= tarea['paciencia']:
            break
        model.train(); started = time.perf_counter(); loss_sum = 0.; seen = 0
        for rows in E.training_batches(train, tarea['semilla'], epoch, proto['batch_size']):
            x, y, v = lote(rows, device, fuente)
            optimizer.zero_grad(set_to_none=True)
            loss = perdida(model(x), y, v, tarea['perdida'], proto)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Pérdida no finita en {tarea["id"]}, época {epoch+1}')
            loss.backward(); optimizer.step()
            loss_sum += loss.item()*len(rows); seen += len(rows)
        entreno_s = time.perf_counter() - started
        score, details = E.evaluate_validation(model, val, device)
        if not np.isfinite(score):
            raise RuntimeError('Validación no finita')
        if score > best:
            best, stale = score, 0
            E.atomic_json(dest/'best_val_predictions.json', {'epoch': epoch+1, 'validation': details})
            if tarea['conservar'] == 'best':
                best_file = f'best_epoch_{epoch+1:03d}.pt'
                E.atomic_torch(dest/best_file, {'model': model.state_dict(), 'epoch': epoch,
                                                'fingerprint': huella, 'validation': details})
            else:
                best_file = None
        else:
            stale += 1
        history.append({'epoch': epoch+1, 'train_loss': loss_sum/seen, 'val_group_Dice': score,
                        'seconds': time.perf_counter()-started, 'train_seconds': entreno_s})
        E.atomic_torch(last, {'fingerprint': huella, 'epoch': epoch, 'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(), 'best': best, 'stale': stale,
                              'history': history, 'best_file': best_file, 'rng': E.rng_state()})
        E.atomic_json(dest/'history.json', history)
        for old in dest.glob('best_epoch_*.pt'):
            if old.name != best_file:
                old.unlink()
        print(f'{tarea["id"]} epoch={epoch+1} loss={loss_sum/seen:.4f} val={score:.5f} '
              f'{history[-1]["seconds"]:.1f}s', flush=True)
        if detener_tras is not None and epoch+1 >= detener_tras:
            return {'completed': False, 'id': tarea['id'], 'epochs_done': epoch+1}
    result = {'completed': True, 'fingerprint': huella, 'id': tarea['id'], 'bloque': tarea['bloque'],
              'modelo': tarea['modelo'], 'variante': tarea.get('variante'), 'semilla': tarea['semilla'],
              'optim': tarea['optim'], 'perdida': tarea['perdida'],
              'permutacion': tarea.get('permutacion'), 'permutacion_info': info_perm,
              'intento': tarea.get('intento'),
              'best_val_group_Dice': best, 'best_epoch': max(history, key=lambda h: h['val_group_Dice'])['epoch'],
              'epochs_completed': len(history), 'best_file': best_file,
              'parameters': sum(p.numel() for p in model.parameters()),
              'seconds_this_session': time.perf_counter()-t0,
              'seconds_total': sum(h['seconds'] for h in history),
              'device': str(device), 'test_evaluated': False}
    if tarea['conservar'] == 'best':
        result['best_sha256'] = E.sha(dest/best_file)
    E.atomic_json(res_file, result)
    last.unlink()  # el resultado queda cerrado; last.pt solo servía para reanudar
    return result


# ---------------------------------------------------------------- listas de tareas
def receta(proto, modelo):
    return dict(proto['models'][modelo])


def tareas_ablacion(cfg, proto, variantes, semillas, args):
    modelo = cfg['ablacion']['modelo']
    perdidas = {k: v['perdida'] for k, v in cfg['ablacion']['variantes'].items()}
    perdidas['completo'] = 'bce_dice'
    return [ajustar({'id': f'abl_{var}_s{s}', 'bloque': 'ablacion', 'modelo': modelo, 'variante': var,
                     'semilla': s, 'optim': receta(proto, modelo), 'epocas': proto['epochs'],
                     'paciencia': proto['patience'], 'perdida': perdidas[var],
                     'conservar': cfg['ablacion']['conservar']}, args)
            for var in variantes for s in semillas]


def tareas_permutacion(cfg, proto, bloque, B, args):
    pc = cfg['permutacion']; spec = pc['bloques'][bloque]; modelo = spec['modelo']
    return [ajustar({'id': f'perm_{modelo}_b{b:03d}', 'bloque': f'permutacion:{bloque}', 'modelo': modelo,
                     'semilla': pc['semilla_entrenamiento'], 'optim': receta(proto, modelo),
                     'epocas': proto['epochs'], 'paciencia': proto['patience'], 'perdida': 'bce_dice',
                     'permutacion': {'semilla': pc['semilla_permutaciones'], 'b': b},
                     'conservar': pc['conservar']}, args)
            for b in range(1, B+1)]


def secuencia_hpo(cfg):
    """Secuencia común de intentos; se genera completa para que todo prefijo sea estable."""
    h = cfg['hpo']; rng = np.random.default_rng(h['semilla_secuencia'])
    intentos = [{'t': 0, 'exponente_lr': 0.0, 'weight_decay': None}]
    for t in range(1, h['T_disenado']):
        intentos.append({'t': t, 'exponente_lr': float(rng.uniform(-1, 1)),
                         'weight_decay': float(rng.choice([0.0, 1e-4, 1e-2]))})
    return intentos


def tareas_hpo(cfg, proto, T, modelos, args):
    h = cfg['hpo']; presupuesto = h['presupuesto_por_intento']; out = []
    for intento in secuencia_hpo(cfg)[:T]:
        for modelo in modelos:
            optim = receta(proto, modelo)
            optim['lr'] = float(optim['lr'] * 10**intento['exponente_lr'])
            if intento['weight_decay'] is not None:
                optim['weight_decay'] = intento['weight_decay']
            out.append(ajustar({'id': f'hpo_{modelo}_t{intento["t"]:02d}', 'bloque': 'hpo', 'modelo': modelo,
                                'semilla': presupuesto['semilla'], 'optim': optim, 'intento': intento,
                                'epocas': presupuesto['epocas'], 'paciencia': presupuesto['paciencia'],
                                'perdida': 'bce_dice', 'conservar': h['conservar']}, args))
    return out


def ajustar(tarea, args):
    """Solo pruebas locales: recortar épocas y marcar la tarea para que no se confunda."""
    if args.epocas_max:
        tarea['epocas'] = min(tarea['epocas'], args.epocas_max)
        tarea['prueba_local'] = True
    if args.limite_train or args.limite_val:
        tarea['prueba_local'] = True
    return tarea


def fragmento(tareas, texto):
    if not texto:
        return tareas
    i, n = map(int, texto.split('/'))
    if not 0 <= i < n:
        raise ValueError('--fragmento debe ser i/n con 0 <= i < n')
    return [t for k, t in enumerate(tareas) if k % n == i]


# ---------------------------------------------------------------- sobreajuste a un lote
def sobreajuste(modelo, salida, cfg, proto, ident, device, train, pasos):
    sc = cfg['sobreajuste']
    elegidas = sorted(np.random.default_rng(sc['semilla_eleccion_imagenes'])
                      .choice(len(train), size=min(sc['imagenes'], len(train)), replace=False).tolist())
    rows = [train[i] for i in elegidas]
    spec = {'modelo': modelo, 'pasos': pasos, 'imagenes': [r['id'] for r in rows],
            'optim': receta(proto, modelo), 'identidad': ident, 'entorno': entorno()}
    huella = E.digest(spec)
    dest = Path(salida)/'sobreajuste'; dest.mkdir(parents=True, exist_ok=True)
    archivo = dest/f'{modelo}.json'
    if archivo.exists():
        previo = json.loads(archivo.read_text(encoding='utf-8'))
        if previo['fingerprint'] == huella:
            return previo
        raise RuntimeError(f'{archivo} es de otra configuración; usa otra carpeta')
    E.setup_seed(sc['semilla_entrenamiento'])
    model = MODELS_E2[modelo]().to(device)
    optimizer = E.make_optimizer(model, spec['optim'])
    x, y, v = lote(rows, device)
    curva = []; started = time.perf_counter()
    for paso in range(1, pasos+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = perdida(model(x), y, v, 'bce_dice', proto)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Pérdida no finita: {modelo}, paso {paso}')
        loss.backward(); optimizer.step()
        if paso == 1 or paso % sc['registrar_cada'] == 0 or paso == pasos:
            curva.append({'paso': paso, 'perdida': loss.item()})
    model.eval()
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(20261118)
        prob = main_logits(model(x)).sigmoid().cpu().numpy()
    por_imagen = [metrics(prob[i, 0], y[i, 0].cpu().numpy(), v[i, 0].cpu().numpy())['Dice']
                  for i in range(len(rows))]
    dice = float(np.mean(por_imagen))
    result = {'fingerprint': huella, **{k: spec[k] for k in ('modelo', 'pasos', 'imagenes', 'optim')},
              'dice_lote_modo_eval': dice, 'dice_por_imagen': por_imagen,
              'perdida_inicial': curva[0]['perdida'], 'perdida_final': curva[-1]['perdida'],
              'umbral': sc['umbral_dice'], 'pasa': dice >= sc['umbral_dice'], 'curva': curva,
              'segundos': time.perf_counter()-started, 'device': str(device)}
    E.atomic_json(archivo, result)
    print(f'sobreajuste {modelo}: Dice={dice:.4f} pérdida {curva[0]["perdida"]:.4f}->{curva[-1]["perdida"]:.4f} '
          f'pasa={result["pasa"]}', flush=True)
    return result


# ---------------------------------------------------------------- piloto y estimación
PILOTO = [('ablacion', 'abl_sin_dice_s11', 1), ('ablacion', 'abl_sin_transformer_s11', 1),
          ('ablacion', 'abl_sin_saltos_s11', 1), ('permutacion', 'perm_fcn_b001', 2)]


def estimar(salida, cfg, proto):
    """Horas de GPU por bloque. Todo lo que sale de aquí es una ESTIMACIÓN, no una medición."""
    ref = cfg['referencia_original']; salida = Path(salida)

    def historia(tid):
        p = salida/tid/'history.json'
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

    medidas = {tid: historia(tid) for _, tid, _ in PILOTO}
    faltan = [k for k, v in medidas.items() if not v]
    if faltan:
        raise RuntimeError(f'Falta el piloto de: {faltan}')
    s_full = medidas['abl_sin_dice_s11'][0]['seconds']
    factor = s_full / ref['segundos_por_epoca_l4']['transunet_2d']
    s_fcn = medidas['perm_fcn_b001'][-1]['seconds']  # segunda época: caché ya llena
    e_trans = ref['epocas_medias']['transunet_2d']

    def h(seg):
        return round(seg/3600, 2)

    s_var = {v: medidas[f'abl_{v}_s11'][0]['seconds'] for v in ('sin_dice', 'sin_transformer', 'sin_saltos')}
    abl = {'central': h(sum(s_var.values())*e_trans*5), 'cota_superior_30_epocas': h(sum(s_var.values())*30*5),
           'supuesto': f'{e_trans} épocas medias por corrida, como TransUNet en el benchmark'}
    perm_epocas = []
    for p in salida.glob('perm_fcn_b*/result.json'):
        perm_epocas.append(json.loads(p.read_text(encoding='utf-8'))['epochs_completed'])
    e_perm = float(np.mean(perm_epocas)) if perm_epocas else None
    bloques = cfg['permutacion']['bloques']

    def perm(seg_epoca, B):
        d = {'minimo_6_epocas': h(seg_epoca*6*B), 'cota_superior_30_epocas': h(seg_epoca*30*B)}
        if e_perm:
            d['con_epocas_medias_observadas'] = h(seg_epoca*e_perm*B)
        return d

    hpo_epoca = sum(ref['segundos_por_epoca_l4'][m] for m in cfg['sobreajuste']['modelos'])*factor
    hp = cfg['hpo']
    estim = {
        'estimado': True,
        'nota': 'Estimación con las épocas del piloto y los tiempos del benchmark; no es una medición del bloque.',
        'factor_gpu_frente_al_benchmark': round(factor, 3),
        'segundos_por_epoca_medidos': {**s_var, 'fcn_segunda_epoca_con_cache': s_fcn},
        'horas_l4_estimadas': {
            'sobreajuste_10_redes': h(sum(ref['segundos_por_epoca_l4'].values())*factor
                                      * cfg['sobreajuste']['pasos']/600),
            'ablacion_15_corridas': abl,
            'control_completo_5_corridas': h(s_full*e_trans*5),
            'permutacion_pipeline_fcn': {'B': bloques['pipeline_fcn']['B'],
                                         **perm(s_fcn, bloques['pipeline_fcn']['B'])},
            'permutacion_transunet': {'B': bloques['transunet']['B'],
                                      **perm(s_full, bloques['transunet']['B'])},
            'hpo_por_intento_10_redes': {'minimo_4_epocas': h(hpo_epoca*4),
                                         'cota_superior_10_epocas': h(hpo_epoca*10)},
            'hpo_T_recomendado': {'T': hp['T_recomendado_antes_del_domingo'],
                                  'cota_superior': h(hpo_epoca*10*hp['T_recomendado_antes_del_domingo'])},
            'hpo_T_disenado': {'T': hp['T_disenado'], 'cota_superior': h(hpo_epoca*10*hp['T_disenado'])},
        },
        'epocas_medias_permutacion_fcn_observadas': e_perm,
        'permutaciones_fcn_completas': len(perm_epocas),
    }
    E.atomic_json(salida/'estimacion_computo.json', estim)
    print(json.dumps(estim, indent=2, ensure_ascii=False), flush=True)
    return estim


# ---------------------------------------------------------------- principal
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('bloque', choices=['sobreajuste', 'ablacion', 'permutacion', 'hpo', 'piloto', 'estimar'])
    p.add_argument('--output', required=True)
    p.add_argument('--config', default=str(CONFIG))
    p.add_argument('--modelos', nargs='+')
    p.add_argument('--variantes', nargs='+', choices=['sin_transformer', 'sin_saltos', 'sin_dice', 'completo'])
    p.add_argument('--semillas', nargs='+', type=int)
    p.add_argument('--permutacion', choices=['pipeline_fcn', 'transunet'], default='pipeline_fcn')
    p.add_argument('--B', type=int, help='Número de permutaciones; por defecto el de la configuración')
    p.add_argument('--T', type=int, help='Intentos de HPO; por defecto T_recomendado_antes_del_domingo')
    p.add_argument('--pasos', type=int, help='Pasos de sobreajuste; por defecto los de la configuración')
    p.add_argument('--fragmento', help='i/n: esta sesión corre las tareas k con k mod n = i')
    p.add_argument('--sin-cache', action='store_true')
    # Solo pruebas locales en CPU:
    p.add_argument('--local', action='store_true', help='Permite CPU y rutas locales; no es una corrida válida')
    p.add_argument('--raiz-imagenes'); p.add_argument('--raiz-mascaras')
    p.add_argument('--limite-train', type=int); p.add_argument('--limite-val', type=int)
    p.add_argument('--epocas-max', type=int)
    args = p.parse_args(argv)

    global _USAR_CACHE
    _USAR_CACHE = not args.sin_cache
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    proto = json.loads(PROTOCOLO.read_text(encoding='utf-8'))
    salida = Path(args.output).resolve(); salida.mkdir(parents=True, exist_ok=True)
    if args.bloque == 'estimar':
        estimar(salida, cfg, proto); return
    if not args.local:
        if not torch.cuda.is_available():
            raise RuntimeError('El complemento requiere GPU; las pruebas en CPU van con --local')
        if any([args.limite_train, args.limite_val, args.epocas_max, args.raiz_imagenes, args.raiz_mascaras]):
            raise RuntimeError('Los recortes de prueba solo se admiten con --local')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.local else 'cpu')
    train, val = filas(args)
    ident = identidad(args, train, val)
    semillas = args.semillas or cfg['comun']['semillas']
    resultados = []
    if args.bloque == 'sobreajuste':
        for modelo in args.modelos or cfg['sobreajuste']['modelos']:
            resultados.append(sobreajuste(modelo, salida, cfg, proto, ident, device, train,
                                          args.pasos or cfg['sobreajuste']['pasos']))
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    else:
        if args.bloque == 'ablacion':
            tareas = tareas_ablacion(cfg, proto, args.variantes or list(cfg['ablacion']['variantes']), semillas, args)
        elif args.bloque == 'permutacion':
            B = args.B or cfg['permutacion']['bloques'][args.permutacion]['B']
            tareas = tareas_permutacion(cfg, proto, args.permutacion, B, args)
        elif args.bloque == 'hpo':
            T = args.T or cfg['hpo']['T_recomendado_antes_del_domingo']
            if T > cfg['hpo']['T_disenado']:
                raise ValueError('T mayor que la secuencia diseñada')
            tareas = tareas_hpo(cfg, proto, T, args.modelos or cfg['sobreajuste']['modelos'], args)
        else:  # piloto
            todas = {t['id']: t for t in tareas_ablacion(cfg, proto, ['sin_dice', 'sin_transformer', 'sin_saltos'],
                                                         [11], args)
                     + tareas_permutacion(cfg, proto, 'pipeline_fcn', 1, args)}
            for _, tid, epocas in PILOTO:
                resultados.append(entrenar(todas[tid], salida, proto, ident, device, train, val,
                                           detener_tras=epocas))
            if not args.local:
                estimar(salida, cfg, proto)
            tareas = []
        for tarea in fragmento(tareas, args.fragmento):
            resultados.append(entrenar(tarea, salida, proto, ident, device, train, val))
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    E.atomic_json(salida/f'ultima_llamada_{args.bloque}.json',
                  {'argv': sys.argv[1:] if argv is None else argv, 'resultados': resultados})


if __name__ == '__main__':
    main()
