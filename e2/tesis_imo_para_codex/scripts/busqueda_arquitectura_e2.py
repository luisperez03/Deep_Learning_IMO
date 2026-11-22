"""Búsqueda de arquitectura en dos fases para las tres redes compactas (sección 14 de la guía).

Las siete redes nuevas del benchmark tienen arquitectura publicada; FCN, U-Net y SegNet compactas
son diseño propio y por eso recorren el espacio obligatorio: 3 bloques, anchos {16, 32, 64, 128,
256} en todas las combinaciones y dropout 0,2 en todas las posiciones = 1000 por familia.
Protocolo: config/busqueda_arquitectura_e2.json y docs/PROTOCOLO_BUSQUEDA_ARQUITECTURA_E2.md.

Bloques:
  listar      escribe config/arquitecturas_compactas_e2.csv (id, w, b, parámetros y MACs por familia)
  piloto      una época de 5 anchos x 3 familias a 256 y a 128 px, con la caché llena, para medir tiempo
  estimar     horas de GPU de la búsqueda a partir del piloto (marcadas como estimadas)
  fidelidad   fija la fidelidad de la fase 1 con la regla del protocolo y el presupuesto de horas
  ajuste      ajuste previo de la tasa con la referencia (64,64,64)/(0,1,0), por familia
  fase1       las 1000 arquitecturas x 3 familias, semilla 97, 6 épocas; y la repetición de ruido
  finalistas  ruido por semilla, k y lista de finalistas (solo validación)
  fase2       finalistas + referencia + heredada, 5 semillas, presupuesto completo

Solo train y validación: el test no se carga. Reutiliza de complemento_entreno_e2 la caché en RAM,
la identidad de datos y fuentes y el orden de lotes por semilla del benchmark.
"""
import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import complemento_entreno_e2 as C
from dataset_ready import metrics
from modelos_base import block

E = C.E
PROJECT = E.PROJECT
CONFIG = PROJECT/'config/busqueda_arquitectura_e2.json'
LISTADO = PROJECT/'config/arquitecturas_compactas_e2.csv'
FAMILIAS = ('fcn', 'unet', 'segnet')


# ---------------------------------------------------------------- espacio
def espacio(cfg):
    """Las 1000 arquitecturas en el orden del producto cartesiano, con su identificador."""
    W = cfg['espacio']['anchos']
    out = []
    for k, (w1, w2, w3, b1, b2, b3) in enumerate(itertools.product(W, W, W, (0, 1), (0, 1), (0, 1))):
        out.append({'arq': f'a{k:04d}', 'w': (w1, w2, w3), 'b': (b1, b2, b3)})
    if len(out) != cfg['espacio']['cardinalidad']:
        raise RuntimeError('El espacio no tiene la cardinalidad declarada')
    return out


def buscar(arqs, w, b):
    for a in arqs:
        if a['w'] == tuple(w) and a['b'] == tuple(b):
            return a
    raise KeyError((w, b))


def orden_fase1(cfg, arqs):
    perm = np.random.default_rng(cfg['fase1']['semilla_orden']).permutation(len(arqs))
    return [arqs[i] for i in perm]


# ---------------------------------------------------------------- familias parametrizadas
def _drop(activo, p):
    return nn.Dropout(p) if activo else nn.Identity()


class FCNBusqueda(nn.Module):
    """SmallFCN con anchos (w1, w2, w3) y dropout tras cada bloque. Mismo orden de creación de capas."""

    def __init__(self, w, b, p=0.2):
        super().__init__()
        w1, w2, w3 = w
        self.e1, self.e2, self.e3 = block(3, w1), block(w1, w2), block(w2, w3)
        self.head = nn.Conv2d(w3, 1, 1)
        self.r1, self.r2, self.r3 = (_drop(x, p) for x in b)

    def forward(self, x):
        a = self.r1(self.e1(x))
        c = self.r2(self.e2(F.max_pool2d(a, 2)))
        c = self.r3(self.e3(F.max_pool2d(c, 2)))
        return F.interpolate(self.head(c), size=x.shape[-2:], mode='bilinear', align_corners=False)


class UNetBusqueda(nn.Module):
    """SmallUNet con anchos (w1, w2, w3); el decodificador espeja los anchos y el dropout."""

    def __init__(self, w, b, p=0.2):
        super().__init__()
        w1, w2, w3 = w
        self.e1, self.e2, self.e3 = block(3, w1), block(w1, w2), block(w2, w3)
        self.d2, self.d1 = block(w3 + w2, w2), block(w2 + w1, w1)
        self.head = nn.Conv2d(w1, 1, 1)
        self.r1, self.r2, self.r3 = (_drop(x, p) for x in b)
        self.s2, self.s1 = _drop(b[1], p), _drop(b[0], p)  # espejos de los niveles 2 y 1

    def forward(self, x):
        a = self.r1(self.e1(x))
        m = self.r2(self.e2(F.max_pool2d(a, 2)))
        c = self.r3(self.e3(F.max_pool2d(m, 2)))
        d = self.s2(self.d2(torch.cat([F.interpolate(c, size=m.shape[-2:], mode='nearest'), m], 1)))
        d = self.s1(self.d1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='nearest'), a], 1)))
        return self.head(d)


class SegNetBusqueda(nn.Module):
    """SmallSegNet con anchos (w1, w2, w3); reducción w3 -> w2 y decodificador espejo con unpooling."""

    def __init__(self, w, b, p=0.2):
        super().__init__()
        w1, w2, w3 = w
        self.e1, self.e2, self.e3 = block(3, w1), block(w1, w2), block(w2, w3)
        self.reduce = nn.Conv2d(w3, w2, 1)
        self.d2, self.d1 = block(w2, w1), block(w1, w1)
        self.head = nn.Conv2d(w1, 1, 1)
        self.r1, self.r2, self.r3 = (_drop(x, p) for x in b)
        self.s2, self.s1 = _drop(b[1], p), _drop(b[0], p)

    def forward(self, x):
        a = self.r1(self.e1(x))
        p, i1 = F.max_pool2d(a, 2, return_indices=True)
        m = self.r2(self.e2(p))
        p, i2 = F.max_pool2d(m, 2, return_indices=True)
        c = self.reduce(self.r3(self.e3(p)))
        d = self.s2(self.d2(F.max_unpool2d(c, i2, 2, output_size=m.shape)))
        d = self.s1(self.d1(F.max_unpool2d(d, i1, 2, output_size=a.shape)))
        return self.head(d)


CLASES = {'fcn': FCNBusqueda, 'unet': UNetBusqueda, 'segnet': SegNetBusqueda}


def construir_red(familia, w, b, p):
    return CLASES[familia](tuple(w), tuple(b), p)


class EntradaReducida(nn.Module):
    """Fidelidad reducida de la fase 1: la red ve la imagen a px x px (promedio por área) y su salida
    vuelve a 256 por interpolación bilineal; pérdida y Dice se miden a 256 contra la máscara original."""

    def __init__(self, red, px):
        super().__init__()
        self.red, self.px = red, px

    def forward(self, x):
        z = self.red(F.interpolate(x, size=(self.px, self.px), mode='area'))
        return F.interpolate(z, size=x.shape[-2:], mode='bilinear', align_corners=False)


def subconjunto_train(train, fraccion, semilla):
    """Subconjunto fijo de train por grupos de similitud (no parte grupos), el mismo para todas."""
    if fraccion >= 1:
        return train
    grupos = sorted({r['group_id'] for r in train})
    orden = np.random.default_rng(semilla).permutation(len(grupos))
    elegidos, n = set(), 0
    por_grupo = defaultdict(int)
    for r in train:
        por_grupo[r['group_id']] += 1
    for i in orden:
        if n >= fraccion*len(train):
            break
        elegidos.add(grupos[i]); n += por_grupo[grupos[i]]
    return [r for r in train if r['group_id'] in elegidos]


def parametros(model):
    return sum(q.numel() for q in model.parameters())


def macs(model, size=256, size_calculo=32):
    """Multiplicaciones-acumulaciones de las convoluciones por imagen (sin pooling ni interpolación).

    Se calcula a 32 px y se escala por (size/32)^2: todas las capas son convoluciones con relleno
    que conservan el tamaño y el pooling divide exacto, así que el conteo es proporcional al área.
    """
    total = [0]

    def gancho(mod, inp, out):
        total[0] += out.numel() // out.shape[0] * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
    hs = [m.register_forward_hook(gancho) for m in model.modules() if isinstance(m, nn.Conv2d)]
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, size_calculo, size_calculo))
    model.train(was)
    for h in hs:
        h.remove()
    return int(total[0] * (size // size_calculo)**2)


def listar(cfg, destino=LISTADO):
    arqs = espacio(cfg); p = cfg['espacio']['dropout_p']
    lineas = ['arq,w1,w2,w3,b1,b2,b3,' + ','.join(f'{f}_parametros,{f}_macs' for f in FAMILIAS)]
    for a in arqs:
        celdas = [a['arq'], *map(str, a['w']), *map(str, a['b'])]
        for f in FAMILIAS:
            m = construir_red(f, a['w'], a['b'], p)
            celdas += [str(parametros(m)), str(macs(m))]
        lineas.append(','.join(celdas))
    Path(destino).write_text('\n'.join(lineas) + '\n', encoding='utf-8')
    print(f'{len(arqs)} arquitecturas escritas en {destino} (sha256 {E.sha(destino)[:12]})')
    return destino


# ---------------------------------------------------------------- identidad
DEPENDE_DE = ['scripts/busqueda_arquitectura_e2.py', 'scripts/complemento_entreno_e2.py',
              'scripts/experimento_e2_colab.py', 'scripts/modelos_e2.py', 'scripts/modelos_base.py',
              'scripts/dataset_ready.py', 'config/busqueda_arquitectura_e2.json',
              'config/arquitecturas_compactas_e2.csv', 'config/experimento_e2.json', 'requirements-e2.txt']


def identidad(args, train, val):
    """Como C.identidad, pero la firma de fuentes cubre solo los archivos de los que depende la búsqueda:
    otro script nuevo en scripts/ no invalida las corridas hechas."""
    ident = C.identidad(args, train, val)
    ident['source'] = E.digest({rel: E.sha(PROJECT/rel) for rel in DEPENDE_DE})
    return ident


# ---------------------------------------------------------------- evaluación
@torch.no_grad()
def evaluar(model, rows, device, proto, con_detalle=False):
    """Dice por grupo (como E.evaluate_validation) y pérdida media por imagen, en modo evaluación."""
    model.eval(); grupos = defaultdict(list); detalle = []; perdidas = []
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(20261118)
        for row in rows:
            x, y, v = E.batch_tensors([row], device)
            z = model(x)
            perdidas.append(float(E.compute_loss(z, y, v, proto)))
            prob = z.sigmoid()[0, 0].cpu().numpy()
            s = metrics(prob, y[0, 0].cpu().numpy(), v[0, 0].cpu().numpy())
            grupos[row['group_id']].append(s['Dice'])
            if con_detalle:
                detalle.append({'id': row['id'], 'group_id': row['group_id'], **s})
    return float(np.mean([np.mean(g) for g in grupos.values()])), float(np.mean(perdidas)), detalle


# ---------------------------------------------------------------- entrenamiento
def entrenar(tarea, salida, proto, ident, device, train, val, detener_tras=None):
    """Entrena una arquitectura. Fase 1 y ajuste: sin pesos, se reanuda por tarea. Fase 2: last.pt por época."""
    dest = Path(salida)/tarea['bloque']/tarea['id']
    dest.mkdir(parents=True, exist_ok=True)
    train = subconjunto_train(train, tarea['fraccion_train'], tarea['semilla_subconjunto'])
    spec = {'tarea': tarea, 'protocolo': {k: proto[k] for k in ('batch_size', 'size', 'threshold')},
            'identidad': ident, 'entorno': C.entorno(), 'filas': {'train': len(train), 'val': len(val)}}
    huella = E.digest(spec)
    res_file = dest/'result.json'
    if res_file.exists():
        result = json.loads(res_file.read_text(encoding='utf-8'))
        if result.get('fingerprint') != huella or not result.get('completed'):
            raise RuntimeError(f'Resultado incompatible: {dest}. Usa una carpeta de salida nueva.')
        return result
    cfg_file = dest/'config.json'
    if cfg_file.exists() and json.loads(cfg_file.read_text(encoding='utf-8'))['fingerprint'] != huella:
        raise RuntimeError(f'Corrida incompatible: {dest}')
    E.atomic_json(cfg_file, {'fingerprint': huella, **spec})
    E.setup_seed(tarea['semilla'])
    model = construir_red(tarea['familia'], tarea['w'], tarea['b'], tarea['dropout_p'])
    n_param = parametros(model)
    if tarea['px'] != proto['size']:
        model = EntradaReducida(model, tarea['px'])
    model = model.to(device)
    optimizer = E.make_optimizer(model, tarea['optim'])
    guardar = tarea['conservar'] == 'best'
    last = dest/'last.pt'
    start, best, stale, history, best_state = 0, -1., 0, [], None
    if guardar and last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)  # solo archivos propios
        if saved['fingerprint'] != huella:
            raise RuntimeError('Checkpoint incompatible')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start, best, stale, history = saved['epoch'] + 1, saved['best'], saved['stale'], saved['history']
        best_state = saved['best_state']
        E.restore_rng(saved['rng'])
    t0 = time.perf_counter()
    for epoch in range(start, tarea['epocas']):
        if stale >= tarea['paciencia']:
            break
        model.train(); started = time.perf_counter(); loss_sum = 0.; seen = 0
        for rows in E.training_batches(train, tarea['semilla'], epoch, proto['batch_size']):
            x, y, v = C.lote(rows, device)
            optimizer.zero_grad(set_to_none=True)
            loss = E.compute_loss(model(x), y, v, proto)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Pérdida no finita en {tarea["id"]}, época {epoch+1}')
            loss.backward(); optimizer.step()
            loss_sum += loss.item()*len(rows); seen += len(rows)
        entreno_s = time.perf_counter() - started
        score, val_loss, detalle = evaluar(model, val, device, proto, con_detalle=guardar)
        if not np.isfinite(score):
            raise RuntimeError('Validación no finita')
        if score > best:
            best, stale = score, 0
            if guardar:
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
                E.atomic_json(dest/'best_val_predictions.json', {'epoch': epoch+1, 'validation': detalle})
        else:
            stale += 1
        history.append({'epoch': epoch+1, 'train_loss': loss_sum/seen, 'val_loss': val_loss,
                        'val_group_Dice': score, 'seconds': time.perf_counter()-started,
                        'train_seconds': entreno_s})
        if guardar:
            E.atomic_torch(last, {'fingerprint': huella, 'epoch': epoch, 'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict(), 'best': best, 'stale': stale,
                                  'history': history, 'best_state': best_state, 'rng': E.rng_state()})
        print(f'{tarea["id"]} epoch={epoch+1} loss={loss_sum/seen:.4f} val={score:.5f} '
              f'{history[-1]["seconds"]:.1f}s', flush=True)
        if detener_tras is not None and epoch+1 >= detener_tras:
            return {'completed': False, 'id': tarea['id'], 'epochs_done': epoch+1, 'history': history}
    mejor = max(history, key=lambda h: h['val_group_Dice'])
    result = {'completed': True, 'fingerprint': huella, **{k: tarea[k] for k in
              ('id', 'bloque', 'familia', 'arq', 'w', 'b', 'semilla', 'optim', 'epocas', 'paciencia',
               'px', 'fraccion_train')},
              'n_train': len(train),
              'best_val_group_Dice': best, 'best_epoch': mejor['epoch'],
              'val_loss_best_epoch': mejor['val_loss'], 'train_loss_best_epoch': mejor['train_loss'],
              'epochs_completed': len(history), 'history': history,
              'parameters': n_param, 'macs': tarea['macs'], 'macs_efectivos': tarea['macs_efectivos'],
              'seconds_this_session': time.perf_counter()-t0,
              'seconds_total': sum(h['seconds'] for h in history),
              'device': str(device), 'test_evaluated': False}
    if guardar:
        model.load_state_dict(best_state)
        tr, tr_loss, _ = evaluar(model, train, device, proto)
        result.update({'best_train_group_Dice': tr, 'best_train_loss_eval': tr_loss,
                       'brecha_train_val': tr - best, 'best_file': 'best.pt'})
        E.atomic_torch(dest/'best.pt', {'model': best_state, 'epoch': mejor['epoch'], 'fingerprint': huella})
        result['best_sha256'] = E.sha(dest/'best.pt')
    E.atomic_json(res_file, result)
    if last.exists():
        last.unlink()
    return result


# ---------------------------------------------------------------- tareas
def receta(cfg, familia, lr=None):
    r = cfg['receta']
    return {'optimizer': r['optimizer'], 'lr': float(lr if lr is not None else r['lr']),
            'weight_decay': r['weight_decay']}


def tarea(cfg, bloque, familia, arq, semilla, lr, epocas, paciencia, conservar, args, macs_tabla,
          px=256, fraccion=1.0):
    sufijo = (f'_px{px}' if px != 256 else '') + (f'_f{round(100*fraccion)}' if fraccion < 1 else '')
    t = {'id': f'{familia}_{arq["arq"]}_s{semilla}' + (f'_lr{lr:g}' if bloque == 'ajuste' else '') + sufijo,
         'bloque': bloque, 'familia': familia, 'arq': arq['arq'], 'w': list(arq['w']), 'b': list(arq['b']),
         'dropout_p': cfg['espacio']['dropout_p'], 'semilla': semilla, 'optim': receta(cfg, familia, lr),
         'epocas': epocas, 'paciencia': paciencia, 'conservar': conservar,
         'macs': macs_tabla[(familia, arq['arq'])],
         'macs_efectivos': int(macs_tabla[(familia, arq['arq'])]*(px/256)**2),
         'px': px, 'fraccion_train': fraccion, 'semilla_subconjunto': cfg['fidelidad_fase1']['semilla_subconjunto']}
    if args.epocas_max:
        t['epocas'] = min(t['epocas'], args.epocas_max); t['prueba_local'] = True
    if args.limite_train or args.limite_val:
        t['prueba_local'] = True
    return t


def tabla_macs(cfg, arqs, familias):
    """MACs por (familia, arq) desde el listado versionado; si falta, se calcula y se comprueba."""
    tabla = {}
    if LISTADO.exists():
        lineas = LISTADO.read_text(encoding='utf-8').splitlines()
        cab = lineas[0].split(',')
        for linea in lineas[1:]:
            c = dict(zip(cab, linea.split(',')))
            for f in FAMILIAS:
                tabla[(f, c['arq'])] = int(c[f'{f}_macs'])
    if len(tabla) != len(arqs)*len(FAMILIAS):
        p = cfg['espacio']['dropout_p']
        for a in arqs:
            for f in familias:
                tabla[(f, a['arq'])] = macs(construir_red(f, a['w'], a['b'], p))
    return tabla


def fidelidad(salida):
    archivo = Path(salida)/'fidelidad.json'
    if not archivo.exists():
        raise RuntimeError('Falta fidelidad.json: corre antes el bloque fidelidad')
    f = json.loads(archivo.read_text(encoding='utf-8'))['elegida']
    return f['px'], f['fraccion_train']


def lr_elegida(salida, familia):
    archivo = Path(salida)/'ajuste_previo.json'
    if not archivo.exists():
        raise RuntimeError('Falta ajuste_previo.json: corre antes el bloque ajuste')
    return json.loads(archivo.read_text(encoding='utf-8'))['lr_elegida'][familia]


def tareas_ajuste(cfg, arqs, familias, salida, args, mt):
    ref = buscar(arqs, cfg['referencia']['w'], cfg['referencia']['b']); f1 = cfg['fase1']
    px, fr = fidelidad(salida)
    return [tarea(cfg, 'ajuste', f, ref, f1['semilla'], lr, f1['epocas'], f1['paciencia'], 'ninguno', args, mt,
                  px, fr)
            for f in familias for lr in cfg['ajuste_previo']['lr']]


def tareas_fase1(cfg, arqs, familias, salida, args, mt, n_arq=None):
    f1 = cfg['fase1']; orden = orden_fase1(cfg, arqs)[:n_arq]
    n_ruido = cfg['ruido']['n_arquitecturas'] if n_arq is None else min(n_arq, 2)
    lrs = {f: lr_elegida(salida, f) for f in familias}
    px, fr = fidelidad(salida)
    out = []
    for i, a in enumerate(orden):
        for f in familias:
            out.append(tarea(cfg, 'fase1', f, a, f1['semilla'], lrs[f], f1['epocas'], f1['paciencia'],
                             'ninguno', args, mt, px, fr))
        if i < n_ruido:  # repetición para el ruido por semilla, intercalada al principio
            for f in familias:
                out.append(tarea(cfg, 'fase1', f, a, cfg['ruido']['semilla_extra'], lrs[f], f1['epocas'],
                                 f1['paciencia'], 'ninguno', args, mt, px, fr))
    return out


def tareas_fase2(cfg, arqs, familias, salida, args, mt):
    fin = json.loads((Path(salida)/'finalistas.json').read_text(encoding='utf-8'))
    f2 = cfg['fase2']; por_id = {a['arq']: a for a in arqs}; out = []
    for s in f2['semillas']:
        for f in familias:
            for arq in fin['familias'][f]['fase2']:
                out.append(tarea(cfg, 'fase2', f, por_id[arq], s, lr_elegida(salida, f), f2['epocas'],
                                 f2['paciencia'], 'best', args, mt))
    return out


# ---------------------------------------------------------------- decisiones entre bloques
def resultados(salida, bloque):
    out = []
    for p in sorted((Path(salida)/bloque).glob('*/result.json')):
        r = json.loads(p.read_text(encoding='utf-8'))
        if r.get('completed'):
            out.append(r)
    return out


def cerrar_ajuste(cfg, salida, familias):
    """Regla de 14.3, fijada antes: mayor Dice; si gana a la receta por menos de 0,005, la receta."""
    res = resultados(salida, 'ajuste'); ap = cfg['ajuste_previo']; base = cfg['receta']['lr']
    elegida, tabla = {}, {}
    for f in familias:
        por_lr = {r['optim']['lr']: r['best_val_group_Dice'] for r in res if r['familia'] == f}
        if sorted(por_lr) != sorted(ap['lr']):
            raise RuntimeError(f'Ajuste previo incompleto en {f}: {sorted(por_lr)}')
        mejor = max(por_lr, key=por_lr.get)
        elegida[f] = mejor if por_lr[mejor] - por_lr[base] >= 0.005 else base
        tabla[f] = {f'{k:g}': v for k, v in por_lr.items()}
    out = {'regla': ap['regla'], 'dice_validacion': tabla, 'lr_elegida': elegida,
           'arquitectura': {'w': cfg['referencia']['w'], 'b': cfg['referencia']['b']}}
    E.atomic_json(Path(salida)/'ajuste_previo.json', out)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return out


def cerrar_finalistas(cfg, salida, familias, arqs, permitir_incompleta=False):
    res = resultados(salida, 'fase1'); f1 = cfg['fase1']
    por_id = {a['arq']: a for a in arqs}
    ref = buscar(arqs, cfg['referencia']['w'], cfg['referencia']['b'])['arq']
    her = buscar(arqs, cfg['heredada']['w'], cfg['heredada']['b'])['arq']
    info, rs = {}, {}
    for f in familias:
        principal = {r['arq']: r for r in res if r['familia'] == f and r['semilla'] == f1['semilla']}
        extra = {r['arq']: r for r in res if r['familia'] == f and r['semilla'] == cfg['ruido']['semilla_extra']}
        if len(principal) < len(arqs) and not permitir_incompleta:
            raise RuntimeError(f'Fase 1 incompleta en {f}: {len(principal)} de {len(arqs)}')
        y = np.array([r['best_val_group_Dice'] for r in principal.values()])
        pares = [(principal[a]['best_val_group_Dice'], extra[a]['best_val_group_Dice']) for a in extra if a in principal]
        d = np.array([p - q for p, q in pares])
        s_sem = float(np.sqrt(np.mean(d**2)/2)) if len(d) else float('nan')
        s_tot = float(np.std(y, ddof=1))
        s_arq = float(np.sqrt(max(s_tot**2 - s_sem**2, 0.)))
        rs[f] = s_sem/s_arq if s_arq > 0 else float('inf')
        ranking = sorted(principal.values(), key=lambda r: (-r['best_val_group_Dice'], r['parameters']))
        info[f] = {'n_fase1': len(principal), 'pares_ruido': len(d), 'sigma_semilla': s_sem,
                   'sigma_total': s_tot, 'sigma_arquitectura': s_arq, 'r': rs[f],
                   'ranking': [r['arq'] for r in ranking],
                   'rango_referencia': [r['arq'] for r in ranking].index(ref) + 1 if ref in principal else None,
                   'rango_heredada': [r['arq'] for r in ranking].index(her) + 1 if her in principal else None}
    k = 10 if all(r <= 1 for r in rs.values()) else 20
    out = {'regla_k': cfg['ruido']['regla_k'], 'k': k, 'referencia': ref, 'heredada': her,
           'fase1_completa': all(info[f]['n_fase1'] == len(arqs) for f in familias), 'familias': {}}
    for f in familias:
        top = info[f]['ranking'][:k]
        fase2 = top + [a for a in (ref, her) if a not in top]
        out['familias'][f] = {**{x: y for x, y in info[f].items() if x != 'ranking'},
                              'finalistas': top, 'fase2': fase2,
                              'detalle': [{'arq': a, 'w': por_id[a]['w'], 'b': por_id[a]['b']} for a in fase2]}
    E.atomic_json(Path(salida)/'finalistas.json', out)
    print(json.dumps({f: {'k': k, 'r': rs[f], 'fase2': out['familias'][f]['fase2']} for f in familias},
                     indent=2), flush=True)
    return out


# ---------------------------------------------------------------- piloto, estimación y fidelidad
PILOTO = [((16, 16, 16), (0, 0, 0)), ((16, 32, 64), (0, 0, 0)), ((64, 64, 64), (0, 1, 0)),
          ((128, 128, 128), (0, 0, 0)), ((256, 256, 256), (1, 1, 1))]


def tareas_piloto(cfg, arqs, args, mt):
    """Cinco anchos que cubren el rango de costo, en las tres familias, a 256 y a 128 px, una época."""
    out = []
    for px in sorted({o['px'] for o in cfg['fidelidad_fase1']['opciones']}, reverse=True):
        for w, b in PILOTO:
            for f in FAMILIAS:
                out.append(tarea(cfg, 'piloto', f, buscar(arqs, w, b), cfg['fase1']['semilla'], None, 1, 5,
                                 'ninguno', args, mt, px, 1.0))
    return out


def estimar(cfg, salida, arqs, mt):
    """t_epoca = a + c * MACs efectivos por familia y resolución, con el piloto. Es una ESTIMACIÓN."""
    salida = Path(salida); filas = []
    for p in (salida/'piloto').glob('*/result.json'):
        r = json.loads(p.read_text(encoding='utf-8')); h = r['history'][0]
        filas.append({'familia': r['familia'], 'px': r['px'], 'macs': r['macs_efectivos'],
                      'train_s': h['train_seconds'], 'val_s': h['seconds'] - h['train_seconds']})
    modelo = {}
    for f in FAMILIAS:
        for px in sorted({x['px'] for x in filas}):
            sel = [x for x in filas if x['familia'] == f and x['px'] == px]
            if len(sel) < 2:
                raise RuntimeError(f'Piloto insuficiente en {f} a {px} px')
            m = np.array([x['macs'] for x in sel], float)
            ct, at = np.polyfit(m, [x['train_s'] for x in sel], 1)
            cv, av = np.polyfit(m, [x['val_s'] for x in sel], 1)
            modelo[f'{f}@{px}'] = {'train_a': float(at), 'train_c': float(ct), 'val_a': float(av), 'val_c': float(cv),
                                   'n': len(sel)}

    def epoca(f, arq, px, fr=1.0):
        mo = modelo[f'{f}@{px}']; m = mt[(f, arq)]*(px/256)**2
        return max(mo['train_a'] + mo['train_c']*m, .5)*fr + max(mo['val_a'] + mo['val_c']*m, .2)

    f1, f2 = cfg['fase1'], cfg['fase2']; n_r = cfg['ruido']['n_arquitecturas']
    orden = orden_fase1(cfg, arqs)
    ref = buscar(arqs, cfg['referencia']['w'], cfg['referencia']['b'])['arq']
    opciones = []
    for o in cfg['fidelidad_fase1']['opciones']:
        px, fr = o['px'], o['fraccion_train']
        h1 = sum(epoca(f, a['arq'], px, fr) for f in FAMILIAS for a in arqs)*f1['epocas']/3600
        hr = sum(epoca(f, a['arq'], px, fr) for f in FAMILIAS for a in orden[:n_r])*f1['epocas']/3600
        ha = sum(epoca(f, ref, px, fr) for f in FAMILIAS)*f1['epocas']*len(cfg['ajuste_previo']['lr'])/3600
        opciones.append({**o, 'horas_fase1': round(h1, 1), 'horas_ruido': round(hr, 1), 'horas_ajuste': round(ha, 2),
                         'horas_total_antes_de_fase2': round(h1 + hr + ha, 1)})
    fase2 = {}
    for k in (10, 20):
        n = k + 2
        por_f = {f: sorted(epoca(f, a['arq'], 256) for a in arqs) for f in FAMILIAS}
        fase2[f'k{k}'] = {
            'mediana_del_espacio': round(sum(float(np.median(v))*n for v in por_f.values())*5*f2['epocas']/3600, 1),
            'los_mas_caros': round(sum(sum(v[-n:]) for v in por_f.values())*5*f2['epocas']/3600, 1)}
    estim = {'estimado': True, 'nota': 'Piloto de una época con la caché llena, un proceso. No es una medición de la búsqueda.',
             'modelo_tiempo': modelo, 'opciones_fase1': opciones, 'horas_fase2_cota_30_epocas': fase2,
             'nota_fase2': 'Cota con 30 épocas; con paciencia 5 las compactas del benchmark promediaron 19,6 a 29,2. '
                           'Depende de qué finalistas salgan: los anchos cuestan más.',
             'nota_paralelo': 'Horas de un proceso. Varios procesos en la misma GPU bajan el reloj mientras la GPU '
                              'no esté saturada.'}
    E.atomic_json(salida/'estimacion_computo.json', estim)
    print(json.dumps(estim, indent=2, ensure_ascii=False), flush=True)
    return estim


def fijar_fidelidad(cfg, salida, horas):
    """Regla del protocolo: la primera opción (de mayor a menor fidelidad) cuya fase 1 estimada cabe."""
    archivo = Path(salida)/'fidelidad.json'
    if archivo.exists():
        previa = json.loads(archivo.read_text(encoding='utf-8'))
        if previa['presupuesto_horas_fase1'] != horas:
            raise RuntimeError('La fidelidad ya se fijó con otro presupuesto; no se cambia después')
        return previa
    if any((Path(salida)/b).exists() for b in ('ajuste', 'fase1')):
        raise RuntimeError('Ya hay corridas de ajuste o fase 1: la fidelidad no se puede fijar después')
    est = json.loads((Path(salida)/'estimacion_computo.json').read_text(encoding='utf-8'))
    caben = [o for o in est['opciones_fase1'] if o['horas_total_antes_de_fase2'] <= horas]
    elegida = caben[0] if caben else est['opciones_fase1'][-1]
    out = {'regla': cfg['fidelidad_fase1']['regla'], 'presupuesto_horas_fase1': horas,
           'opciones_estimadas': est['opciones_fase1'], 'elegida': elegida,
           'cabe_en_el_presupuesto': bool(caben),
           'desviacion': None if elegida['px'] == 256 and elegida['fraccion_train'] == 1 else
           'Fidelidad de la fase 1 reducida respecto del 20 % de épocas a resolución completa; se declara y se '
           'mide su efecto con la correlación de rangos fase 1 - fase 2.'}
    E.atomic_json(archivo, out)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return out


# ---------------------------------------------------------------- principal
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('bloque', choices=['listar', 'piloto', 'estimar', 'fidelidad', 'ajuste', 'cerrar_ajuste',
                                      'fase1', 'finalistas', 'fase2'])
    p.add_argument('--output')
    p.add_argument('--config', default=str(CONFIG))
    p.add_argument('--familias', nargs='+', choices=FAMILIAS)
    p.add_argument('--fragmento', help='i/n: este proceso corre las tareas j con j mod n = i')
    p.add_argument('--horas-fase1', type=float, help='Presupuesto de horas de GPU para ajuste + fase 1 + ruido')
    p.add_argument('--permitir-incompleta', action='store_true',
                   help='finalistas con la fase 1 cortada; queda registrado como desviación')
    # Solo pruebas locales en CPU:
    p.add_argument('--local', action='store_true')
    p.add_argument('--raiz-imagenes'); p.add_argument('--raiz-mascaras')
    p.add_argument('--limite-train', type=int); p.add_argument('--limite-val', type=int)
    p.add_argument('--epocas-max', type=int)
    p.add_argument('--n-arq', type=int, help='Solo pruebas locales: primeras n arquitecturas del orden')
    p.add_argument('--n-semillas', type=int, help='Solo pruebas locales: primeras n semillas de la fase 2')
    args = p.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.bloque == 'listar':
        listar(cfg); return
    if not args.output:
        raise ValueError('--output es obligatorio')
    proto = json.loads(C.PROTOCOLO.read_text(encoding='utf-8'))
    for f in FAMILIAS:  # la receta declarada es la del benchmark
        r = proto['models'][f]
        assert (r['optimizer'], r['lr'], r['weight_decay']) == (cfg['receta']['optimizer'], cfg['receta']['lr'],
                                                                cfg['receta']['weight_decay']), f
    assert proto['size'] == 256 and proto['batch_size'] == cfg['receta']['batch_size']
    salida = Path(args.output).resolve(); salida.mkdir(parents=True, exist_ok=True)
    familias = tuple(args.familias or FAMILIAS)
    arqs = espacio(cfg)
    if (args.n_arq or args.n_semillas) and not args.local:
        raise RuntimeError('--n-arq y --n-semillas solo en pruebas locales')
    if args.n_semillas:
        cfg['fase2']['semillas'] = cfg['fase2']['semillas'][:args.n_semillas]
    if not args.local:
        if not torch.cuda.is_available():
            raise RuntimeError('La búsqueda requiere GPU; las pruebas en CPU van con --local')
        if any([args.limite_train, args.limite_val, args.epocas_max, args.raiz_imagenes, args.raiz_mascaras]):
            raise RuntimeError('Los recortes de prueba solo se admiten con --local')
    mt = tabla_macs(cfg, arqs, familias)
    if args.bloque == 'estimar':
        estimar(cfg, salida, arqs, mt); return
    if args.bloque == 'fidelidad':
        if args.horas_fase1 is None:
            raise ValueError('--horas-fase1 es obligatorio: es el presupuesto que decide la fidelidad')
        fijar_fidelidad(cfg, salida, args.horas_fase1); return
    if args.bloque == 'cerrar_ajuste':
        cerrar_ajuste(cfg, salida, familias); return
    if args.bloque == 'finalistas':
        cerrar_finalistas(cfg, salida, familias, arqs, args.permitir_incompleta); return
    device = torch.device('cuda' if torch.cuda.is_available() and not args.local else 'cpu')
    train, val = C.filas(args)
    ident = identidad(args, train, val)
    if args.bloque == 'piloto':
        tareas = tareas_piloto(cfg, arqs, args, mt)
        for row in train + val:  # caché llena antes de medir: así corre la búsqueda en un proceso largo
            C.ejemplo(row)
    elif args.bloque == 'ajuste':
        tareas = tareas_ajuste(cfg, arqs, familias, salida, args, mt)
    elif args.bloque == 'fase1':
        tareas = tareas_fase1(cfg, arqs, familias, salida, args, mt, args.n_arq)
    else:
        tareas = tareas_fase2(cfg, arqs, familias, salida, args, mt)
    hechos = []
    for t in C.fragmento(tareas, args.fragmento):
        r = entrenar(t, salida, proto, ident, device, train, val)
        hechos.append({k: r.get(k) for k in ('id', 'best_val_group_Dice', 'epochs_completed', 'seconds_total')})
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    sello = (args.fragmento or 'todo').replace('/', 'de')
    E.atomic_json(salida/f'ultima_llamada_{args.bloque}_{sello}.json',
                  {'argv': sys.argv[1:] if argv is None else argv, 'tareas': len(hechos), 'resultados': hechos})
    if args.bloque == 'ajuste' and not args.fragmento:
        cerrar_ajuste(cfg, salida, familias)


if __name__ == '__main__':
    main()
