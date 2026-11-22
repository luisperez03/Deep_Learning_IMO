"""Prueba funcional en CPU de la búsqueda de arquitectura. NO mide calidad ni costo de GPU.

Sobre 6 imágenes de train y 4 de validación y a lo sumo 2 épocas por corrida:
  1. las compactas heredadas son idénticas a modelos_base.py con (16,32,64)/(0,0,0): parámetros,
     inicialización con la misma semilla y salida;
  2. el espacio tiene 1000 arquitecturas distintas, 125 sin dropout, y el listado versionado coincide;
  3. el dropout está donde dice el protocolo (número de capas Dropout por patrón y familia);
  4. corre piloto y estimar, la regla de fidelidad (con presupuesto 0 queda la opción C: 128 px y mitad
     de train), ajuste previo (y su regla), fase 1 con 4 arquitecturas en dos fragmentos (más la
     repetición de ruido), finalistas, fase 2 con 2 semillas a 256 px y todo train, el resumidor;
  5. una corrida de la fase 2 cortada tras la época 1 se reanuda desde last.pt y termina igual que
     una corrida sin corte;
  6. rehacer un bloque terminado no reentrena (lee result.json).

Uso:
  python scripts/prueba_cpu_busqueda_arquitectura_e2.py --raiz-imagenes <carpeta con VOC2012> --trabajo <carpeta temporal>
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import busqueda_arquitectura_e2 as B  # noqa: E402
import resumir_busqueda_arquitectura_e2 as R  # noqa: E402
from modelos_base import MODELS  # noqa: E402

PROJECT = B.PROJECT


def comprobar(cond, texto, informe):
    informe['comprobaciones'].append({'que': texto, 'ok': bool(cond)})
    print(('OK  ' if cond else 'FALLA ') + texto, flush=True)
    if not cond:
        raise AssertionError(texto)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raiz-imagenes', required=True)
    p.add_argument('--trabajo', required=True)
    p.add_argument('--informe', default=str(PROJECT/'resultados/busqueda_arquitectura_e2/prueba_cpu.json'))
    args = p.parse_args()
    trabajo = Path(args.trabajo).resolve()
    if trabajo.exists():
        shutil.rmtree(trabajo)
    trabajo.mkdir(parents=True)
    t0 = time.perf_counter()
    informe = {'que_es': 'Prueba funcional en CPU; ningún número es un resultado de la búsqueda',
               'fecha': time.strftime('%Y-%m-%d %H:%M'), 'torch': torch.__version__, 'comprobaciones': []}
    cfg = json.loads(B.CONFIG.read_text(encoding='utf-8'))

    # 1. Heredadas idénticas
    x = torch.randn(2, 3, 64, 64)
    for f in B.FAMILIAS:
        torch.manual_seed(11); a = MODELS[f]()
        torch.manual_seed(11); b = B.construir_red(f, cfg['heredada']['w'], cfg['heredada']['b'], 0.2)
        iguales = all(torch.equal(u, v) for u, v in zip(a.parameters(), b.parameters()))
        comprobar(B.parametros(a) == B.parametros(b) and iguales and torch.equal(a(x), b(x)),
                  f'{f}: (16,32,64)/(0,0,0) idéntica a modelos_base con la misma semilla', informe)

    # 2. Espacio y listado
    arqs = B.espacio(cfg)
    comprobar(len({(a['w'], a['b']) for a in arqs}) == 1000, '1000 arquitecturas distintas', informe)
    comprobar(sum(a['b'] == (0, 0, 0) for a in arqs) == 125, '125 sin dropout y 875 con al menos uno', informe)
    lineas = B.LISTADO.read_text(encoding='utf-8').splitlines()
    muestra = [arqs[i] for i in (0, 337, 999)]
    ok = True
    for a in muestra:
        fila = dict(zip(lineas[0].split(','), lineas[1 + int(a['arq'][1:])].split(',')))
        for f in B.FAMILIAS:
            m = B.construir_red(f, a['w'], a['b'], 0.2)
            ok &= int(fila[f'{f}_parametros']) == B.parametros(m) and int(fila[f'{f}_macs']) == B.macs(m)
    comprobar(len(lineas) == 1001 and ok, 'listado versionado coincide con las redes (3 filas muestreadas)', informe)

    # 3. Posiciones de dropout
    esperado = {'fcn': lambda b: sum(b), 'unet': lambda b: sum(b) + b[0] + b[1], 'segnet': lambda b: sum(b) + b[0] + b[1]}
    ok = all(sum(isinstance(m, torch.nn.Dropout) for m in B.construir_red(f, (16, 16, 16), b, 0.2).modules())
             == esperado[f](b) for f in B.FAMILIAS for b in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)])
    comprobar(ok, 'dropout en el codificador y su espejo (cuello sin espejo)', informe)

    # 4. Bloques
    mascaras = PROJECT/'resultados/preentrenamiento/masks_binary'
    comunes = ['--output', str(trabajo), '--local', '--raiz-imagenes', args.raiz_imagenes,
               '--raiz-mascaras', str(mascaras), '--limite-train', '6', '--limite-val', '4', '--epocas-max', '2']
    B.main(['piloto', *comunes])
    B.main(['estimar', '--output', str(trabajo), '--local'])
    est = json.loads((trabajo/'estimacion_computo.json').read_text(encoding='utf-8'))
    comprobar(est['estimado'] and len(est['modelo_tiempo']) == 6 and len(est['opciones_fase1']) == 3,
              'piloto a 256 y 128 px y estimación de las tres opciones de fidelidad', informe)
    # Presupuesto 0: ninguna opción cabe y queda la C (128 px, media train): ejercita la fidelidad reducida.
    B.main(['fidelidad', '--output', str(trabajo), '--local', '--horas-fase1', '0'])
    fid = json.loads((trabajo/'fidelidad.json').read_text(encoding='utf-8'))
    comprobar(fid['elegida']['nombre'] == 'C' and not fid['cabe_en_el_presupuesto'],
              'regla de fidelidad: sin presupuesto queda C y se registra', informe)
    sub = B.subconjunto_train(B.C.filas(argparse.Namespace(local=True, raiz_imagenes=args.raiz_imagenes,
                              raiz_mascaras=str(mascaras), limite_train=None, limite_val=None))[0], .5, 20261123)
    comprobar(500 <= len(sub) <= 540, f'mitad de train por grupos: {len(sub)} de 1015', informe)
    red = B.EntradaReducida(B.construir_red('unet', (16, 16, 16), (0, 0, 0), .2), 128)
    comprobar(red(torch.randn(2, 3, 256, 256)).shape == (2, 1, 256, 256), 'entrada a 128 px, salida a 256', informe)
    B.main(['ajuste', *comunes])
    aj = json.loads((trabajo/'ajuste_previo.json').read_text(encoding='utf-8'))
    regla_ok = True
    for f in B.FAMILIAS:
        d = {float(k): v for k, v in aj['dice_validacion'][f].items()}; m = max(d, key=d.get)
        regla_ok &= aj['lr_elegida'][f] == (m if d[m] - d[0.001] >= 0.005 else 0.001)
    comprobar(regla_ok, 'ajuste previo: 3 tasas por familia y regla aplicada', informe)
    B.main(['fase1', *comunes, '--n-arq', '4', '--fragmento', '0/2'])
    B.main(['fase1', *comunes, '--n-arq', '4', '--fragmento', '1/2'])
    f1 = R.leer(trabajo/'fase1')
    comprobar(len(f1) == 4*3 + 2*3 and all(r['px'] == 128 and r['n_train'] < 6 for r in f1),
              'fase 1: 4 arquitecturas x 3 familias + repetición de ruido, en dos fragmentos, con la fidelidad C', informe)
    t1 = time.perf_counter(); B.main(['fase1', *comunes, '--n-arq', '4']); rehacer = time.perf_counter() - t1
    comprobar(rehacer < 5, f'rehacer la fase 1 terminada no reentrena ({rehacer:.1f} s)', informe)
    B.main(['finalistas', '--output', str(trabajo), '--local', '--permitir-incompleta'])
    fin = json.loads((trabajo/'finalistas.json').read_text(encoding='utf-8'))
    comprobar(all(fin['referencia'] in fin['familias'][f]['fase2'] and fin['heredada'] in fin['familias'][f]['fase2']
                  for f in B.FAMILIAS) and fin['k'] in (10, 20), 'finalistas con k y los dos controles', informe)

    # 5. Reanudación de la fase 2: cortar una corrida tras la época 1 y seguir
    proto = json.loads((PROJECT/'config/experimento_e2.json').read_text(encoding='utf-8'))
    ns = argparse.Namespace(local=True, raiz_imagenes=args.raiz_imagenes, raiz_mascaras=str(mascaras),
                            limite_train=6, limite_val=4, epocas_max=2)
    train, val = B.C.filas(ns); ident = B.identidad(ns, train, val)
    mt = B.tabla_macs(cfg, arqs, B.FAMILIAS); cfg2 = json.loads(json.dumps(cfg)); cfg2['fase2']['semillas'] = [11, 23]
    t = B.tareas_fase2(cfg2, arqs, B.FAMILIAS, trabajo, ns, mt)[0]
    corte = B.entrenar(t, trabajo, proto, ident, torch.device('cpu'), train, val, detener_tras=1)
    hay_last = (trabajo/'fase2'/t['id']/'last.pt').exists()
    seguida = B.entrenar(t, trabajo, proto, ident, torch.device('cpu'), train, val)
    t_ref = dict(t); t_ref['id'] = t['id'] + '_sin_corte'
    sin_corte = B.entrenar(t_ref, trabajo/'control', proto, ident, torch.device('cpu'), train, val)
    comprobar(not corte['completed'] and hay_last and seguida['epochs_completed'] == 2
              and abs(seguida['best_val_group_Dice'] - sin_corte['best_val_group_Dice']) < 1e-6
              and abs(seguida['history'][-1]['train_loss'] - sin_corte['history'][-1]['train_loss']) < 1e-5,
              'fase 2 cortada tras la época 1 se reanuda y termina igual que sin corte', informe)
    B.main(['fase2', *comunes, '--n-semillas', '2'])
    f2 = R.leer(trabajo/'fase2')
    n_esp = 2*sum(len(fin['familias'][f]['fase2']) for f in B.FAMILIAS)
    comprobar(len(f2) == n_esp and all('brecha_train_val' in r and (r['_dir']/'best.pt').exists()
                                       and r['px'] == 256 and r['n_train'] == 6 for r in f2),
              f'fase 2: {n_esp} corridas con brecha train-val y mejor checkpoint', informe)

    # 6. Resumen
    R.main(['--output', str(trabajo)])
    res = json.loads((trabajo/'RESUMEN.json').read_text(encoding='utf-8'))
    comprobar(all('fase2' in res['familias'][f] and 'ganador' in res['familias'][f]['fase2'] for f in B.FAMILIAS),
              'resumidor: ganador por familia y comparación con la heredada', informe)
    informe['segundos'] = round(time.perf_counter() - t0, 1)
    informe['ganadores_de_prueba_sin_valor'] = {f: res['familias'][f]['fase2']['ganador']['arq'] for f in B.FAMILIAS}
    Path(args.informe).parent.mkdir(parents=True, exist_ok=True)
    Path(args.informe).write_text(json.dumps(informe, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Prueba completa en {informe["segundos"]} s; informe en {args.informe}')


if __name__ == '__main__':
    main()
