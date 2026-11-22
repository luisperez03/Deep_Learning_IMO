"""Atajos y taxonomía exacta sobre los 50 checkpoints cerrados, en validación (checklist 13, 65 y 67). Exploratorio.

Protocolo a priori: config/errores_atajos_e2_20261122.json (sección shortcuts). Por corrida y por imagen de validación:
  C0 limpia      Dice, TP/FP/FN, HD95 y componentes predichas: píxeles de exceso en componentes sin contacto con la
                 máscara real dilatada 5 px (exceso distante) y rótulos reales sin ningún píxel predicho
  C1 sin contexto  fuera de las cajas de los rótulos (+15 %, mínimo 8 px) el valor medio de train (cero normalizado)
  C2 sin rótulo    rótulo (+3 px) relleno con el píxel exterior más cercano y suavizado (sigma 3): negativo sintético
  C3 gris          luminancia en los tres canales
Guarda <modelo>_seed<s>_atajos.json y, para el top 3, la predicción limpia binaria de validación
(<modelo>_seed<s>_mascaras_val.npz, bits empaquetados) para los paneles de revisión humana.
Sin optimizador ni cambio de pesos; no abre el test. Reanudable por firma, como diagnosticos_gpu_e2.py.

  python scripts/atajos_gpu_e2.py --runs RUTA/tesis_imo_e2_arquitecturas_reales_v2 --output RUTA/salida
  python scripts/atajos_gpu_e2.py --output CARPETA --humo-cpu --limit 3 --data-root RUTA   (prueba: modelos SIN entrenar)
"""
import argparse, gc, json, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi
import diagnosticos_gpu_e2 as D
from dataset_ready import partition, load_example, metrics
from metricas_diagnostico_e2 import border_metrics
from experimento_e2_colab import PROJECT, sha, digest, atomic_json, environment, setup_seed

CFG = json.loads((PROJECT / 'config/errores_atajos_e2_20261122.json').read_text(encoding='utf-8'))
SC = CFG['shortcuts']
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def a_rgb(x, norm):
    return x.transpose(1, 2, 0) * np.asarray(norm['std'], np.float32) + np.asarray(norm['mean'], np.float32)


def a_x(rgb, vm, norm):
    out = (rgb - np.asarray(norm['mean'], np.float32)) / np.maximum(np.asarray(norm['std'], np.float32), 1e-8)
    out[~vm] = 0
    return out.transpose(2, 0, 1).astype(np.float32)


def sin_contexto(x, y, vm):
    keep = np.zeros_like(vm)
    lab, n = ndi.label(y, structure=np.ones((3, 3)))
    for sl in ndi.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        mh = max(SC['box_margin_min_px'], round(SC['box_margin_fraction'] * h))
        mw = max(SC['box_margin_min_px'], round(SC['box_margin_fraction'] * w))
        keep[max(0, sl[0].start - mh):sl[0].stop + mh, max(0, sl[1].start - mw):sl[1].stop + mw] = True
    out = x.copy(); out[:, ~(keep & vm)] = 0
    return out


def sin_rotulo(x, y, vm, norm):
    zona = ndi.binary_dilation(y.astype(bool), iterations=SC['erase_dilation_px']) & vm
    rgb = a_rgb(x, norm)
    if zona.any() and (vm & ~zona).any():
        _, idx = ndi.distance_transform_edt(zona | ~vm, return_indices=True)
        lleno = rgb[idx[0], idx[1]]
        suave = ndi.gaussian_filter(lleno, sigma=(SC['erase_blur_sigma'], SC['erase_blur_sigma'], 0))
        rgb = rgb.copy(); rgb[zona] = suave[zona]
    return a_x(np.clip(rgb, 0, 1), vm, norm), zona


def gris(x, vm, norm):
    rgb = a_rgb(x, norm); l = np.clip(rgb, 0, 1) @ LUMA
    return a_x(np.repeat(l[..., None], 3, axis=2), vm, norm)


def componentes(pred, y):
    yb = y.astype(bool); cerca = ndi.binary_dilation(yb, iterations=5)
    lab, n = ndi.label(pred, structure=np.ones((3, 3)))
    toca = set(np.unique(lab[cerca & pred])) - {0}
    distantes = int(sum(((lab == k).sum() for k in range(1, n + 1) if k not in toca), 0)) if n else 0
    gl, gn = ndi.label(yb, structure=np.ones((3, 3)))
    omit = sum(1 for k in range(1, gn + 1) if not pred[gl == k].any())
    return {'pred_componentes': int(n), 'fp_px_distantes': distantes, 'gt_componentes': int(gn), 'gt_omitidas': int(omit)}


def correr(model, run, out, sig, norm, data_root, limit):
    dest = out / f"{run['model']}_seed{run['seed']}_atajos.json"
    mfile = out / f"{run['model']}_seed{run['seed']}_mascaras_val.npz"
    if D.valid(dest, sig):
        r = D.read(dest)
        if r.get('masks_sha256') and (not mfile.exists() or sha(mfile) != r['masks_sha256']):
            raise RuntimeError('Máscaras corruptas: ' + mfile.name)
        return
    rows = partition('val')[:limit] if limit else partition('val')
    det, bits, t0 = [], [], time.perf_counter()
    for i, row in enumerate(rows):
        x, y, vm = load_example(row, data_root)
        p0 = D.infer(model, x); pred = (p0 >= .5) & vm
        d = {'id': row['id'], 'group_id': row['group_id'], 'Dice_C0': metrics(p0, y, vm)['Dice']}
        b = border_metrics(p0, y, vm)
        d.update({k: b[k] for k in ['TP', 'FP', 'FN', 'HD95', 'empty_prediction']}, **componentes(pred, y))
        d['Dice_C1'] = metrics(D.infer(model, sin_contexto(x, y, vm)), y, vm)['Dice']
        d['Dice_C3'] = metrics(D.infer(model, gris(x, vm, norm)), y, vm)['Dice']
        x2, zona = sin_rotulo(x, y, vm, norm); p2 = (D.infer(model, x2) >= .5) & vm
        d['C2_fraccion_zona_predicha'] = float(p2[zona].mean()) if zona.any() else 0.0
        d['C2_pixeles_predichos'] = int(p2.sum())
        d['C2_falsa_alarma'] = int(p2.sum() >= 50)
        det.append(d)
        if run['model'] in CFG['top3_by_validation']:
            bits.append(np.packbits(pred.ravel()))
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(run['model'], run['seed'], i + 1, '/', len(rows), flush=True)
    r = {'model': run['model'], 'seed': run['seed'], 'checkpoint_sha256': run.get('best_sha256'), 'details': det,
         'images': len(det), 'seconds': time.perf_counter() - t0}
    if bits:
        D.atomic_npz(mfile, ids=np.array([d['id'] for d in det]), bits=np.stack(bits)); r['masks_sha256'] = sha(mfile)
    D.write(dest, r, sig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs'); ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--models', nargs='*'); ap.add_argument('--seeds', nargs='*', type=int)
    ap.add_argument('--limit', type=int, help='imágenes de validación (solo prueba; cambia la firma)')
    ap.add_argument('--data-root', help='carpeta que contiene VOC2012/ (por defecto, la del paquete)')
    ap.add_argument('--humo-cpu', action='store_true', help='modelos sin entrenar en CPU: prueba del flujo, no de los números')
    a = ap.parse_args(); out = a.output; out.mkdir(parents=True, exist_ok=True)
    protocol = D.read(PROJECT / 'config/protocolo_test_e2_cerrado.json')
    norm = D.read(PROJECT / 'resultados/preentrenamiento/normalization_train.json')
    if a.humo_cpu:
        from modelos_e2 import MODELS_E2
        D.DEVICE = 'cpu'
        runs = [{'model': m, 'seed': 11, 'folder': f'{m}_seed11'} for m in (a.models or ['fcn', 'unet'])]
        sig = digest({'humo_cpu_sin_entrenar': True, 'config': CFG, 'limit': a.limit, 'script': sha(Path(__file__))})
        for run in runs:
            torch.manual_seed(0); model = MODELS_E2[run['model']]().eval()
            correr(model, run, out, sig, norm, a.data_root, a.limit)
        print('Humo CPU terminado (números sin significado):', out); return
    if not torch.cuda.is_available():
        raise RuntimeError('Se requiere GPU CUDA')
    if not a.runs:
        raise SystemExit('Falta --runs')
    from evaluar_e2_cerrado import verify_inputs
    t0 = time.perf_counter(); print('Verificando datos y 50 checkpoints...', flush=True)
    verify_inputs(protocol, Path(a.runs))
    for row in partition('val'):
        if sha(PROJECT.parent / row['image_path']) != row['image_sha256'] or sha(PROJECT / row['binary_mask_path']) != row['binary_mask_sha256']:
            raise RuntimeError('Datos de validación alterados: ' + row['id'])
    env = environment()
    if env['packages'] != protocol['training_environment']['packages']:
        raise RuntimeError(f"Software principal diferente al entrenamiento: {env['packages']}")
    sig = digest({'config': CFG, 'closed_protocol': protocol, 'environment': env, 'script': sha(Path(__file__)),
                  'metrics': sha(PROJECT / 'scripts/metricas_diagnostico_e2.py'), 'limit': a.limit})
    lock = out / 'PROTOCOLO_ATAJOS.json'
    if lock.exists():
        if D.read(lock)['signature'] != sig:
            raise RuntimeError('Firma distinta a la de la carpeta (GPU, paquete o --limit): usa la misma GPU o otra carpeta')
    else:
        atomic_json(lock, {'signature': sig, 'config': CFG, 'environment': env, 'limit': a.limit})
    runs = [r for r in protocol['runs'] if (not a.models or r['model'] in a.models) and (not a.seeds or r['seed'] in a.seeds)]
    log = {'device': env['device'], 'verification_seconds': time.perf_counter() - t0, 'runs': []}
    for run in runs:
        setup_seed(run['seed']); t1 = time.perf_counter(); model = D.load_model(run, a.runs)
        correr(model, run, out, sig, norm, None, a.limit); D.sync()
        log['runs'].append({'folder': run['folder'], 'seconds': time.perf_counter() - t1})
        del model; gc.collect(); torch.cuda.empty_cache()
    (out / 'tiempos').mkdir(exist_ok=True)
    atomic_json(out / 'tiempos' / f'atajos_{time.strftime("%Y%m%dT%H%M%S")}.json', log)
    print('Atajos terminados:', len(runs), 'corridas', flush=True)


if __name__ == '__main__':
    main()
