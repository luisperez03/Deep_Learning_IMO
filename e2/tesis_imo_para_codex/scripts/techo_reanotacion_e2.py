"""Techo de desempeño por reanotación independiente y γ de las 10 redes (checklist 63; guía §12.1). Exploratorio.

  python scripts/techo_reanotacion_e2.py preparar --voc RUTA/VOC2012 --destino CARPETA [--muestra muestra.csv]
  python scripts/techo_reanotacion_e2.py medir    --voc RUTA/VOC2012 --destino CARPETA
  python scripts/techo_reanotacion_e2.py medir    --voc RUTA/VOC2012 --simular        (prueba del cálculo)
  python scripts/techo_reanotacion_e2.py gamma    [--techo resultados/techo_e2/techo_reanotacion.json]

preparar  50 imágenes de VALIDACIÓN (nunca test), una por grupo, estratificadas por tercil del área etiquetada media
          del grupo (a 256 con letterbox), semilla 20261117. Copia a CARPETA/imagenes/ solo las fotos, sin máscara, y
          escribe CARPETA/muestra.csv y CARPETA/LEEME.md. Con --muestra, reutiliza una muestra ya sorteada y la valida.
medir     Lee CARPETA/reanotaciones/ (<id>.png binario del tamaño de la foto, o <id>.json de LabelMe; los rótulos
          cortados por el borde van aparte: etiqueta `cortado` en LabelMe o <id>_cortado.png) y calcula Dice e IoU
          contra la máscara del conjunto con la misma geometría y métrica que los modelos (letterbox 256, zona válida,
          `dataset_ready.metrics`). Media por grupo con IC 95 % bootstrap (5.000), por tercil, y con/sin cortados.
          Con --simular fabrica reanotaciones perturbadas (erosión/dilatación de 1-2 px) para probar el cálculo.
gamma     γ_k = (s_techo − s_k)/(s_techo − s_base) para cada red k (guía, p. 49), con s_k el Dice de validación (misma
          partición que el techo) y, aparte, el de test; s_base el clásico HistGradientBoosting elegido en
          validación. Sin techo medido, NO inventa uno: escribe γ como función de valores hipotéticos de s_techo,
          rotulados como tales, para ver a partir de qué techo la ventaja restante sería marginal.

Escribe en resultados/techo_e2/. No copia imágenes al repositorio.
"""
import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_ready import metrics  # noqa: E402
from inicio_voc import palette  # noqa: E402
from preparar_base import PROJECT, binary_mask, letterbox, read_csv  # noqa: E402

OUT = PROJECT / 'resultados/techo_e2'
SEMILLA = 20261117
TAM = 256
HIPOTETICOS = [0.88, 0.90, 0.92, 0.95, 0.98, 1.0]
LEEME = """# Reanotación para el techo de desempeño

Hay {n} fotografías de validación en `imagenes/`. Para cada una, marcar **todos los rótulos de mercancía peligrosa**
(los rombos de las 13 clases), sin mirar la máscara original. Solo importa rótulo contra fondo; no hace falta la clase.

Cómo marcar:
- Con LabelMe: un polígono por rótulo, siguiendo el borde exterior del rombo. Guardar `<id>.json` en `reanotaciones/`.
- O con cualquier editor: un PNG del mismo tamaño que la foto, negro con los rótulos en blanco, `<id>.png`.

Criterio: el rótulo completo hasta su borde exterior; si está tapado, solo la parte visible. Sin rótulos visibles,
imagen vacía (PNG negro o JSON sin polígonos).

**Rótulos cortados por el borde de la foto: marcarlos APARTE** (etiqueta `cortado` en LabelMe, o `<id>_cortado.png`).
El conjunto original los anota a veces y a veces no; separarlos evita confundir esa inconsistencia con desacuerdo.

Quien reanota no debe haber visto las máscaras de estas imágenes ni las predicciones de los modelos sobre ellas.
Al terminar: `python scripts/techo_reanotacion_e2.py medir --voc ... --destino <esta carpeta>`.
"""


def leer(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def mascara_original(voc, fila, colores):
    with Image.open(Path(voc) / fila['original_mask_path'].split('VOC2012/', 1)[1]) as m:
        return binary_mask(m, colores).astype(bool)


def a_256(b):
    m, v = letterbox(Image.fromarray(b.astype(np.uint8)), TAM, True)
    return np.asarray(m) > 0, v


def sortear(voc, n):
    colores = palette(Path(voc) / 'labelmap.txt')
    filas = read_csv(PROJECT / 'resultados/preentrenamiento/val.csv')
    area = {}
    for f in filas:
        y, v = a_256(mascara_original(voc, f, colores))
        area.setdefault(f['group_id'], []).append((f, int(y[v].sum())))
    grupos = sorted(area)
    medias = np.array([np.mean([x[1] for x in area[g]]) for g in grupos])
    tercil = np.digitize(medias, np.quantile(medias, [1 / 3, 2 / 3]))
    rng = np.random.default_rng(SEMILLA)
    elegidas = []
    for t in range(3):
        cand = [g for g, k in zip(grupos, tercil) if k == t]
        for g in rng.choice(cand, size=min(n // 3 + (1 if t < n % 3 else 0), len(cand)), replace=False):
            f, _ = area[g][rng.integers(len(area[g]))]
            elegidas.append({'id': f['id'], 'group_id': g, 'tercil_area': str(t + 1)})
    return sorted(elegidas, key=lambda x: x['id'])


def preparar(a):
    D = Path(a.destino)
    (D / 'imagenes').mkdir(parents=True, exist_ok=True); (D / 'reanotaciones').mkdir(exist_ok=True)
    elegidas = sortear(a.voc, a.n)
    if a.muestra:
        with open(a.muestra, encoding='utf-8-sig') as f:
            previa = list(csv.DictReader(f))
        if [(x['id'], x['group_id'], x['tercil_area']) for x in previa] != [(x['id'], x['group_id'], x['tercil_area']) for x in elegidas]:
            raise SystemExit('La muestra dada no coincide con el sorteo (semilla 20261117); no se reutiliza.')
    val = {f['id']: f['group_id'] for f in read_csv(PROJECT / 'resultados/preentrenamiento/val.csv')}
    assert all(e['id'] in val for e in elegidas) and len({e['group_id'] for e in elegidas}) == len(elegidas)
    for e in elegidas:
        shutil.copy2(Path(a.voc) / 'JPEGImages' / f"{e['id']}.jpg", D / 'imagenes' / f"{e['id']}.jpg")
    for destino in (D / 'muestra.csv', OUT / 'muestra.csv'):
        destino.parent.mkdir(parents=True, exist_ok=True)
        with open(destino, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, ['id', 'group_id', 'tercil_area']); w.writeheader(); w.writerows(elegidas)
    (D / 'LEEME.md').write_text(LEEME.format(n=len(elegidas)), encoding='utf-8')
    print(f'{len(elegidas)} imágenes de validación en {D / "imagenes"}; muestra en {OUT / "muestra.csv"}')


def reanotacion(D, i, tam, con_cortados=False):
    png, js = D / 'reanotaciones' / f'{i}.png', D / 'reanotaciones' / f'{i}.json'
    if png.exists():
        m = np.asarray(Image.open(png).convert('L')) > 127
        if m.shape != tam:
            raise ValueError(f'{i}: la reanotación mide {m.shape} y la foto {tam}')
        extra = D / 'reanotaciones' / f'{i}_cortado.png'
        if con_cortados and extra.exists():
            m = m | (np.asarray(Image.open(extra).convert('L')) > 127)
        return m
    if js.exists():
        import cv2
        d = leer(js); m = np.zeros(tam, np.uint8)
        for s in d.get('shapes', []):
            if s.get('label', '').strip().lower() == 'cortado' and not con_cortados:
                continue
            pts = np.round(np.array(s['points'], float)).astype(np.int32)
            if s.get('shape_type', 'polygon') == 'rectangle':
                cv2.rectangle(m, tuple(pts[0]), tuple(pts[1]), 1, -1)
            else:
                cv2.fillPoly(m, [pts], 1)
        return m.astype(bool)
    return None


def medir_carpeta(voc, D):
    colores = palette(Path(voc) / 'labelmap.txt')
    por_id = {f['id']: f for f in read_csv(PROJECT / 'resultados/preentrenamiento/val.csv')}
    with open(D / 'muestra.csv', encoding='utf-8-sig') as f:
        muestra = list(csv.DictReader(f))
    filas, faltan = [], []
    for e in muestra:
        orig = mascara_original(voc, por_id[e['id']], colores)
        nueva = reanotacion(D, e['id'], orig.shape)
        if nueva is None:
            faltan.append(e['id']); continue
        yo, v = a_256(orig)
        fila = dict(e)
        for suf, m in (('', nueva), ('_con_cortados', reanotacion(D, e['id'], orig.shape, True))):
            yn, _ = a_256(m)
            r = metrics(yn.astype(np.float32), yo, v)
            fila['Dice' + suf], fila['IoU' + suf] = r['Dice'], r['IoU']
        filas.append(fila)
    if not filas:
        raise SystemExit(f'No hay reanotaciones en {D / "reanotaciones"}')
    grupos = sorted({f['group_id'] for f in filas})
    d = np.array([np.mean([f['Dice'] for f in filas if f['group_id'] == g]) for g in grupos])
    rng = np.random.default_rng(SEMILLA)
    bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return {'imagenes_medidas': len(filas), 'faltan': faltan, 'Dice_techo': float(d.mean()),
            'IC95_bootstrap_grupos': np.quantile(bs, [.025, .975]).tolist(),
            'Dice_mediano': float(np.median(d)), 'IoU_medio': float(np.mean([f['IoU'] for f in filas])),
            'por_tercil_area': {t: float(np.mean([f['Dice'] for f in filas if f['tercil_area'] == t]))
                                for t in ('1', '2', '3') if any(f['tercil_area'] == t for f in filas)},
            'Dice_con_cortados': float(np.mean([f['Dice_con_cortados'] for f in filas])),
            'menor_acuerdo': sorted(filas, key=lambda f: f['Dice'])[:5], 'por_imagen': filas}


def medir(a):
    if a.simular:
        from scipy import ndimage as ndi
        colores = palette(Path(a.voc) / 'labelmap.txt')
        val = {f['id']: f for f in read_csv(PROJECT / 'resultados/preentrenamiento/val.csv')}
        with tempfile.TemporaryDirectory() as tmp:
            D = Path(tmp); (D / 'reanotaciones').mkdir()
            elegidas = sortear(a.voc, a.n)[:12]
            with open(D / 'muestra.csv', 'w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, ['id', 'group_id', 'tercil_area']); w.writeheader(); w.writerows(elegidas)
            for k, e in enumerate(elegidas):
                y = mascara_original(a.voc, val[e['id']], colores)
                y = ndi.binary_dilation(y, iterations=2) if k % 2 else ndi.binary_erosion(y, iterations=2)
                Image.fromarray((y * 255).astype(np.uint8)).save(D / 'reanotaciones' / f"{e['id']}.png")
            res = medir_carpeta(a.voc, D)
        assert res['imagenes_medidas'] == 12 and 0.5 < res['Dice_techo'] < 1.0, res['Dice_techo']
        print('Prueba simulada correcta: Dice entre reanotación perturbada y original =', round(res['Dice_techo'], 4))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'prueba_simulada_medir.json').write_text(json.dumps(
            {'nota': 'Reanotaciones fabricadas por erosión/dilatación de 2 px de la máscara original; prueba del cálculo, '
                     'no es un techo.', 'imagenes': 12, 'Dice': res['Dice_techo'], 'IC95': res['IC95_bootstrap_grupos']},
            indent=1, ensure_ascii=False), encoding='utf-8')
        return
    res = medir_carpeta(a.voc, Path(a.destino))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'techo_reanotacion.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k not in ('por_imagen', 'menor_acuerdo')}, indent=1, ensure_ascii=False))


def gamma(a):
    val = {r['model']: r['Dice_mean'] for r in leer(PROJECT / 'resultados/auditoria_e2_recibida/auditoria.json')['summary']}
    test = {r['model']: r['Dice_test'] for r in leer(PROJECT / 'resultados/auditoria_test_e2/AUDITORIA.json')['table']}
    sel = leer(PROJECT / 'resultados/baseline_clasico_e2/SELECCION_ANTES_TEST.json')
    base = {'val': sel['validation_means'][str(sel['chosen_max_leaf_nodes'])],
            'test': leer(PROJECT / 'resultados/baseline_clasico_e2/RESUMEN.json')['test_Dice']}
    s = {'val': val, 'test': test}

    def tabla(techo):
        return [{'modelo': m, **{f'gamma_{p}': (techo - s[p][m]) / (techo - base[p]) if techo > base[p] else None
                                 for p in ('val', 'test')}} for m in sorted(test, key=lambda m: -test[m])]
    R = {'definicion': 'γ = (s_techo − s_k)/(s_techo − s_base), guía p. 49: fracción del margen total aún sin cubrir',
         's_base': {'modelo': f"HistGradientBoosting, {sel['chosen_max_leaf_nodes']} hojas (elegido en validación)", **base},
         's_k': {'val': val, 'test': test}}
    techo = Path(a.techo) if a.techo else OUT / 'techo_reanotacion.json'
    if techo.exists():
        t = leer(techo)
        R['techo_medido'] = {'Dice': t['Dice_techo'], 'IC95': t['IC95_bootstrap_grupos'], 'imagenes': t['imagenes_medidas'],
                             'particion': 'validación (50 imágenes, una por grupo)'}
        R['gamma_con_techo_medido'] = tabla(t['Dice_techo'])
        R['gamma_extremos_IC'] = {f'{q:.4f}': tabla(q) for q in t['IC95_bootstrap_grupos']}
        R['comparabilidad'] = ('El techo se mide en 50 imágenes de validación y s_k es el Dice de validación de las '
                               '218 imágenes; γ_test mezcla particiones y es solo orientativo.')
    else:
        R['techo_medido'] = None
        R['aviso'] = ('NO HAY TECHO MEDIDO. Las filas siguientes son γ para valores HIPOTÉTICOS de s_techo, no '
                      'estimaciones: muestran a partir de qué techo el margen restante sería pequeño. No reportar '
                      'ninguna como techo del problema.')
        R['gamma_hipotetico'] = {f'{h:.2f}': tabla(h) for h in HIPOTETICOS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'gamma.json').write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding='utf-8')
    filas = R.get('gamma_con_techo_medido') or R['gamma_hipotetico']['0.95']
    print('techo medido' if R['techo_medido'] else 'HIPOTÉTICO s_techo=0,95', [(f['modelo'], round(f['gamma_test'], 3)) for f in filas])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('accion', choices=['preparar', 'medir', 'gamma'])
    ap.add_argument('--voc'); ap.add_argument('--destino'); ap.add_argument('--muestra')
    ap.add_argument('--techo'); ap.add_argument('--n', type=int, default=50); ap.add_argument('--simular', action='store_true')
    a = ap.parse_args()
    {'preparar': preparar, 'medir': medir, 'gamma': gamma}[a.accion](a)
