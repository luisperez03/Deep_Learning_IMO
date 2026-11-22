"""Taxonomía de errores, subgrupos, daño potencial y atajos (checklist 13, 65 y 67). Exploratorio.

Protocolo a priori: resultados/errores_atajos_e2/PROTOCOLO_ERRORES_ATAJOS_20261122.md y
config/errores_atajos_e2_20261122.json (su SHA-256 queda en cada salida).

  python scripts/errores_atajos_e2.py atributos --voc RUTA/VOC2012
  python scripts/errores_atajos_e2.py extraer   --diagnosticos RUTA/diag_descarga   (los *_clean.json del complemento GPU)
  python scripts/errores_atajos_e2.py analizar
  python scripts/errores_atajos_e2.py atajos    --entrada RUTA/atajos_errores_e2     (JSON del cuaderno Atajos_Errores_E2)
  python scripts/errores_atajos_e2.py revision  --entrada RUTA/atajos_errores_e2 --voc RUTA/VOC2012 --destino CARPETA
  python scripts/errores_atajos_e2.py medir-revision --hoja CARPETA/hoja_revision_errores.csv

`atributos`, `extraer` y `analizar` son locales: solo usan la imagen, la máscara real y los conteos TP/FP/FN/HD95 por
imagen. `atajos` y `revision` necesitan las predicciones del cuaderno de Colab. Los paneles de revisión llevan
imágenes del conjunto y van a una carpeta fuera del repositorio.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preparar_base import PROJECT, read_csv  # noqa: E402

OUT = PROJECT / 'resultados/errores_atajos_e2'
CFG_PATH = PROJECT / 'config/errores_atajos_e2_20261122.json'
CFG = json.loads(CFG_PATH.read_text(encoding='utf-8'))
ATRIB = OUT / 'atributos_por_imagen.csv'
CORRIDAS = OUT / 'por_imagen_corridas.csv'
B = CFG['bootstrap']['draws']
ERR = 0.80


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def firma():
    return {'config': str(CFG_PATH.relative_to(PROJECT)).replace('\\', '/'), 'config_sha256': sha_file(CFG_PATH),
            'script_sha256': sha_file(Path(__file__))}


def escribir_csv(path, filas, campos=None):
    campos = campos or list(filas[0])
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def leer_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def wilson(k, n, z=1.959964):
    if n == 0:
        return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - r) / d, 4), round((c + r) / d, 4)]


def r4(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 4)


# ---------------------------------------------------------------- atributos (imagen y máscara real, sin predicción)

def colores_clase(voc):
    clases = {}
    for linea in (Path(voc) / 'labelmap.txt').read_text(encoding='utf-8').splitlines():
        if linea.startswith('#') or not linea.strip():
            continue
        nombre, color = linea.split(':')[:2]
        if nombre != 'background':
            clases[tuple(int(c) for c in color.split(','))] = nombre
    return clases


def atributos_imagen(fila, voc, clases):
    with Image.open(PROJECT / fila['binary_mask_path']) as m:
        y = np.asarray(m) > 0
    h, w = y.shape
    rel = fila['image_path'].split('VOC2012/', 1)[1]
    with Image.open(Path(voc) / rel) as im:
        rgb = np.asarray(im.convert('RGB'), dtype=np.float32) / 255
    if rgb.shape[:2] != y.shape:
        raise RuntimeError(f"Imagen y máscara de distinto tamaño: {fila['id']}")
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lab, n = ndi.label(y, structure=np.ones((3, 3)))
    tam = ndi.sum(np.ones_like(y), lab, range(1, n + 1)) if n else []
    minimo = CFG['taxonomy_condition']['region_min_fraction'] * h * w
    regiones = int(sum(t >= minimo for t in tam))
    borde_lab = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    ancho = max(1, round(CFG['taxonomy_condition']['ring_width_fraction_of_max_side'] * max(h, w)))
    anillo = (ndi.distance_transform_edt(~y) <= ancho) & ~y if y.any() else np.zeros_like(y)
    l_in = float(lum[y].mean()) if y.any() else float('nan')
    l_ring = float(lum[anillo].mean()) if anillo.any() else float('nan')
    with Image.open(Path(voc) / fila['original_mask_path'].split('VOC2012/', 1)[1]) as mc:
        col = np.asarray(mc.convert('RGB'))[y]
    cuenta = Counter(clases.get(tuple(c), 'otra') for c in map(tuple, col)) if len(col) else Counter()
    total = sum(cuenta.values())
    presentes = sorted(k for k, v in cuenta.items() if total and v / total >= 0.05 and k != 'otra')
    clase = presentes[0] if len(presentes) == 1 else ('mixta' if presentes else 'sin_clase')
    xml = Path(voc) / 'bboxes/annotations' / (fila['id'] + '.xml')
    origen, trunc = '', 0
    if xml.exists():
        t = xml.read_text(encoding='utf-8', errors='ignore')
        mo = re.search(r'<folder>(.*?)</folder>', t)
        origen = mo.group(1) if mo else ''
        trunc = int('<truncated>1' in t)
    return {'id': fila['id'], 'split': fila['split'], 'group_id': fila['group_id'],
            'area_frac': y.mean(), 'regiones': regiones, 'borde': int(bool(borde_lab)), 'lum_rotulo': l_in,
            'lum_anillo': l_ring, 'contraste': abs(l_in - l_ring), 'clase': clase, '_origen': origen,
            'xml_truncated': trunc}


def cmd_atributos(a):
    voc = Path(a.voc)
    clases = colores_clase(voc)
    filas = []
    for split in ['train', 'val', 'test']:
        for f in read_csv(PROJECT / 'resultados/preentrenamiento' / f'{split}.csv'):
            filas.append(atributos_imagen({**f, 'split': split}, voc, clases))
        print(split, 'listo', flush=True)
    tr = [f for f in filas if f['split'] == 'train']
    frec = Counter(f['_origen'] for f in tr if f['_origen'])
    codigo = {o: f'O{k + 1}' for k, (o, _) in enumerate(frec.most_common())}
    q = lambda key, ps: [float(v) for v in np.nanpercentile([f[key] for f in tr], ps)]
    umbrales = {'area_cuartiles': q('area_frac', [25, 50, 75]), 'lum_p10_p90': q('lum_rotulo', [10, 90]),
                'lum_terciles': q('lum_rotulo', [100 / 3, 200 / 3]), 'contraste_p10': q('contraste', [10])[0],
                'contraste_terciles': q('contraste', [100 / 3, 200 / 3]), 'origenes_codificados': len(codigo),
                'imagenes_train': len(tr), 'nota': 'Cortes calculados solo con train', **firma()}
    for f in filas:
        f['origen'] = codigo.get(f.pop('_origen'), 'O?')
        for k in ['area_frac', 'lum_rotulo', 'lum_anillo', 'contraste']:
            f[k] = r4(f[k]) if k != 'area_frac' else round(float(f[k]), 6)
    OUT.mkdir(parents=True, exist_ok=True)
    escribir_csv(ATRIB, filas)
    (OUT / 'umbrales_train.json').write_text(json.dumps(umbrales, indent=1, ensure_ascii=False), encoding='utf-8')
    print(ATRIB, len(filas), 'imágenes;', json.dumps({k: v for k, v in umbrales.items() if 'sha' not in k}))


# ---------------------------------------------------------------- extraer conteos por imagen de los clean.json

def cmd_extraer(a):
    fuente = Path(a.diagnosticos)
    archivos = sorted(fuente.glob('*_seed*_clean.json'))
    if len(archivos) != 50:
        raise SystemExit(f'Se esperaban 50 *_clean.json y hay {len(archivos)}')
    filas, origen = [], {}
    for p in archivos:
        r = json.loads(p.read_text(encoding='utf-8'))
        origen[p.name] = {'sha256': sha_file(p), 'payload_sha256': r.get('payload_sha256'),
                          'checkpoint_sha256': r['checkpoint_sha256']}
        for split in ['val', 'test']:
            for d in r['splits'][split]['details']:
                filas.append({'model': r['model'], 'seed': r['seed'], 'split': split, 'id': d['id'],
                              'group_id': d['group_id'], 'Dice': round(d['Dice'], 6), 'TP': d['TP'], 'FP': d['FP'],
                              'FN': d['FN'], 'HD95': round(d['HD95'], 3), 'recall': round(d['recall'], 6),
                              'empty_prediction': d['empty_prediction']})
    OUT.mkdir(parents=True, exist_ok=True)
    escribir_csv(CORRIDAS, filas)
    (OUT / 'fuente_corridas.json').write_text(json.dumps({'fuente': 'salida clean del complemento GPU (50 corridas)',
                                                          'archivos': origen, **firma()}, indent=1), encoding='utf-8')
    print(CORRIDAS, len(filas), 'filas')


# ---------------------------------------------------------------- estadística por grupos

class Grupos:
    """Bootstrap por grupos: pesos multinomiales sobre los grupos, fijos para todas las estadísticas de un análisis."""

    def __init__(self, grupos, semilla):
        self.g = sorted(set(grupos))
        self.ix = {g: k for k, g in enumerate(self.g)}
        rng = np.random.default_rng(semilla)
        self.W = rng.multinomial(len(self.g), np.full(len(self.g), 1 / len(self.g)), size=B).astype(np.float64)

    def vector(self, pares):
        v = np.zeros(len(self.g))
        for g, x in pares:
            v[self.ix[g]] += x
        return v

    def razon(self, num, den):
        """num, den: vectores por grupo. Estimación y IC del cociente sum(num)/sum(den)."""
        est = num.sum() / den.sum() if den.sum() else float('nan')
        with np.errstate(invalid='ignore', divide='ignore'):
            bs = (self.W @ num) / (self.W @ den)
        bs = bs[np.isfinite(bs)]
        return est, bs


def ic(bs):
    return [r4(np.percentile(bs, 2.5)), r4(np.percentile(bs, 97.5))] if len(bs) else [None, None]


def p_bilateral(bs, nulo=0.0):
    if not len(bs):
        return 1.0
    return min(1.0, 2 * min((np.sum(bs <= nulo) + 1) / (len(bs) + 1), (np.sum(bs >= nulo) + 1) / (len(bs) + 1)))


def holm(ps):
    orden = np.argsort(ps); m = len(ps); adj = np.empty(m); acum = 0
    for k, i in enumerate(orden):
        acum = max(acum, min(1.0, (m - k) * ps[i])); adj[i] = acum
    return adj


def media_por_grupo(G, filas, valor, filtro):
    """Dice medio por grupo dentro del filtro: media de las medias de cada grupo (celda grupo-subgrupo)."""
    celdas = defaultdict(list)
    for f in filas:
        if filtro(f):
            celdas[f['group_id']].append(valor(f))
    medias = G.vector((g, float(np.mean(v))) for g, v in celdas.items())
    tiene = G.vector((g, 1.0) for g in celdas)
    est, bs = G.razon(medias, tiene)
    return est, bs, len(celdas), sum(len(v) for v in celdas.values())


# ---------------------------------------------------------------- taxonomía

def eje_a(f, hd=CFG['taxonomy_geometric']['hd95_distant_px']):
    tp, fp, fn = f['TP'], f['FP'], f['FN']
    if tp == 0 and fp == 0:
        return 'A1_omision_total'
    if tp == 0:
        return 'A2_deteccion_desplazada'
    if fn >= 2 * fp:
        return 'A3_subsegmentacion'
    if fp >= 2 * fn:
        return 'A4_exceso_adyacente' if f['HD95'] <= hd else 'A5_falso_positivo_distante'
    return 'A6_error_mixto'


def eje_b(at, um, ruido):
    if at['id'] in ruido:
        return 'B1_ruido_confirmado'
    if at['borde']:
        return 'B2_rotulo_en_borde'
    if at['regiones'] >= 2:
        return 'B3_varios_rotulos'
    if at['area_frac'] < um['area_cuartiles'][0]:
        return 'B4_rotulo_pequeno'
    lo, hi = um['lum_p10_p90']
    if not (lo <= at['lum_rotulo'] <= hi):
        return 'B5_luz_extrema'
    if at['contraste'] < um['contraste_p10']:
        return 'B6_bajo_contraste'
    return 'B7_sin_condicion'


CAT_A = [r[0] for r in CFG['taxonomy_geometric']['rules']]
CAT_B = [r[0] for r in CFG['taxonomy_condition']['rules']]


def cargar():
    at = {f['id']: f for f in leer_csv(ATRIB)}
    for f in at.values():
        for k in ['area_frac', 'lum_rotulo', 'lum_anillo', 'contraste']:
            f[k] = float(f[k]) if f[k] not in ('', 'None', 'nan') else float('nan')
        for k in ['regiones', 'borde', 'xml_truncated']:
            f[k] = int(f[k])
    um = json.loads((OUT / 'umbrales_train.json').read_text(encoding='utf-8'))
    corr = leer_csv(CORRIDAS)
    for f in corr:
        for k in ['TP', 'FP', 'FN', 'empty_prediction', 'seed']:
            f[k] = int(f[k])
        for k in ['Dice', 'HD95', 'recall']:
            f[k] = float(f[k])
    ruido = {f['id'] for f in leer_csv(PROJECT / 'resultados/ruido_etiqueta_e2/confirmar.csv')
             if f.get('veredicto_final', '').startswith('error')}
    return at, um, corr, ruido


def frecuencias(G, casos, categorias, clave):
    total = G.vector((c['group_id'], 1.0) for c in casos)
    out = []
    for cat in categorias:
        num = G.vector((c['group_id'], 1.0) for c in casos if c[clave] == cat)
        est, bs = G.razon(num, total)
        k = int(num.sum())
        out.append({'categoria': cat, 'casos': k, 'frecuencia': r4(est), 'IC95_grupos': ic(bs)})
    return out


def taxonomia(at, um, corr, ruido, split='val'):
    res = {}
    filas = [f for f in corr if f['split'] == split]
    G = Grupos([f['group_id'] for f in filas], CFG['bootstrap']['seed'])
    casos_todos = []
    for m in CFG['models']:
        fm = [f for f in filas if f['model'] == m]
        sens = {}
        for umbral in [ERR] + CFG['error_case']['sensitivity']:
            cs = [{**f, 'A': eje_a(f), 'B': eje_b(at[f['id']], um, ruido)} for f in fm if f['Dice'] < umbral]
            if umbral == ERR:
                casos = cs
            sens[str(umbral)] = {'casos': len(cs), 'imagenes': len({c['id'] for c in cs}),
                                 'A': {r['categoria']: r['frecuencia'] for r in frecuencias(G, cs, CAT_A, 'A')} if cs else {},
                                 'B': {r['categoria']: r['frecuencia'] for r in frecuencias(G, cs, CAT_B, 'B')} if cs else {}}
        cruce = Counter((c['A'], c['B']) for c in casos)
        res[m] = {'pares_imagen_semilla': len(fm), 'casos_error': len(casos),
                  'imagenes_con_error': len({c['id'] for c in casos}),
                  'grupos_con_error': len({c['group_id'] for c in casos}),
                  'llega_a_100': len(casos) >= CFG['error_case']['minimum_per_model'],
                  'eje_A': frecuencias(G, casos, CAT_A, 'A'), 'eje_B': frecuencias(G, casos, CAT_B, 'B'),
                  'cruce_A_B': {f'{a}|{b}': n for (a, b), n in sorted(cruce.items())}, 'sensibilidad_umbral': sens}
        casos_todos += casos
    return res, casos_todos


def compartidos(at, um, corr, ruido):
    medias = defaultdict(list)
    for f in corr:
        if f['split'] == 'val':
            medias[(f['model'], f['id'])].append(f['Dice'])
    ids = sorted({f['id'] for f in corr if f['split'] == 'val'})
    nfallan = {i: sum(np.mean(medias[(m, i)]) < ERR for m in CFG['models']) for i in ids}
    dificiles = [i for i in ids if nfallan[i] >= 8]
    G = Grupos([at[i]['group_id'] for i in ids], CFG['bootstrap']['seed'] + 1)
    est, bs = G.razon(G.vector((at[i]['group_id'], 1.0) for i in dificiles), G.vector((at[i]['group_id'], 1.0) for i in ids))
    dist = Counter(nfallan.values())
    falla_alguna = [i for i in ids if nfallan[i] >= 1]
    return {'imagenes_val': len(ids), 'dificiles_8_de_10': len(dificiles), 'fraccion_dificiles': r4(est),
            'IC95_grupos': ic(bs), 'distribucion_redes_que_fallan': {str(k): dist[k] for k in sorted(dist)},
            'entre_imagenes_con_algun_error_fraccion_dificiles': r4(len(dificiles) / len(falla_alguna)) if falla_alguna else None,
            'eje_B_de_dificiles': dict(Counter(eje_b(at[i], um, ruido) for i in dificiles)),
            'eje_B_de_todas_val': dict(Counter(eje_b(at[i], um, ruido) for i in ids)), 'ids_dificiles': dificiles}


# ---------------------------------------------------------------- subgrupos y daño

def nivel(at, attr, um):
    if attr == 'clase':
        return at['clase']
    if attr == 'origen':
        return at['origen']
    if attr == 'borde':
        return 'si' if at['borde'] else 'no'
    if attr == 'n_rotulos':
        return '2+' if at['regiones'] >= 2 else '1'
    if attr == 'area_cuartil_train':
        return f"Q{1 + int(np.searchsorted(um['area_cuartiles'], at['area_frac'], side='right'))}"
    if attr == 'luminancia_tercil_train':
        return f"T{1 + int(np.searchsorted(um['lum_terciles'], at['lum_rotulo'], side='right'))}"
    if attr == 'contraste_tercil_train':
        return f"T{1 + int(np.searchsorted(um['contraste_terciles'], at['contraste'], side='right'))}"
    raise ValueError(attr)


def subgrupos(at, um, corr, split):
    filas = [f for f in corr if f['split'] == split]
    por_img = defaultdict(list); sev = defaultdict(list)
    for f in filas:
        por_img[(f['model'], f['id'])].append(f['Dice'])
        sev[(f['model'], f['id'])].append(f['recall'] < 0.5)
    ids = sorted({f['id'] for f in filas})
    G = Grupos([at[i]['group_id'] for i in ids], CFG['bootstrap']['seed'] + 2)
    tabla, dano = [], []
    for m in CFG['models']:
        img = [{'id': i, 'group_id': at[i]['group_id'], 'Dice': float(np.mean(por_img[(m, i)])),
                'sev': float(np.mean(sev[(m, i)]))} for i in ids]
        for attr in CFG['subgroups']['attributes']:
            niveles = sorted({nivel(at[i], attr, um) for i in ids})
            bloque = []
            for nv in niveles:
                dentro = lambda r, nv=nv: nivel(at[r['id']], attr, um) == nv
                est, bs, ng, ni = media_por_grupo(G, img, lambda r: r['Dice'], dentro)
                fila = {'split': split, 'model': m, 'atributo': attr, 'nivel': nv, 'imagenes': ni, 'grupos': ng,
                        'Dice': r4(est), 'IC95': ic(bs), 'dif_vs_resto': None, 'IC95_dif': [None, None], 'p': None,
                        'p_holm': None}
                if ng >= CFG['subgroups']['min_groups'] and ng < len(G.g):
                    est2, _, _, _ = media_por_grupo(G, img, lambda r: r['Dice'], lambda r, d=dentro: not d(r))
                    # mismos pesos W en ambos: la diferencia se remuestrea pareada por grupos
                    est_d, bs_d = est - est2, media_por_grupo_pareado(G, img, dentro)
                    fila.update({'dif_vs_resto': r4(est_d), 'IC95_dif': ic(bs_d), 'p': p_bilateral(bs_d)})
                bloque.append(fila)
                if attr in CFG['harm']['by']:
                    num = G.vector((r['group_id'], r['sev']) for r in img if dentro(r))
                    den = G.vector((r['group_id'], 1.0) for r in img if dentro(r))
                    e, b = G.razon(num, den)
                    dano.append({'split': split, 'model': m, 'atributo': attr, 'nivel': nv, 'imagenes': ni,
                                 'omision_grave': r4(e), 'IC95': ic(b)})
            con_p = [f for f in bloque if f['p'] is not None]
            if con_p:
                for f, adj in zip(con_p, holm(np.array([f['p'] for f in con_p]))):
                    f['p_holm'] = r4(adj); f['p'] = r4(f['p'])
            tabla += bloque
        num = G.vector((r['group_id'], r['sev']) for r in img); den = G.vector((r['group_id'], 1.0) for r in img)
        e, b = G.razon(num, den)
        dano.append({'split': split, 'model': m, 'atributo': 'todas', 'nivel': 'todas', 'imagenes': len(img),
                     'omision_grave': r4(e), 'IC95': ic(b)})
    return tabla, dano


def media_por_grupo_pareado(G, img, dentro):
    """Bootstrap de (media por grupo dentro) - (media por grupo fuera) con los mismos pesos de grupo."""
    def partes(filtro):
        celdas = defaultdict(list)
        for r in img:
            if filtro(r):
                celdas[r['group_id']].append(r['Dice'])
        return G.vector((g, float(np.mean(v))) for g, v in celdas.items()), G.vector((g, 1.0) for g in celdas)
    m1, t1 = partes(dentro); m0, t0 = partes(lambda r: not dentro(r))
    with np.errstate(invalid='ignore', divide='ignore'):
        bs = (G.W @ m1) / (G.W @ t1) - (G.W @ m0) / (G.W @ t0)
    return bs[np.isfinite(bs)]


# ---------------------------------------------------------------- muestra para revisión humana

def muestra_revision(casos):
    rh = CFG['human_review']; rng = np.random.default_rng(rh['seed']); filas = []
    for m in rh['models']:
        cm = [c for c in casos if c['model'] == m]
        cuenta = Counter(c['A'] for c in cm)
        n = rh['cases_per_model']
        cuota = {k: max(3, n * v / len(cm)) for k, v in cuenta.items()}
        esc = n / sum(cuota.values())
        base = {k: max(min(3, cuenta[k]), int(cuota[k] * esc)) for k in cuota}
        resto = sorted(cuota, key=lambda k: -(cuota[k] * esc - int(cuota[k] * esc)))
        k = 0
        while sum(base.values()) < n and k < 10 * len(resto):
            c = resto[k % len(resto)]
            if base[c] < cuenta[c]:
                base[c] += 1
            k += 1
        for cat, q in sorted(base.items()):
            pool = [c for c in cm if c['A'] == cat]
            orden = rng.permutation(len(pool)); vistos = set(); elegidos = []
            for i in orden:
                if pool[i]['id'] not in vistos:
                    vistos.add(pool[i]['id']); elegidos.append(pool[i])
                if len(elegidos) == q:
                    break
            for c in elegidos:
                filas.append({'model': m, 'seed': c['seed'], 'id': c['id'], 'group_id': c['group_id'],
                              'Dice': round(c['Dice'], 4), 'A_asignada': c['A'], 'B_asignada': c['B']})
    rng.shuffle(filas)
    for k, f in enumerate(filas, 1):
        f['n'] = k
    return [{'n': f['n'], **{k: v for k, v in f.items() if k != 'n'}} for f in filas]


# ---------------------------------------------------------------- tablas

def tabla_md(cab, filas):
    return '\n'.join(['| ' + ' | '.join(cab) + ' |', '|' + '|'.join('---' if k == 0 else '---:' for k in range(len(cab))) + '|']
                     + ['| ' + ' | '.join(str(x) for x in f) + ' |' for f in filas])


def fmt(x, ci=None):
    if x is None:
        return '—'
    s = f'{x:.3f}'.replace('.', ',')
    if ci and ci[0] is not None:
        s += f' [{ci[0]:.3f}; {ci[1]:.3f}]'.replace('.', ',').replace(',;', ';')
        s = s.replace('; ', '; ')
    return s


def cmd_analizar(a):
    at, um, corr, ruido = cargar()
    tax, casos = taxonomia(at, um, corr, ruido)
    comp = compartidos(at, um, corr, ruido)
    sub_v, dano_v = subgrupos(at, um, corr, 'val')
    sub_t, dano_t = subgrupos(at, um, corr, 'test')
    muestra = muestra_revision(casos)
    escribir_csv(OUT / 'taxonomia_casos_val.csv',
                 [{k: c[k] for k in ['model', 'seed', 'id', 'group_id', 'Dice', 'TP', 'FP', 'FN', 'HD95', 'A', 'B']}
                  for c in casos])
    escribir_csv(OUT / 'subgrupos.csv', [{**f, 'IC95': f['IC95'], 'IC95_dif': f['IC95_dif']} for f in sub_v + sub_t])
    escribir_csv(OUT / 'dano_omision_grave.csv', dano_v + dano_t)
    escribir_csv(OUT / 'muestra_revision.csv', muestra)
    resumen = {'status': 'exploratorio, posterior al test; protocolo a priori del 22/11/2026', **firma(),
               'umbral_error_Dice': ERR, 'split': 'val', 'taxonomia': tax, 'errores_compartidos': comp,
               'umbrales_train': {k: v for k, v in um.items() if 'sha' not in k},
               'muestra_revision': {'casos': len(muestra), 'archivo': 'muestra_revision.csv'}}
    (OUT / 'RESUMEN_LOCAL.json').write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding='utf-8')
    escribir_tablas(tax, comp, sub_v, sub_t, dano_v, dano_t)
    print('Casos de error por red (val, Dice < 0,80):', {m: tax[m]['casos_error'] for m in CFG['models']})
    print('Difíciles para 8+ redes:', comp['dificiles_8_de_10'], 'de', comp['imagenes_val'])


def escribir_tablas(tax, comp, sub_v, sub_t, dano_v, dano_t):
    L = ['# Errores, subgrupos y daño potencial: tablas locales', '',
         'Generado por `scripts/errores_atajos_e2.py analizar`. Exploratorio, posterior al test; reglas fijadas antes en '
         '`PROTOCOLO_ERRORES_ATAJOS_20261122.md`. Validación (218 imágenes, 122 grupos), 10 redes × 5 semillas. '
         'IC95 bootstrap por grupos (4.000). Caso de error: par imagen-semilla con Dice < 0,80.', '',
         '## Taxonomía, eje A: forma del error (fila 65)', '',
         'Frecuencia sobre el censo de casos de error de cada red [IC95 por grupos].', '']
    cab = ['Red', 'Casos', 'Imágenes'] + [c.split('_', 1)[1].replace('_', ' ') for c in CAT_A]
    L.append(tabla_md(cab, [[m, tax[m]['casos_error'], tax[m]['imagenes_con_error']] +
                            [fmt(r['frecuencia'], r['IC95_grupos']) for r in tax[m]['eje_A']] for m in CFG['models']]))
    L += ['', '## Taxonomía, eje B: condición de la imagen (fila 65)', '']
    cab = ['Red'] + [c.split('_', 1)[1].replace('_', ' ') for c in CAT_B]
    L.append(tabla_md(cab, [[m] + [fmt(r['frecuencia'], r['IC95_grupos']) for r in tax[m]['eje_B']] for m in CFG['models']]))
    L += ['', 'Línea base del eje B en todas las imágenes de validación (sin condicionar a error): ' +
          ', '.join(f'{k} {v}' for k, v in sorted(comp['eje_B_de_todas_val'].items())) + '.', '',
          '## Sensibilidad al umbral de error', '']
    L.append(tabla_md(['Red', 'Casos < 0,70', 'Casos < 0,80', 'Casos < 0,90'],
                      [[m] + [tax[m]['sensibilidad_umbral'][u]['casos'] for u in ['0.7', '0.8', '0.9']] for m in CFG['models']]))
    L += ['', '## Errores compartidos', '',
          f"Imágenes con Dice medio < 0,80 en al menos 8 de las 10 redes: {comp['dificiles_8_de_10']} de "
          f"{comp['imagenes_val']} ({fmt(comp['fraccion_dificiles'], comp['IC95_grupos'])}). Condición de esas imágenes: "
          + ', '.join(f'{k} {v}' for k, v in sorted(comp['eje_B_de_dificiles'].items())) + '. Redes que fallan por imagen '
          '(número de redes: imágenes): ' + ', '.join(f'{k}: {v}' for k, v in comp['distribucion_redes_que_fallan'].items()) + '.',
          '', '## Subgrupos en validación, top 3 (fila 67)', '',
          'Dice por grupo en el subgrupo [IC95]; diferencia con el resto [IC95] y p con Holm dentro de red y atributo. '
          'Todas las redes y el test están en `subgrupos.csv`.', '']
    for attr in CFG['subgroups']['attributes']:
        L += [f'### {attr}', '']
        niveles = sorted({f['nivel'] for f in sub_v if f['atributo'] == attr})
        filas = []
        for nv in niveles:
            fila = [nv]
            for m in CFG['top3_by_validation']:
                f = next(x for x in sub_v if x['model'] == m and x['atributo'] == attr and x['nivel'] == nv)
                if m == CFG['top3_by_validation'][0]:
                    fila.append(f"{f['imagenes']}/{f['grupos']}")
                d = '' if f['dif_vs_resto'] is None else f"; Δ {fmt(f['dif_vs_resto'], f['IC95_dif'])}, p Holm {fmt(f['p_holm'])}"
                fila.append(fmt(f['Dice'], f['IC95']) + d)
            filas.append(fila)
        L += [tabla_md(['Nivel', 'Imág./grupos'] + CFG['top3_by_validation'], filas), '']
    sig = [f for f in sub_v if f['p_holm'] is not None and f['p_holm'] < 0.05]
    L += ['### Diferencias con p Holm < 0,05 en validación (las 10 redes)', '']
    L += [tabla_md(['Red', 'Atributo', 'Nivel', 'Imág.', 'Δ vs resto [IC95]', 'p Holm'],
                   [[f['model'], f['atributo'], f['nivel'], f['imagenes'], fmt(f['dif_vs_resto'], f['IC95_dif']), fmt(f['p_holm'])]
                    for f in sig]) if sig else 'Ninguna.', '']
    sig_t = [f for f in sub_t if f['p_holm'] is not None and f['p_holm'] < 0.05]
    L += ['### Las mismas en test (descriptivo)', '',
          tabla_md(['Red', 'Atributo', 'Nivel', 'Imág.', 'Δ vs resto [IC95]', 'p Holm'],
                   [[f['model'], f['atributo'], f['nivel'], f['imagenes'], fmt(f['dif_vs_resto'], f['IC95_dif']), fmt(f['p_holm'])]
                    for f in sig_t]) if sig_t else 'Ninguna.', '']
    L += ['## Daño potencial: omisión grave (recall < 0,5), validación', '',
          'Fracción de pares imagen-semilla en que la red pierde la mayor parte del rótulo [IC95 por grupos].', '']
    filas = []
    for m in CFG['models']:
        tot = next(d for d in dano_v if d['model'] == m and d['atributo'] == 'todas')
        tt = next(d for d in dano_t if d['model'] == m and d['atributo'] == 'todas')
        peor = max((d for d in dano_v if d['model'] == m and d['atributo'] == 'clase' and d['imagenes'] >= 8),
                   key=lambda d: d['omision_grave'])
        filas.append([m, fmt(tot['omision_grave'], tot['IC95']), fmt(tt['omision_grave'], tt['IC95']),
                      f"{peor['nivel']} ({peor['imagenes']} imág.) {fmt(peor['omision_grave'], peor['IC95'])}"])
    L += [tabla_md(['Red', 'Val', 'Test (descr.)', 'Clase con más omisión grave en val (≥ 8 imágenes)'], filas), '']
    for attr in ['area_cuartil_train', 'luminancia_tercil_train']:
        niveles = sorted({d['nivel'] for d in dano_v if d['atributo'] == attr})
        L += [f'Omisión grave por {attr} (top 3, val):', '',
              tabla_md(['Red'] + niveles, [[m] + [fmt(next(d for d in dano_v if d['model'] == m and d['atributo'] == attr
                                                              and d['nivel'] == nv)['omision_grave']) for nv in niveles]
                                           for m in CFG['top3_by_validation']]), '']
    (OUT / 'TABLAS_LOCAL.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


# ---------------------------------------------------------------- después de Colab: atajos y taxonomía exacta

def cmd_atajos(a):
    ent = Path(a.entrada)
    at, um, corr, ruido = cargar()
    archivos = sorted(ent.glob('*_seed*_atajos.json'))
    if not archivos:
        raise SystemExit(f'No hay *_atajos.json en {ent}')
    filas = []
    for p in archivos:
        r = json.loads(p.read_text(encoding='utf-8'))
        for d in r['details']:
            filas.append({'model': r['model'], 'seed': r['seed'], **d})
    ids = sorted({f['id'] for f in filas})
    G = Grupos([at[i]['group_id'] for i in ids], CFG['bootstrap']['seed'] + 3)
    delta = CFG['delta']; res = {'archivos': len(archivos), **firma(), 'modelos': {}}
    pruebas = []
    for m in CFG['models']:
        fm = [f for f in filas if f['model'] == m]
        if not fm:
            continue
        sem = sorted({f['seed'] for f in fm}); r = {'semillas': sem}
        por = defaultdict(lambda: defaultdict(list))
        for f in fm:
            for c in ['C0', 'C1', 'C3']:
                por[c][f['id']].append(f[f'Dice_{c}'])
            por['C2frac'][f['id']].append(f['C2_fraccion_zona_predicha'])
            por['C2alarma'][f['id']].append(float(f['C2_falsa_alarma']))
        img = lambda c: [{'id': i, 'group_id': at[i]['group_id'], 'v': float(np.mean(por[c][i]))} for i in por[c]]
        base = {i: float(np.mean(por['C0'][i])) for i in por['C0']}
        for c in ['C1', 'C3']:
            dif = [{'id': i, 'group_id': at[i]['group_id'], 'v': float(np.mean(por[c][i])) - base[i]} for i in base]
            est, bs, _, _ = media_por_grupo(G, dif, lambda x: x['v'], lambda x: True)
            p_uno = (np.sum(bs >= -delta) + 1) / (len(bs) + 1)
            r[c] = {'Dice_condicion': r4(media_por_grupo(G, img(c), lambda x: x['v'], lambda x: True)[0]),
                    'Dice_C0': r4(media_por_grupo(G, img('C0'), lambda x: x['v'], lambda x: True)[0]),
                    'delta': r4(est), 'IC95': ic(bs), 'p_unilateral_H0_delta_mayor_igual_menos_d': float(p_uno)}
            pruebas.append((m, c, p_uno))
        for c, key in [('C2frac', 'C2_fraccion_zona_predicha'), ('C2alarma', 'C2_tasa_falsa_alarma')]:
            est, bs, _, _ = media_por_grupo(G, img(c), lambda x: x['v'], lambda x: True)
            r[key] = {'estimacion': r4(est), 'IC95': ic(bs)}
        r['C2_decision'] = ('atajo de contexto' if r['C2_fraccion_zona_predicha']['IC95'][0] > 0.10
                            else 'sin atajo de contexto material')
        # taxonomía exacta frente al sustituto HD95
        acuerdo, total, exacta = 0, 0, Counter()
        for f in fm:
            if f['Dice_C0'] >= ERR:
                continue
            fx = {'TP': f['TP'], 'FP': f['FP'], 'FN': f['FN'], 'HD95': f['HD95']}
            a_proxy = eje_a(fx)
            if a_proxy in ('A4_exceso_adyacente', 'A5_falso_positivo_distante'):
                a_ex = 'A5_falso_positivo_distante' if f['fp_px_distantes'] > 0.5 * f['FP'] else 'A4_exceso_adyacente'
            else:
                a_ex = a_proxy
            exacta[a_ex] += 1; total += 1; acuerdo += a_ex == a_proxy
        r['taxonomia_exacta'] = {'casos': total, 'A': dict(exacta), 'acuerdo_con_sustituto_HD95': r4(acuerdo / total) if total else None,
                                 'rotulos_omitidos_en_varios': int(sum(f['gt_omitidas'] for f in fm if f['gt_componentes'] >= 2))}
        res['modelos'][m] = r
    adj = holm(np.array([p for _, _, p in pruebas]))
    for (m, c, _), pa in zip(pruebas, adj):
        r = res['modelos'][m][c]; r['p_holm'] = r4(pa)
        r['decision'] = ('dependencia material' if r['IC95'][1] < -delta and pa < 0.05 else
                         'sin dependencia material' if r['IC95'][0] > -delta else 'indeterminado')
    (OUT / 'RESUMEN_ATAJOS.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    L = ['# Atajos: contexto, rótulo borrado y color (filas 13 y 67)', '',
         'Generado por `scripts/errores_atajos_e2.py atajos` con la salida de `colab/Atajos_Errores_E2.ipynb`. '
         'Validación, exploratorio, reglas fijadas antes (protocolo del 22/11/2026). Δ = Dice(condición) − Dice(limpia), '
         'IC95 por grupos; δ = 0,02; Holm sobre 20 pruebas.', '',
         tabla_md(['Red', 'Dice limpia', 'Sin contexto Δ [IC95]', 'Decisión', 'Gris Δ [IC95]', 'Decisión',
                   'Zona borrada predicha [IC95]', 'Falsa alarma en negativo [IC95]'],
                  [[m, fmt(r['C1']['Dice_C0']), fmt(r['C1']['delta'], r['C1']['IC95']), r['C1']['decision'],
                    fmt(r['C3']['delta'], r['C3']['IC95']), r['C3']['decision'],
                    fmt(r['C2_fraccion_zona_predicha']['estimacion'], r['C2_fraccion_zona_predicha']['IC95']),
                    fmt(r['C2_tasa_falsa_alarma']['estimacion'], r['C2_tasa_falsa_alarma']['IC95'])]
                   for m, r in res['modelos'].items()]), '',
         '## Taxonomía exacta con componentes predichas', '',
         tabla_md(['Red', 'Casos', 'Acuerdo con HD95', 'Rótulos omitidos en imágenes con varios'],
                  [[m, r['taxonomia_exacta']['casos'], fmt(r['taxonomia_exacta']['acuerdo_con_sustituto_HD95']),
                    r['taxonomia_exacta']['rotulos_omitidos_en_varios']] for m, r in res['modelos'].items()])]
    (OUT / 'TABLAS_ATAJOS.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


def letterbox_rgb(path, size=256):
    with Image.open(path) as im:
        im = im.convert('RGB'); s = min(size / im.width, size / im.height)
        r = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.BILINEAR)
        c = Image.new('RGB', (size, size), 0); c.paste(r, ((size - r.width) // 2, (size - r.height) // 2))
    return np.asarray(c)


def letterbox_mask(path, size=256):
    with Image.open(path) as im:
        im = im.convert('L'); s = min(size / im.width, size / im.height)
        r = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.NEAREST)
        c = Image.new('L', (size, size), 0); c.paste(r, ((size - r.width) // 2, (size - r.height) // 2))
    return np.asarray(c) > 0


def cmd_revision(a):
    """Paneles imagen | real | predicción (verde acierto, rojo exceso, azul omisión) y hoja para anotar."""
    ent, voc, dest = Path(a.entrada), Path(a.voc), Path(a.destino)
    dest.mkdir(parents=True, exist_ok=True)
    muestra = leer_csv(OUT / 'muestra_revision.csv')
    filas = {f['id']: f for split in ['val'] for f in read_csv(PROJECT / 'resultados/preentrenamiento' / f'{split}.csv')}
    cache = {}
    paginas, actual = [], []
    for c in muestra:
        clave = (c['model'], int(c['seed']))
        if clave not in cache:
            z = np.load(ent / f"{c['model']}_seed{c['seed']}_mascaras_val.npz")
            cache[clave] = {i: k for k, i in enumerate(z['ids'])}, np.unpackbits(z['bits'], axis=1)[:, :65536].reshape(-1, 256, 256)
        ix, masks = cache[clave]
        pred = masks[ix[c['id']]].astype(bool)
        f = filas[c['id']]
        rgb = letterbox_rgb(voc / f['image_path'].split('VOC2012/', 1)[1])
        y = letterbox_mask(PROJECT / f['binary_mask_path'])
        ov = (rgb * 0.45).astype(np.uint8).copy()
        ov[pred & y] = [40, 200, 60]; ov[pred & ~y] = [230, 40, 40]; ov[~pred & y] = [40, 90, 240]
        gt = rgb.copy(); gt[y] = (0.5 * gt[y] + [0, 127, 0]).astype(np.uint8)
        actual.append((c, np.concatenate([rgb, np.full((256, 4, 3), 255, np.uint8), gt,
                                          np.full((256, 4, 3), 255, np.uint8), ov], axis=1)))
        if len(actual) == 4:
            paginas.append(actual); actual = []
    if actual:
        paginas.append(actual)
    from PIL import ImageDraw
    for k, pag in enumerate(paginas, 1):
        lienzo = Image.new('RGB', (776, 4 * 280), 'white'); d = ImageDraw.Draw(lienzo)
        for j, (c, tira) in enumerate(pag):
            d.text((4, j * 280 + 4), f"caso {c['n']}  ({c['id']})", fill='black')
            lienzo.paste(Image.fromarray(tira), (0, j * 280 + 20))
        lienzo.save(dest / f'pagina_{k:02d}.png')
    hoja = [{'n': c['n'], 'id': c['id'], 'pagina': (k // 4) + 1, 'A_asignada': c['A_asignada'], 'B_asignada': c['B_asignada'],
             'acuerdo_A': '', 'A_correcta': '', 'acuerdo_B': '', 'B_correcta': '', 'exceso_es_rotulo_sin_anotar': '',
             'nota': ''} for k, c in enumerate(muestra)]
    escribir_csv(dest / 'hoja_revision_errores.csv', hoja)
    (dest / 'LEEME.txt').write_text(
        'Revisión de la taxonomía de errores. Cada página tiene 4 casos: imagen | máscara real (verde) | predicción '
        '(verde acierto, rojo exceso, azul omisión).\nEn hoja_revision_errores.csv marca acuerdo_A y acuerdo_B con si/no; '
        'si es no, escribe la categoría correcta (A1..A6, B1..B7). Categorías:\n' +
        '\n'.join(f'{n}: {r}' for n, r in CFG['taxonomy_geometric']['rules'] + CFG['taxonomy_condition']['rules']) +
        '\nLa hoja no muestra la red ni la semilla para no sesgar.\n', encoding='utf-8')
    print(dest, len(paginas), 'páginas,', len(hoja), 'casos')


def cmd_medir_revision(a):
    hoja = leer_csv(a.hoja)
    muestra = {f['n']: f for f in leer_csv(OUT / 'muestra_revision.csv')}
    llenas = [f for f in hoja if f['acuerdo_A'].strip() and f['acuerdo_B'].strip()]
    if not llenas:
        raise SystemExit('La hoja no tiene respuestas')
    res = {'casos_revisados': len(llenas), 'casos_muestra': len(hoja), **firma(), 'por_modelo': {}}
    for eje in ['A', 'B']:
        k = sum(f[f'acuerdo_{eje}'].strip().lower().startswith('s') for f in llenas)
        res[f'acuerdo_{eje}'] = {'k': k, 'n': len(llenas), 'tasa': r4(k / len(llenas)), 'IC95_wilson': wilson(k, len(llenas))}
    for m in CFG['human_review']['models']:
        fm = [f for f in llenas if muestra[f['n']]['model'] == m]
        res['por_modelo'][m] = {eje: wilson(sum(f[f'acuerdo_{eje}'].strip().lower().startswith('s') for f in fm), len(fm))
                                for eje in ['A', 'B']} | {'n': len(fm)}
    exc = [f for f in llenas if f.get('exceso_es_rotulo_sin_anotar', '').strip()]
    k = sum(f['exceso_es_rotulo_sin_anotar'].strip().lower().startswith('s') for f in exc)
    res['exceso_es_rotulo_sin_anotar'] = {'k': k, 'n': len(exc), 'tasa': r4(k / len(exc)) if exc else None,
                                          'IC95_wilson': wilson(k, len(exc))}
    res['correcciones'] = [{k: f[k] for k in ['n', 'A_asignada', 'A_correcta', 'B_asignada', 'B_correcta', 'nota']}
                           for f in llenas if not f['acuerdo_A'].lower().startswith('s') or not f['acuerdo_B'].lower().startswith('s')]
    (OUT / 'REVISION_HUMANA.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ['casos_revisados', 'acuerdo_A', 'acuerdo_B']}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('accion', choices=['atributos', 'extraer', 'analizar', 'atajos', 'revision', 'medir-revision'])
    ap.add_argument('--voc'); ap.add_argument('--diagnosticos'); ap.add_argument('--entrada'); ap.add_argument('--destino')
    ap.add_argument('--hoja')
    a = ap.parse_args()
    {'atributos': cmd_atributos, 'extraer': cmd_extraer, 'analizar': cmd_analizar, 'atajos': cmd_atajos,
     'revision': cmd_revision, 'medir-revision': cmd_medir_revision}[a.accion](a)


if __name__ == '__main__':
    main()
