"""Fuga residual por embeddings: casi duplicados que dHash/aHash no ven (checklist 11 y 46). Exploratorio, posterior al test.

  python scripts/fuga_embeddings_e2.py --voc RUTA/VOC2012 --embeddings RUTA/embeddings_resnet50.npz
         [--protocolo-referencia RUTA] [--calcular]

Qué hace, en orden:
1. Verifica que la partición del proyecto es la misma que la de la auditoría de origen: SHA-256 de train/val/test.csv
   y normalization_train.json contra `--protocolo-referencia` (si se da) y contra los SHA fijados abajo, e igualdad
   fila a fila de id, grupo y partición. Con `--voc`, verifica además el SHA-256 de las 1.451 imágenes.
2. Embeddings: ResNet-50 de ImageNet-1k (`timm`, `resnet50.tv_in1k`), promedio global, imagen RGB con el mismo
   letterbox a 256 de `preparar_base.py`, normalización de ImageNet. Train en su vista original; val y test en 8
   variantes (rotaciones de 90° y volteo horizontal). Si `--embeddings` existe se reutiliza (orden de filas = orden
   de los CSV; se guarda su SHA-256); si no existe y se pasa `--calcular`, se calcula (necesita torch y timm).
3. Por imagen: coseno MÁXIMO contra train. La referencia no es un percentil de pares sueltos (el máximo sobre ~1.000
   comparaciones casi siempre lo supera), sino la misma estadística dentro de train: cada imagen de train contra el
   resto de train sin su propio grupo. Ancla de casi duplicado conocido: pares dentro de un mismo grupo dHash.
4. Lista de grupos de val y test con casi duplicado en train (coseno > 0,97 con rotaciones y volteos), que usa
   `sensibilidad_casi_duplicados_e2.py`, y cuántas imágenes de train tienen casi duplicado en OTRO grupo de train
   (los grupos dHash no separan sesiones de fotos).

Solo usa imágenes: ninguna máscara ni predicción. Escribe en resultados/fuga_embeddings_e2/.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preparar_base import PROJECT, letterbox, read_csv  # noqa: E402

OUT = PROJECT / 'resultados/fuga_embeddings_e2'
PRE = PROJECT / 'resultados/preentrenamiento'
SHA_PARTICION = {  # partición congelada, idéntica a la del protocolo de la auditoría de origen
    'train.csv': 'fb536572de79fcc255c9ba291581604ef0db37fd33de2c6b7ee86543bcb387a0',
    'val.csv': '630d2a72fe6fee60fc5a3fe99302fe74eb8ae633b33225548a4b6b9ee1c00208',
    'test.csv': '7340a0cab140a8180b4a669260809eb79e86be7cb43ef87f5cac9bbd22317e81',
    'normalization_train.json': '6fe0f53f427488358e295a49e1c7abd7dc69d3cb9f15e430964f30b298c3d02e',
}
UMBRALES = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98)
CASI_DUPLICADO = 0.97
TAM = 256


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verificar_particion(referencia, voc):
    r = {'sha256_proyecto': {f: sha(PRE / f) for f in SHA_PARTICION}}
    r['coincide_con_fijados'] = all(r['sha256_proyecto'][f] == h for f, h in SHA_PARTICION.items())
    if referencia:
        r['sha256_referencia'] = {f: sha(Path(referencia) / f) for f in SHA_PARTICION}
        r['coincide_con_referencia'] = r['sha256_referencia'] == r['sha256_proyecto']
        dif = 0
        for s in ('train', 'val', 'test'):
            a = [(x['id'], x['group_id'], x['split']) for x in read_csv(PRE / f'{s}.csv')]
            b = [(x['id'], x['group_id'], x['split']) for x in read_csv(Path(referencia) / f'{s}.csv')]
            dif += sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
        r['filas_distintas_id_grupo_particion'] = dif
    if voc:
        malas = [x['id'] for s in ('train', 'val', 'test') for x in read_csv(PRE / f'{s}.csv')
                 if sha(Path(voc) / x['image_path'].split('VOC2012/', 1)[1]) != x['image_sha256']]
        r['imagenes_verificadas'] = 1451 - len(malas)
        r['imagenes_distintas'] = malas
    return r


def calcular_embeddings(man, voc):
    import timm
    import torch
    m = timm.create_model('resnet50.tv_in1k', pretrained=True, num_classes=0).eval()
    media = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    desv = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = {}
    for s, filas in man.items():
        x = np.zeros((len(filas), TAM, TAM, 3), np.uint8)
        for i, f in enumerate(filas):
            with Image.open(Path(voc) / f['image_path'].split('VOC2012/', 1)[1]) as im:
                x[i] = np.asarray(letterbox(im.convert('RGB'), TAM)[0])
        t = (torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255 - media) / desv
        variantes = []
        for k in range(8 if s != 'train' else 1):
            v = torch.rot90(t, k % 4, dims=(2, 3))
            v = torch.flip(v, dims=(3,)) if k >= 4 else v
            with torch.no_grad():
                e = torch.cat([m(v[j:j + 32]) for j in range(0, len(v), 32)])
            variantes.append(torch.nn.functional.normalize(e, dim=1).numpy())
        out[s] = np.stack(variantes)
    return out


def resumen_max(m):
    return {'n': int(len(m)), 'coseno_max_p50_p90': [round(float(np.percentile(m, q)), 4) for q in (50, 90)],
            'fraccion_sobre': {f'{u:.2f}': round(float((m > u).mean()), 4) for u in UMBRALES}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--voc', type=Path)
    ap.add_argument('--embeddings', type=Path, required=True)
    ap.add_argument('--protocolo-referencia', type=Path)
    ap.add_argument('--calcular', action='store_true')
    a = ap.parse_args(argv)
    R = {'estado': 'Exploratorio, posterior al test. Solo imágenes; no usa máscaras ni predicciones.',
         'particion': verificar_particion(a.protocolo_referencia, a.voc)}
    if not R['particion']['coincide_con_fijados'] or R['particion'].get('filas_distintas_id_grupo_particion', 0):
        raise SystemExit('La partición no coincide con la de la auditoría de origen; no se reutilizan sus embeddings.')
    man = {s: read_csv(PRE / f'{s}.csv') for s in ('train', 'val', 'test')}
    if a.embeddings.exists():
        emb = dict(np.load(a.embeddings))
        R['embeddings'] = {'origen': 'caché', 'sha256': sha(a.embeddings)}
    elif a.calcular:
        emb = calcular_embeddings(man, a.voc)
        a.embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.embeddings, **emb)
        R['embeddings'] = {'origen': 'calculados', 'sha256': sha(a.embeddings)}
    else:
        raise SystemExit(f'No existe {a.embeddings}; usar --calcular')
    for s in man:
        esperado = (1 if s == 'train' else 8, len(man[s]), 2048)
        if emb[s].shape != esperado:
            raise SystemExit(f'Embeddings de {s}: forma {emb[s].shape}, se esperaba {esperado}')
    R['embeddings'].update({'modelo': 'resnet50.tv_in1k (timm), promedio global, L2', 'entrada': 'letterbox 256, ImageNet',
                            'variantes_val_test': '4 rotaciones de 90° x {original, volteo horizontal}'})

    E = emb['train'][0]
    g_tr = np.array([f['group_id'] for f in man['train']])
    S = E @ E.T
    iu = np.triu_indices(len(E), 1)
    mismo = g_tr[iu[0]] == g_tr[iu[1]]
    S_fuera = S.copy()
    S_fuera[g_tr[:, None] == g_tr[None, :]] = -1
    mx = {'train': (S_fuera.max(1), S_fuera.argmax(1))}
    for s in ('val', 'test'):
        sim = np.einsum('vnd,md->vnm', emb[s], E).max(0)
        mx[s] = (sim.max(1), sim.argmax(1))
        uno = emb[s][0] @ E.T
        mx[s + '_original'] = (uno.max(1), uno.argmax(1))
    R['referencia'] = {
        'coseno_pares_mismo_grupo_dhash_p5_p10_p50': [round(float(np.percentile(S[iu][mismo], q)), 4) for q in (5, 10, 50)],
        'pares_mismo_grupo_dhash': int(mismo.sum()),
        'nota': 'train: vista original, sin su grupo. *_original: vista original (comparable con train). '
                'val y test: máximo sobre 8 variantes (cota de rotaciones y volteos; sube por construcción).'}
    R['coseno_maximo_contra_train'] = {s: resumen_max(mx[s][0]) for s in ('train', 'val_original', 'test_original', 'val', 'test')}

    # Los grupos dHash no separan sesiones: pares de train de grupos distintos por encima del umbral.
    cruz = (S[iu] > CASI_DUPLICADO) & ~mismo
    con_vecino = (S_fuera > CASI_DUPLICADO).any(1)
    grupos_union = len({g_tr[i] for i in np.where(con_vecino)[0]})
    R['grupos_dhash_no_separan_sesiones'] = {
        'umbral': CASI_DUPLICADO,
        'pares_train_de_grupos_distintos_sobre_umbral': int(cruz.sum()),
        'imagenes_train_con_casi_duplicado_en_otro_grupo': int(con_vecino.sum()),
        'fraccion_imagenes_train': round(float(con_vecino.mean()), 4),
        'grupos_train_afectados': grupos_union, 'grupos_train': int(len(set(g_tr))),
        'lectura': 'Los grupos se formaron con dHash/aHash, que no ven rotaciones, recortes ni cambios de encuadre; '
                   'fotos casi iguales de una misma sesión quedan en grupos distintos dentro de train y entre particiones.'}

    OUT.mkdir(parents=True, exist_ok=True)
    casi = {}
    for s in ('val', 'test'):
        m, j = mx[s]
        filas = [{'id': f['id'], 'group_id': f['group_id'], 'train_mas_parecida': man['train'][j[i]]['id'],
                  'coseno': round(float(m[i]), 4)} for i, f in enumerate(man[s])]
        (OUT / f'similitud_{s}_train.json').write_text(json.dumps(filas, indent=1), encoding='utf-8')
        sobre = [x for x in filas if x['coseno'] > CASI_DUPLICADO]
        grupos = sorted({x['group_id'] for x in sobre})
        casi[s] = {'imagenes': len(sobre), 'grupos': len(grupos), 'grupos_total': len({f['group_id'] for f in man[s]}),
                   'lista_grupos': grupos, 'pares': sobre}
    R['casi_duplicados_contra_train'] = {'umbral': CASI_DUPLICADO, 'criterio': 'coseno máximo sobre 8 variantes > umbral', **casi}
    (OUT / 'grupos_casi_duplicado.json').write_text(json.dumps(
        {'umbral': CASI_DUPLICADO, 'val': casi['val']['lista_grupos'], 'test': casi['test']['lista_grupos']}, indent=1),
        encoding='utf-8')
    R['lectura'] = [
        'Validación y test se parecen a train como train se parece a sí mismo entre grupos: la partición por grupos no '
        'añade fuga respecto de lo que ya hay entre grupos de train.',
        'Pero los grupos dHash no separan sesiones: hay fotos casi iguales en grupos distintos, también entre particiones.',
        'El Dice de test mide generalización a fotos nuevas de las mismas sesiones y escenas, no a sesiones nuevas.',
        'La partición congelada no se modifica; la sensibilidad sin los grupos de test con casi duplicado está en '
        'resultados/fuga_embeddings_e2/sensibilidad_*.json.']
    (OUT / 'RESUMEN.json').write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: R[k] for k in ('particion', 'grupos_dhash_no_separan_sesiones')}, indent=1, ensure_ascii=False))
    print({s: (casi[s]['imagenes'], casi[s]['grupos']) for s in casi})


if __name__ == '__main__':
    main()
