"""Shift train/validación: KS por rasgo, MMD y clasificador adversarial (checklist 15). Complemento exploratorio.

El análisis ya hecho (completar_analisis_locales_e2.py) es train/test con 200 permutaciones. Este hace el par que
faltaba, train/validación, con los MISMOS descriptores por imagen (resultados/complementos_locales_e2/
descriptores_imagen.json, sin máscara) y el MISMO clasificador y protocolo, y añade dos pruebas que no dependen de
un clasificador: KS por rasgo con Holm y MMD con núcleo gaussiano. Validación ya se usó para elegir checkpoints, así
que un shift train/val afecta a la selección, no al test.

Unidad: grupo (media de los descriptores de sus imágenes). Rasgos de máscara (fracción de primer plano, regiones,
perímetro) solo entran en KS y MMD de la variante `con_mascara`; el adversarial usa solo imagen, como el original.

  python scripts/shift_train_val_e2.py [--permutaciones 200] [--origen RUTA/eda_ampliado.json]

Escribe resultados/shift_train_val_e2/RESUMEN.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

P = Path(__file__).resolve().parents[1]
C = P / 'resultados/complementos_locales_e2'
OUT = P / 'resultados/shift_train_val_e2'
SEED = 20261120
# Orden de los 61 descriptores de completar_analisis_locales_e2.py: media RGB (3), desviación RGB (3), histograma de
# 16 bins por canal (48), gradiente medio, desviación de gris, aspecto del lienzo válido.
NOMBRES = (['media_R', 'media_G', 'media_B', 'desv_R', 'desv_G', 'desv_B']
           + [f'hist_{c}_{k:02d}' for c in 'RGB' for k in range(16)] + ['gradiente_medio', 'contraste_gris', 'aspecto'])
INTERPRETABLES = ['media_R', 'media_G', 'media_B', 'desv_R', 'desv_G', 'desv_B', 'gradiente_medio', 'contraste_gris', 'aspecto']
MASCARA = ['foreground_fraction', 'regions', 'perimeter_pixels']


def leer(p):
    return json.loads(p.read_text(encoding='utf-8'))


def por_grupo(filas, campos):
    grupos = sorted({r['group_id'] for r in filas})
    idx = {g: [r for r in filas if r['group_id'] == g] for g in grupos}
    return grupos, np.array([[np.mean([r[c] for r in idx[g]]) for c in campos] for g in grupos])


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); out = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); out[i] = run
    return out


def mmd2(x, y, ancho):
    z = np.vstack([x, y]); d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1); K = np.exp(-d2 / ancho)
    n = len(x); kxx, kyy, kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
    return kxx.sum() / (n * (n - 1)) + kyy.sum() / (len(y) * (len(y) - 1)) - 2 * kxy.mean()


def prueba_mmd(x, y, perm, rng):
    z = StandardScaler().fit_transform(np.vstack([x, y]))
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    ancho = float(np.median(d2[np.triu_indices(len(z), 1)]))
    n = len(x); obs = mmd2(z[:n], z[n:], ancho); nulo = []
    for _ in range(perm):
        k = rng.permutation(len(z)); nulo.append(mmd2(z[k[:n]], z[k[n:]], ancho))
    return {'mmd2': float(obs), 'p_permutacion': float((1 + sum(v >= obs for v in nulo)) / (perm + 1)),
            'permutaciones': perm, 'nucleo': 'gaussiano, ancho = mediana de d² (rasgos estandarizados)'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--permutaciones', type=int, default=200)
    ap.add_argument('--origen', type=Path, help='eda_ampliado.json de la auditoría de origen (rasgos propios, misma partición)')
    a = ap.parse_args(argv)
    desc = [r for r in leer(C / 'descriptores_imagen.json') if r['split'] in ('train', 'val')]
    if len(desc[0]['features']) != len(NOMBRES):
        raise SystemExit(f"Se esperaban {len(NOMBRES)} descriptores y hay {len(desc[0]['features'])}")
    filas = [{'group_id': r['group_id'], 'split': r['split'], **dict(zip(NOMBRES, r['features']))} for r in desc]
    eda = {r['id']: r for r in leer(C / 'eda_por_imagen.json')}
    for r, d in zip(filas, desc):
        r.update({k: eda[d['id']][k] for k in MASCARA})
    grupos, X = por_grupo(filas, NOMBRES)
    _, M = por_grupo(filas, MASCARA)
    y = np.array([int(next(r['split'] for r in filas if r['group_id'] == g) == 'val') for g in grupos])
    rng = np.random.default_rng(SEED)

    # 1) KS por rasgo interpretable (y de máscara), por grupo, con Holm.
    cols = [NOMBRES.index(c) for c in INTERPRETABLES]
    ks = []
    for nombre, v in [(c, X[:, j]) for c, j in zip(INTERPRETABLES, cols)] + [(c, M[:, j]) for j, c in enumerate(MASCARA)]:
        r = stats.ks_2samp(v[y == 0], v[y == 1])
        ks.append({'rasgo': nombre, 'D': float(r.statistic), 'p': float(r.pvalue)})
    for f, ph in zip(ks, holm([f['p'] for f in ks])):
        f['p_holm'] = float(ph)

    # 2) MMD: solo imagen (61 descriptores) y con los rasgos de máscara.
    mmd = {'solo_imagen': prueba_mmd(X[y == 0], X[y == 1], a.permutaciones, rng),
           'con_mascara': prueba_mmd(np.hstack([X, M])[y == 0], np.hstack([X, M])[y == 1], a.permutaciones, rng)}

    # 3) Adversarial: mismo clasificador, folds y semilla que el train/test original.
    def auc(labels):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight='balanced', max_iter=1000, random_state=SEED))
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        prob = cross_val_predict(pipe, X, labels, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
        return float(roc_auc_score(labels, prob)), prob
    with threadpool_limits(limits=2):
        obs, prob = auc(y); rng2 = np.random.default_rng(SEED)
        nulo = [auc(rng2.permutation(y))[0] for _ in range(a.permutaciones)]
    bs = []
    for _ in range(5000):
        ix = rng2.integers(len(y), size=len(y))
        if len(np.unique(y[ix])) == 2:
            bs.append(roc_auc_score(y[ix], prob[ix]))
    adv = {'auc_crossfit': obs, 'IC95_condicional_predicciones': np.quantile(bs, [.025, .975]).tolist(),
           'p_permutacion': float((1 + sum(v >= obs for v in nulo)) / (len(nulo) + 1)),
           'auc_nulo_p95': float(np.quantile(nulo, .95)), 'permutaciones': a.permutaciones,
           'modelo': 'StandardScaler + LogisticRegression C=1 balanceada, 5 folds estratificados, semilla 20261120'}

    origen = None
    if a.origen:
        o = leer(a.origen)
        origen = {'archivo': a.origen.name, 'rasgos': 'ancho, alto, aspecto, luminancia, contraste, saturación, nitidez, '
                  'fracción etiquetada, número de regiones, área media de región; por grupo',
                  **{k: o[k] for k in ('ks_por_rasgo_grupos', 'mmd_grupos', 'adversarial_imagenes_cv_por_grupo',
                                       'categorias_train_vs_val')}}
    tt = leer(C / 'validacion_adversarial.json')
    R = {'estado': 'Exploratorio; complemento del train/test ya hecho, no lo repite.',
         'unidad': 'grupo', 'grupos': {'train': int((y == 0).sum()), 'val': int((y == 1).sum())},
         'ks_por_rasgo': ks, 'mmd': mmd, 'adversarial_train_val': adv,
         'adversarial_train_test_existente': {k: tt[k] for k in ('auc_crossfit', 'CI95_conditional_predictions', 'p_permutation_MC')},
         'auditoria_de_origen_train_val': origen,
         'lectura': 'Sin rechazo no demuestra ausencia de shift: con ~690 grupos la potencia para cambios pequeños es '
                    'limitada y los descriptores son globales (color, textura, geometría), no semánticos.'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'RESUMEN.json').write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: R[k] for k in ('grupos', 'mmd', 'adversarial_train_val')}, indent=1, ensure_ascii=False))
    print('KS min p_holm', min(f['p_holm'] for f in ks), [f['rasgo'] for f in ks if f['p'] < .05])


if __name__ == '__main__':
    main()
