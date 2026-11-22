"""Resumen de la búsqueda de arquitectura (secciones 14.4 a 14.7 de la guía). Solo lee JSON; no entrena.

Escrito antes de correr la búsqueda: los análisis y la regla de selección están fijados aquí y en
config/busqueda_arquitectura_e2.json. Usa numpy y scipy (están en el entorno de Colab); las figuras
se hacen aparte, en local, a partir de los CSV que deja.

Por familia:
  fase 1   superficie (CSV), modelo lineal de efectos principales e interacciones de dropout con IC95,
           F por factor con Holm, proporción de varianza, tendencia sobre log2 w, Wilcoxon pareado
           sin dropout frente a con dropout (125 bloques), frente de Pareto, brecha de pérdidas
  ruido    sigma por semilla (repetición) frente a la dispersión entre arquitecturas
  fase 2   media, DE e IC95 por finalista, ganador con la regla de 14.5, conjunto equivalente (TOST),
           Spearman fase 1 - fase 2, borde del espacio, ganador frente a la heredada

Uso: python resumir_busqueda_arquitectura_e2.py --output <carpeta de la búsqueda> [--destino <carpeta>]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT/'config/busqueda_arquitectura_e2.json'
FAMILIAS = ('fcn', 'unet', 'segnet')
W = (16, 32, 64, 128, 256)


def leer(carpeta):
    out = []
    for p in sorted(Path(carpeta).glob('*/result.json')):
        r = json.loads(p.read_text(encoding='utf-8'))
        if r.get('completed'):
            r['_dir'] = p.parent
            out.append(r)
    return out


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); orden = np.argsort(ps); adj = np.empty(m); acum = 0.
    for i, j in enumerate(orden):
        acum = max(acum, min(1., (m - i)*ps[j])); adj[j] = acum
    return adj.tolist()


# ---------------------------------------------------------------- fase 1: modelo lineal
TERMINOS = ['w1', 'w2', 'w3', 'b1', 'b2', 'b3', 'b1:b2', 'b1:b3', 'b2:b3']


def columnas(filas, terminos):
    cols, nombres = [np.ones(len(filas))], ['intercepto']
    for t in terminos:
        if t in ('w1', 'w2', 'w3'):
            i = int(t[1]) - 1
            for w in W[1:]:
                cols.append(np.array([1. if r['w'][i] == w else 0. for r in filas])); nombres.append(f'{t}={w}')
        elif ':' in t:
            i, j = (int(x[1]) - 1 for x in t.split(':'))
            cols.append(np.array([float(r['b'][i]*r['b'][j]) for r in filas])); nombres.append(t)
        else:
            i = int(t[1]) - 1
            cols.append(np.array([float(r['b'][i]) for r in filas])); nombres.append(t)
    return np.column_stack(cols), nombres


def mco(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X@beta
    return beta, float(res@res)


def modelo_lineal(filas):
    y = np.array([r['best_val_group_Dice'] for r in filas])
    X, nombres = columnas(filas, TERMINOS)
    beta, rss = mco(X, y); n, p = X.shape; gl = n - p; s2 = rss/gl
    cov = s2*np.linalg.inv(X.T@X); se = np.sqrt(np.diag(cov)); tc = stats.t.ppf(.975, gl)
    coef = [{'termino': nm, 'estimado': float(b), 'ee': float(e), 'ic95': [float(b - tc*e), float(b + tc*e)]}
            for nm, b, e in zip(nombres, beta, se)]
    sct = float(np.sum((y - y.mean())**2)); factores = []
    for t in TERMINOS:
        if ':' in t:  # interacción: se quita del modelo completo
            sin = [x for x in TERMINOS if x != t]; con = TERMINOS
        else:          # efecto principal, tipo II: sin las interacciones que lo contienen
            sin = [x for x in TERMINOS if t not in x.split(':')]; con = sin + [t]
        _, rss_sin = mco(columnas(filas, sin)[0], y)
        Xc = columnas(filas, con)[0]; _, rss_con = mco(Xc, y)
        gl_t = Xc.shape[1] - columnas(filas, sin)[0].shape[1]
        ss = rss_sin - rss_con; F = (ss/gl_t)/s2
        factores.append({'factor': t, 'gl': gl_t, 'ss': ss, 'proporcion_varianza': ss/sct, 'F': F,
                         'p': float(stats.f.sf(F, gl_t, gl))})
    for f, ph in zip(factores, holm([f['p'] for f in factores])):
        f['p_holm'] = ph
    return {'n': n, 'gl_residual': gl, 'r2': 1 - rss/sct, 'sigma_residual': float(np.sqrt(s2)),
            'coeficientes': coef, 'factores': factores,
            'nota': 'Con una semilla, el residuo mezcla ruido por semilla e interacciones de anchos no modeladas.'}


def tendencia(filas):
    """y ~ sum_i (a_i log2 w_i + q_i (log2 w_i)^2) + dropout; q_i < 0 indica rendimientos decrecientes."""
    y = np.array([r['best_val_group_Dice'] for r in filas])
    cols = [np.ones(len(filas))]; nombres = ['intercepto']
    for i in range(3):
        l = np.array([np.log2(r['w'][i]) - 6 for r in filas])  # centrado en 64
        cols += [l, l**2]; nombres += [f'log2_w{i+1}', f'log2_w{i+1}^2']
    for i in range(3):
        cols.append(np.array([float(r['b'][i]) for r in filas])); nombres.append(f'b{i+1}')
    X = np.column_stack(cols); beta, rss = mco(X, y); gl = len(y) - X.shape[1]; s2 = rss/gl
    se = np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
    return [{'termino': nm, 'estimado': float(b), 'ee': float(e), 't': float(b/e),
             'p': float(2*stats.t.sf(abs(b/e), gl))} for nm, b, e in zip(nombres, beta, se)]


def wilcoxon_dropout(filas):
    por_w = defaultdict(dict)
    for r in filas:
        por_w[tuple(r['w'])][tuple(r['b'])] = r['best_val_group_Dice']
    d = []
    for w, pat in por_w.items():
        if (0, 0, 0) in pat and len(pat) == 8:
            d.append(pat[(0, 0, 0)] - np.mean([v for b, v in pat.items() if b != (0, 0, 0)]))
    d = np.array(d)
    if len(d) < 6:
        return {'n_bloques': len(d), 'nota': 'Pocos bloques completos'}
    w = stats.wilcoxon(d)
    rk = stats.rankdata(np.abs(d[d != 0])); pos = rk[d[d != 0] > 0].sum(); neg = rk[d[d != 0] < 0].sum()
    return {'n_bloques': int(len(d)), 'mediana_sin_menos_con': float(np.median(d)), 'media': float(d.mean()),
            'W': float(w.statistic), 'p': float(w.pvalue), 'rango_biserial': float((pos - neg)/(pos + neg)),
            'lectura': 'Positivo: sin dropout rinde más que el promedio de los 7 patrones con dropout'}


def pareto(filas, costo):
    pts = sorted(filas, key=lambda r: (r[costo], -r['best_val_group_Dice']))
    frente, mejor = [], -1.
    for r in pts:
        if r['best_val_group_Dice'] > mejor:
            frente.append({'arq': r['arq'], 'w': r['w'], 'b': r['b'], costo: r[costo],
                           'dice': r['best_val_group_Dice']}); mejor = r['best_val_group_Dice']
    return frente


def brecha_fase1(filas):
    tabla = defaultdict(list)
    for r in filas:
        h = r['history'][-1]
        total = sum(r['w']); tramo = 'bajo' if total <= 96 else ('medio' if total <= 320 else 'alto')
        tabla[(tramo, ''.join(map(str, r['b'])))].append(h['val_loss'] - h['train_loss'])
    return [{'ancho_total': k[0], 'dropout': k[1], 'n': len(v), 'val_menos_train_media': float(np.mean(v))}
            for k, v in sorted(tabla.items())]


# ---------------------------------------------------------------- fase 2
def ic_media(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return [float('nan')]*2
    h = stats.t.ppf(.975, n - 1)*x.std(ddof=1)/np.sqrt(n)
    return [float(x.mean() - h), float(x.mean() + h)]


def tost_pareado(d, delta, alfa=.05):
    d = np.asarray(d, float); n = len(d); se = d.std(ddof=1)/np.sqrt(n)
    if se == 0:
        return {'equivalente': bool(abs(d.mean()) < delta), 'p': 0. if abs(d.mean()) < delta else 1.}
    p1 = stats.t.sf((d.mean() + delta)/se, n - 1); p2 = stats.t.cdf((d.mean() - delta)/se, n - 1)
    p = float(max(p1, p2))
    return {'equivalente': p < alfa, 'p': p}


def por_imagen(r):
    p = r['_dir']/'best_val_predictions.json'
    if not p.exists():
        return None
    return {d['id']: (d['group_id'], d['Dice']) for d in json.loads(p.read_text(encoding='utf-8'))['validation']}


def bootstrap_grupos(a_runs, b_runs, B=2000, semilla=20261122):
    """Diferencia del Dice medio por grupo (promedio de las semillas) entre dos arquitecturas."""
    pa = [por_imagen(r) for r in a_runs]; pb = [por_imagen(r) for r in b_runs]
    if any(x is None for x in pa + pb):
        return None
    ids = sorted(pa[0]); grupos = defaultdict(list)
    for i in ids:
        da = np.mean([p[i][1] for p in pa]); db = np.mean([p[i][1] for p in pb])
        grupos[pa[0][i][0]].append(da - db)
    g = np.array([np.mean(v) for v in grupos.values()])
    rng = np.random.default_rng(semilla)
    boots = np.array([g[rng.integers(0, len(g), len(g))].mean() for _ in range(B)])
    return {'diferencia': float(g.mean()), 'ic95': [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            'grupos': int(len(g)), 'B': B}


def fase2_familia(runs, fin, cfg, f1_por_arq, delta):
    por_arq = defaultdict(list)
    for r in runs:
        por_arq[r['arq']].append(r)
    semillas = sorted({r['semilla'] for r in runs})
    tabla = []
    for arq, rs in por_arq.items():
        rs.sort(key=lambda r: r['semilla']); y = [r['best_val_group_Dice'] for r in rs]
        tabla.append({'arq': arq, 'w': rs[0]['w'], 'b': rs[0]['b'], 'n_semillas': len(rs),
                      'media': float(np.mean(y)), 'de': float(np.std(y, ddof=1)) if len(y) > 1 else None,
                      'ic95': ic_media(y), 'por_semilla': {str(r['semilla']): r['best_val_group_Dice'] for r in rs},
                      'parametros': rs[0]['parameters'], 'macs': rs[0]['macs'],
                      'segundos_medios': float(np.mean([r['seconds_total'] for r in rs])),
                      'epocas_medias': float(np.mean([r['epochs_completed'] for r in rs])),
                      'brecha_train_val_media': float(np.mean([r['brecha_train_val'] for r in rs])),
                      'dice_fase1': f1_por_arq.get(arq),
                      'papel': ('referencia' if arq == fin['referencia'] else
                                'heredada' if arq == fin['heredada'] else 'finalista')})
    completos = [t for t in tabla if t['n_semillas'] == len(semillas)]
    completos.sort(key=lambda t: -t['media'])
    if not completos:
        return {'tabla': tabla, 'nota': 'Sin finalistas con todas las semillas'}
    mejor = completos[0]
    empatados = [t for t in completos if mejor['media'] - t['media'] < delta
                 or t['ic95'][1] >= mejor['ic95'][0]]
    ganador = min(empatados, key=lambda t: (t['parametros'], -t['media']))
    equivalentes = []
    for t in completos:
        d = [mejor['por_semilla'][s] - t['por_semilla'][s] for s in mejor['por_semilla']]
        if t is mejor or tost_pareado(d, delta)['equivalente']:
            equivalentes.append(t['arq'])
    top = [t for t in completos if t['arq'] in fin['_top']]  # solo los k finalistas, sin los controles
    x = [t['dice_fase1'] for t in top]; y = [t['media'] for t in top]
    rho = stats.spearmanr(x, y) if len(x) >= 3 else None
    her = next((t for t in completos if t['arq'] == fin['heredada']), None)
    comp = None
    if her is not None:
        d = [ganador['por_semilla'][s] - her['por_semilla'][s] for s in ganador['por_semilla']]
        comp = {'ganador': ganador['arq'], 'heredada': her['arq'], 'diferencia_media_pareada': float(np.mean(d)),
                'ic95_t': ic_media(d), 'tost_delta': tost_pareado(d, delta),
                'bootstrap_grupos': bootstrap_grupos(por_arq[ganador['arq']], por_arq[her['arq']]),
                'heredada_en_conjunto_equivalente_al_mejor': her['arq'] in equivalentes}
        dm = comp['diferencia_media_pareada']
        comp['lectura'] = ('ganador es la heredada' if ganador['arq'] == her['arq'] else
                           'la heredada es equivalente al mejor (TOST ±δ): herencia justificada'
                           if comp['heredada_en_conjunto_equivalente_al_mejor'] else
                           'el ganador supera a la heredada por δ o más' if dm >= delta and comp['ic95_t'][0] > 0 else
                           'el ganador supera a la heredada por menos de δ' if comp['ic95_t'][0] > 0 else
                           'sin diferencia demostrada ni equivalencia')
    borde = {'algun_ancho_256': 256 in ganador['w'], 'algun_ancho_16': 16 in ganador['w'],
             'todas_256': all(w == 256 for w in ganador['w'])}
    de_sem = [t['de'] for t in completos if t['de'] is not None]
    return {'semillas': semillas, 'tabla': sorted(tabla, key=lambda t: -t['media']),
            'mejor_media': mejor['arq'], 'empate_tecnico': [t['arq'] for t in empatados],
            'ganador': {k: ganador[k] for k in ('arq', 'w', 'b', 'media', 'ic95', 'parametros', 'brecha_train_val_media')},
            'regla': cfg['seleccion']['regla'], 'conjunto_equivalente_tost': equivalentes,
            'tamano_conjunto_equivalente': len(equivalentes),
            'spearman_fase1_fase2': None if rho is None else {'rho': float(rho.statistic), 'p': float(rho.pvalue), 'n': len(x)},
            'borde_del_espacio': borde, 'de_por_semilla_mediana': float(np.median(de_sem)) if de_sem else None,
            'comparacion_heredada': comp}


# ---------------------------------------------------------------- principal
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, help='Carpeta de la búsqueda (Drive o copia local)')
    p.add_argument('--destino', help='Dónde escribir RESUMEN.*; por defecto, --output')
    args = p.parse_args(argv)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8')); delta = cfg['seleccion']['delta']
    base = Path(args.output); dest = Path(args.destino or base); dest.mkdir(parents=True, exist_ok=True)
    s1 = cfg['fase1']['semilla']; s_extra = cfg['ruido']['semilla_extra']
    f1 = leer(base/'fase1'); f2 = leer(base/'fase2')
    fin = json.loads((base/'finalistas.json').read_text(encoding='utf-8')) if (base/'finalistas.json').exists() else None
    ajuste = json.loads((base/'ajuste_previo.json').read_text(encoding='utf-8')) if (base/'ajuste_previo.json').exists() else None
    resumen = {'experimento': cfg['experimento'], 'ajuste_previo': ajuste, 'k': fin and fin['k'],
               'test_evaluado': False, 'familias': {}}
    lineas_csv = ['familia,arq,w1,w2,w3,b1,b2,b3,semilla,dice,parametros,macs,segundos,val_loss,train_loss']
    for f in FAMILIAS:
        filas = [r for r in f1 if r['familia'] == f and r['semilla'] == s1]
        if not filas:
            continue
        for r in f1:
            if r['familia'] == f:
                h = r['history'][-1]
                lineas_csv.append(','.join(map(str, [f, r['arq'], *r['w'], *r['b'], r['semilla'],
                                                     r['best_val_group_Dice'], r['parameters'], r['macs'],
                                                     round(r['seconds_total'], 2), h['val_loss'], h['train_loss']])))
        y = np.array([r['best_val_group_Dice'] for r in filas])
        extra = {r['arq']: r['best_val_group_Dice'] for r in f1 if r['familia'] == f and r['semilla'] == s_extra}
        base1 = {r['arq']: r['best_val_group_Dice'] for r in filas}
        d = np.array([base1[a] - v for a, v in extra.items() if a in base1])
        s_sem = float(np.sqrt(np.mean(d**2)/2)) if len(d) else None
        fam = {'fase1': {'n': len(filas), 'media': float(y.mean()), 'de_entre_arquitecturas': float(y.std(ddof=1)) if len(y) > 1 else None,
                         'min': float(y.min()), 'max': float(y.max()),
                         'top10': [{'arq': r['arq'], 'w': r['w'], 'b': r['b'], 'dice': r['best_val_group_Dice'],
                                    'parametros': r['parameters']}
                                   for r in sorted(filas, key=lambda r: (-r['best_val_group_Dice'], r['parameters']))[:10]]},
               'ruido': {'pares': int(len(d)), 'sigma_semilla_fase1': s_sem,
                         'correlacion_semillas': float(np.corrcoef([base1[a] for a in extra if a in base1],
                                                                   [extra[a] for a in extra if a in base1])[0, 1]) if len(d) > 2 else None}}
        if len(filas) >= 30:
            fam['fase1']['modelo_lineal'] = modelo_lineal(filas)
            fam['fase1']['tendencia_log2_ancho'] = tendencia(filas)
            fam['fase1']['wilcoxon_dropout'] = wilcoxon_dropout(filas)
            fam['fase1']['pareto_parametros'] = pareto([{**r, 'parametros': r['parameters']} for r in filas], 'parametros')
            fam['fase1']['pareto_segundos'] = pareto([{**r, 'segundos': r['seconds_total']} for r in filas], 'segundos')
            fam['fase1']['brecha_perdidas'] = brecha_fase1(filas)
        if fin:
            fam['finalistas'] = {k: fin['familias'][f][k] for k in ('finalistas', 'fase2', 'r',
                                                                     'sigma_semilla', 'sigma_arquitectura',
                                                                     'rango_referencia', 'rango_heredada')}
            runs = [r for r in f2 if r['familia'] == f]
            if runs:
                fin_f = {**fin, '_top': fin['familias'][f]['finalistas']}
                fam['fase2'] = fase2_familia(runs, fin_f, cfg, base1, delta)
                if fam['fase2'].get('de_por_semilla_mediana') and fam['fase1']['de_entre_arquitecturas']:
                    fam['sensibilidad'] = {
                        'de_entre_arquitecturas_fase1': fam['fase1']['de_entre_arquitecturas'],
                        'de_por_semilla_fase2_mediana': fam['fase2']['de_por_semilla_mediana'],
                        'cociente': fam['fase1']['de_entre_arquitecturas']/fam['fase2']['de_por_semilla_mediana'],
                        'regla': 'Si el cociente es cercano a 1, la arquitectura es irrelevante para este problema (14.6).'}
        resumen['familias'][f] = fam
    (dest/'fase1.csv').write_text('\n'.join(lineas_csv) + '\n', encoding='utf-8')
    (dest/'RESUMEN.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    md = ['# Búsqueda de arquitectura de las compactas: resumen automático', '',
          'Solo validación; el test no se carga. Generado por `scripts/resumir_busqueda_arquitectura_e2.py`.', '']
    if ajuste:
        md += [f'Ajuste previo (tasa elegida por familia): {ajuste["lr_elegida"]}', '']
    md += ['| familia | n fase 1 | media | DE entre arq. | σ semilla | k | ganador | media fase 2 | IC95 | vs heredada |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for f, fam in resumen['familias'].items():
        g = fam.get('fase2', {}).get('ganador'); c = fam.get('fase2', {}).get('comparacion_heredada')
        md.append(f"| {f} | {fam['fase1']['n']} | {fam['fase1']['media']:.4f} | "
                  f"{(fam['fase1']['de_entre_arquitecturas'] or 0):.4f} | "
                  f"{fam['ruido']['sigma_semilla_fase1'] if fam['ruido']['sigma_semilla_fase1'] is None else round(fam['ruido']['sigma_semilla_fase1'], 4)} | "
                  f"{resumen['k']} | {g and (tuple(g['w']), tuple(g['b']))} | {g and round(g['media'], 4)} | "
                  f"{g and [round(v, 4) for v in g['ic95']]} | {c and c['lectura']} |")
    (dest/'RESUMEN.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
