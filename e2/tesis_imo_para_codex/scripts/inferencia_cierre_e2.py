"""Equivalencia (TOST), sensibilidad a supuestos, multiplicidad simultánea y sesgo del ganador.

  python scripts/inferencia_cierre_e2.py

Exploratorio, especificado después de observar test. No ejecuta modelos. Dos fuentes:

- TEST: solo los resúmenes de resultados/integracion_e2/analisis.json (los Dice por grupo del test no están en el
  repositorio). Para cada contraste red − FCN hay media de la diferencia por grupo, d_z = media/DE y n = 122 grupos;
  de ahí DE = media/d_z y un TOST t pareado exacto (gl = 121). Sin bootstrap en test: necesitaría los grupos.
- VALIDACIÓN: resultados/validacion_por_imagen_e2/*_val_por_imagen.json, 50 corridas, 218 imágenes, 122 grupos. Es el
  conjunto con el que se eligió el checkpoint de cada corrida: los niveles están sesgados al alza y las diferencias
  pueden estarlo. Se usa para lo que el test no permite sin sus grupos (bootstrap, simetría, max-T, sesgo del ganador).

Estimando, igual al del libro (scripts/analisis_integracion_e2.py): Dice medio por grupo (promedio de sus imágenes),
promediado sobre las cinco semillas fijas; contraste = media sobre grupos de la diferencia pareada por grupo.
δ = 0,02 Dice (heredado). TOST con α = 0,05 por lado, equivalente a que el IC90 quede dentro de (−δ, δ).
Salidas: resultados/inferencia_cierre_e2/inferencia.json y TABLAS.md.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

P = Path(__file__).resolve().parents[1]
OUT = P / 'resultados' / 'inferencia_cierre_e2'
VAL = P / 'resultados' / 'validacion_por_imagen_e2'
SEEDS = [11, 23, 37, 53, 71]
DELTA, ALPHA = 0.02, 0.05
B_BOOT, B_FLIP, R_SPLIT = 10000, 49999, 2000
NAMES = {'fcn': 'FCN compacta', 'segnet': 'SegNet compacta', 'unet': 'U-Net compacta',
         'deeplabv3plus': 'DeepLabv3+ R18', 'unetplusplus': 'U-Net++ R18', 'pidnet_s': 'PIDNet-S',
         'segnext_t': 'SegNeXt-T', 'segformer_b0': 'SegFormer-B0', 'transunet_2d': 'TransUNet', 'vmunet': 'VM-UNet'}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); r = np.empty(len(p))
    r[o] = np.minimum(1, np.maximum.accumulate(p[o] * (len(p) - np.arange(len(p)))))
    return r


def tost_t(media, de, n, delta=DELTA):
    """TOST t pareado: H01 Δ <= −δ y H02 Δ >= δ; p = máx de los dos unilaterales; IC90 t."""
    se = de / np.sqrt(n); gl = n - 1
    t1, t2 = (media + delta) / se, (media - delta) / se
    p1, p2 = stats.t.sf(t1, gl), stats.t.cdf(t2, gl)
    q = stats.t.ppf(1 - ALPHA, gl)
    return {'media': float(media), 'DE': float(de), 'n': int(n), 'EE': float(se), 't_inferior': float(t1),
            't_superior': float(t2), 'p_inferior': float(p1), 'p_superior': float(p2),
            'p_TOST': float(max(p1, p2)), 'IC90_t': [float(media - q * se), float(media + q * se)],
            'p_bilateral_t': float(2 * stats.t.sf(abs(media / se), gl))}


def signflip(d, seed=20261120, draws=B_FLIP):
    """Igual que el libro: bilateral sobre la media, Monte Carlo con +1."""
    d = np.asarray(d, float); obs = abs(d.mean()); rng = np.random.default_rng(seed); c = 0
    for s in range(0, draws, 1000):
        sg = rng.integers(0, 2, size=(min(1000, draws - s), len(d))) * 2 - 1
        c += int(np.count_nonzero(abs(sg @ d / len(d)) >= obs - 1e-14))
    return (c + 1) / (draws + 1)


def hodges_lehmann(d):
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def simetria(d, rng, B=4999):
    """Simetría alrededor de la mediana desconocida. Estadístico de Miao, Gel y Gastwirth (2006):
    (media − mediana)/J, J = sqrt(pi/2)·mean|x − mediana|. Nula por bootstrap simetrizado (Schuster y Barker, 1987):
    se remuestrea de la muestra reflejada alrededor de la mediana, que es simétrica por construcción."""
    def est(x):
        m = np.median(x); J = np.sqrt(np.pi / 2) * np.mean(abs(x - m))
        return (x.mean() - m) / J if J > 0 else 0.0
    obs = est(d); m = np.median(d); refl = np.concatenate([d, 2 * m - d])
    nulo = np.array([est(refl[rng.integers(len(refl), size=len(d))]) for _ in range(B)])
    asim = stats.skew(d, bias=False)
    bs = np.array([stats.skew(d[rng.integers(len(d), size=len(d))], bias=False) for _ in range(2000)])
    return {'MGG': float(obs), 'p_bootstrap_simetrizado': float((np.sum(abs(nulo) >= abs(obs)) + 1) / (B + 1)),
            'asimetria': float(asim), 'asimetria_IC95_boot': np.quantile(bs, [.025, .975]).tolist()}


def leer_validacion():
    raw, grupos = {}, None
    for f in sorted(VAL.glob('*_val_por_imagen.json')):
        j = json.loads(f.read_text(encoding='utf-8'))
        assert j['split'] == 'val'
        g = {}
        for r in j['detail']:
            g.setdefault(r['group_id'], []).append(r['Dice'])
        gm = {k: float(np.mean(v)) for k, v in g.items()}
        assert abs(np.mean(list(gm.values())) - j['val_group_Dice']) < 1e-9
        grupos = grupos or sorted(gm)
        raw.setdefault(j['model'], {})[int(j['seed'])] = np.array([gm[k] for k in grupos])
    assert all(sorted(v) == SEEDS for v in raw.values()) and len(raw) == 10
    scores = {m: np.stack([raw[m][s] for s in SEEDS]) for m in raw}   # 5 x 122
    return scores, grupos


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261122)
    ana = json.loads((P / 'resultados/integracion_e2/analisis.json').read_text(encoding='utf-8'))
    n_test = ana['friedman']['blocks']
    test_dice = {r['model']: r['Dice_test'] for r in
                 json.loads((P / 'resultados/auditoria_test_e2/AUDITORIA.json').read_text(encoding='utf-8'))['table']}

    # ---------- TEST: TOST desde los resúmenes ----------
    tost_test = []
    for c in ana['contrasts']:
        de = c['difference'] / c['d_z_group']
        r = {'red': c['model'], **tost_t(c['difference'], de, n_test),
             'IC95_bootstrap_libro': c['CI95'], 'p_intercambio_Holm9_libro': c['p_signflip_MC_Holm9'],
             'p_wilcoxon_Holm9_libro': c['p_wilcoxon_asymptotic_Holm9']}
        tost_test.append(r)
    for r, a in zip(tost_test, holm([r['p_TOST'] for r in tost_test])):
        r['p_TOST_Holm9'] = float(a); r['equivalente_Holm9'] = bool(a < ALPHA)
        r['equivalente_sin_ajuste'] = r['p_TOST'] < ALPHA
    for r, a in zip(tost_test, holm([r['p_bilateral_t'] for r in tost_test])):
        r['p_bilateral_t_Holm9'] = float(a)
    # Concordancia de decisiones (α = 0,05, Holm 9) entre t, intercambio y Wilcoxon en test.
    concord_test = [{'red': r['red'], 't': r['p_bilateral_t_Holm9'] < ALPHA,
                     'intercambio': r['p_intercambio_Holm9_libro'] < ALPHA,
                     'wilcoxon': r['p_wilcoxon_Holm9_libro'] < ALPHA} for r in tost_test]

    # ---------- VALIDACIÓN ----------
    scores, grupos = leer_validacion(); ng = len(grupos)
    gmean = {m: a.mean(axis=0) for m, a in scores.items()}
    models = sorted(gmean, key=lambda m: -gmean[m].mean())
    ix = rng.integers(ng, size=(B_BOOT, ng))

    def contraste_val(a, b):
        d = gmean[a] - gmean[b]; boot = d[ix].mean(axis=1); r = tost_t(d.mean(), d.std(ddof=1), ng)
        lo, hi = np.quantile(boot, [.05, .95])
        r.update({'red': a, 'referencia': b, 'd_z': float(d.mean() / d.std(ddof=1)),
                  'IC90_bootstrap': [float(lo), float(hi)], 'IC95_bootstrap': np.quantile(boot, [.025, .975]).tolist(),
                  'p_TOST_bootstrap': float(max(np.mean(boot <= -DELTA), np.mean(boot >= DELTA))),
                  'equivalente_bootstrap_IC90': bool(lo > -DELTA and hi < DELTA)})
        return r, d

    tost_val = {'vs_fcn': [], 'vs_transunet': []}
    difs = {}
    for ref, clave in [('fcn', 'vs_fcn'), ('transunet_2d', 'vs_transunet')]:
        for m in models:
            if m == ref:
                continue
            r, d = contraste_val(m, ref); tost_val[clave].append(r)
            if ref == 'fcn':
                difs[m] = d
        for k in ['p_TOST', 'p_TOST_bootstrap']:
            for r, a in zip(tost_val[clave], holm([r[k] for r in tost_val[clave]])):
                r[k + '_Holm9'] = float(a)
        for r in tost_val[clave]:
            r['equivalente_t_Holm9'] = r['p_TOST_Holm9'] < ALPHA
            r['equivalente_bootstrap_Holm9'] = r['p_TOST_bootstrap_Holm9'] < ALPHA

    # Sensibilidad a supuestos: misma pregunta (red − FCN), cinco procedimientos, en validación.
    sens = []
    for m, d in difs.items():
        nz = d[d != 0]
        sens.append({'red': m, 'media': float(d.mean()), 'mediana': float(np.median(d)), 'hodges_lehmann': hodges_lehmann(d),
                     'p_t': float(stats.ttest_1samp(d, 0).pvalue), 'p_intercambio': signflip(d),
                     'p_wilcoxon': float(stats.wilcoxon(d, method='approx').pvalue),
                     'p_signo': float(stats.binomtest(int((nz > 0).sum()), len(nz)).pvalue),
                     'shapiro_p': float(stats.shapiro(d).pvalue), 'simetria': simetria(d, rng)})
    for k in ['p_t', 'p_intercambio', 'p_wilcoxon', 'p_signo']:
        for r, a in zip(sens, holm([r[k] for r in sens])):
            r[k + '_Holm9'] = float(a)
    for r in sens:
        dec = {k: r[k + '_Holm9'] < ALPHA for k in ['p_t', 'p_intercambio', 'p_wilcoxon', 'p_signo']}
        r['decisiones_Holm9'] = dec; r['concuerdan'] = len(set(dec.values())) == 1
        r['concuerdan_media'] = dec['p_t'] == dec['p_intercambio']
        r['concuerdan_localizacion'] = dec['p_wilcoxon'] == dec['p_signo']

    # Dependencia residual entre grupos vecinos (identificadores consecutivos del corpus): autocorrelación de lag 1
    # de las diferencias ordenadas por grupo, con p por permutación.
    def lag1(x):
        x = x - x.mean(); return float((x[:-1] * x[1:]).sum() / (x * x).sum())
    dep = []
    for m, d in difs.items():
        obs = lag1(d); nulo = np.array([lag1(rng.permutation(d)) for _ in range(4999)])
        dep.append({'red': m, 'autocorr_lag1': obs, 'p_perm': float((np.sum(abs(nulo) >= abs(obs)) + 1) / 5000)})
    for r, a in zip(dep, holm([r['p_perm'] for r in dep])):
        r['p_perm_Holm9'] = float(a)

    # Influencia y semillas: dejar un grupo fuera y dejar una semilla fuera; ¿cambia el signo o la decisión t sin
    # ajuste a α = 0,05?
    infl = []
    for m, d in difs.items():
        base = stats.ttest_1samp(d, 0).pvalue < ALPHA
        loo = np.array([np.delete(d, k).mean() for k in range(ng)])
        loo_p = np.array([stats.ttest_1samp(np.delete(d, k), 0).pvalue for k in range(ng)])
        sem = []
        for s in range(5):
            ds = np.delete(scores[m], s, 0).mean(0) - np.delete(scores['fcn'], s, 0).mean(0)
            sem.append({'sin_semilla': SEEDS[s], 'media': float(ds.mean()), 'p_t': float(stats.ttest_1samp(ds, 0).pvalue)})
        infl.append({'red': m, 'media_loo_rango': [float(loo.min()), float(loo.max())],
                     'decision_cambia_dejando_un_grupo': int(np.sum((loo_p < ALPHA) != base)),
                     'dejando_una_semilla': sem,
                     'decision_cambia_dejando_una_semilla': int(sum((x['p_t'] < ALPHA) != base for x in sem))})

    # Multiplicidad simultánea: max-T por cambio de signo (Westfall y Young) sobre los nueve contrastes con FCN, que
    # respeta su correlación; e IC95 simultáneos por bootstrap del máximo |t| estudentizado.
    D = np.stack([difs[m] for m in difs], axis=1)                     # 122 x 9
    tobs = D.mean(0) / (D.std(0, ddof=1) / np.sqrt(ng))
    maxt = np.empty(B_FLIP)
    for s in range(0, B_FLIP, 1000):
        k = min(1000, B_FLIP - s); sg = rng.integers(0, 2, size=(k, ng, 1)) * 2 - 1
        Ds = sg * D; maxt[s:s + k] = np.abs(Ds.mean(1) / (Ds.std(1, ddof=1) / np.sqrt(ng))).max(1)
    p_maxt = [(np.sum(maxt >= abs(t)) + 1) / (B_FLIP + 1) for t in tobs]   # paso único
    Db = D[ix]; tb = np.abs((Db.mean(1) - D.mean(0)) / (Db.std(1, ddof=1) / np.sqrt(ng))).max(1)
    c95 = float(np.quantile(tb, .95)); ee = D.std(0, ddof=1) / np.sqrt(ng)
    simult = [{'red': m, 't': float(tobs[k]), 'p_maxT_paso_unico': float(p_maxt[k]),
               'IC95_simultaneo': [float(D[:, k].mean() - c95 * ee[k]), float(D[:, k].mean() + c95 * ee[k])]}
              for k, m in enumerate(difs)]

    # ---------- Sesgo del ganador en validación ----------
    M = np.stack([gmean[m] for m in models], axis=1)                  # 122 x 10
    est = M.mean(0); ganador = models[int(est.argmax())]
    # (1) Bootstrap: sesgo de "el máximo de las medias" como estimador del nivel del ganador.
    sesgo_nivel, sesgo_vent, frec = [], [], {m: 0 for m in models}
    for b in range(B_BOOT):
        mb = M[ix[b]].mean(0); w = int(mb.argmax()); frec[models[w]] += 1
        sesgo_nivel.append(mb[w] - est[w])
        o = np.sort(mb); sesgo_vent.append((o[-1] - o[-2]) - (np.sort(est)[-1] - np.sort(est)[-2]))
    # (2) División en dos mitades de grupos: se elige en A y se mide en B.
    split = []
    for _ in range(R_SPLIT):
        perm = rng.permutation(ng); A, Bm = perm[:ng // 2], perm[ng // 2:]
        w = int(M[A].mean(0).argmax())
        split.append((models[w], M[A, w].mean(), M[Bm, w].mean(), M[A, w].mean() - M[A, models.index('fcn')].mean(),
                      M[Bm, w].mean() - M[Bm, models.index('fcn')].mean()))
    sp = np.array([s[1:] for s in split], float)
    frec_split = {m: sum(s[0] == m for s in split) / R_SPLIT for m in models}
    # (3) Checkpoint elegido por validación frente a test (conjunto independiente, ya observado).
    val_test = [{'red': m, 'Dice_val_grupos': float(est[models.index(m)]), 'Dice_test': float(test_dice[m]),
                 'test_menos_val': float(test_dice[m] - est[models.index(m)])} for m in models]
    tu_fcn_val = float(gmean['transunet_2d'].mean() - gmean['fcn'].mean())
    tu_fcn_test = next(c['difference'] for c in ana['contrasts'] if c['model'] == 'transunet_2d')
    ganador_val = {
        'ganador_validacion': ganador, 'Dice_ganador': float(est.max()),
        'bootstrap': {'B': B_BOOT, 'sesgo_nivel_ganador': float(np.mean(sesgo_nivel)),
                      'nivel_corregido': float(est.max() - np.mean(sesgo_nivel)),
                      'sesgo_ventaja_sobre_segundo': float(np.mean(sesgo_vent)),
                      'frecuencia_ganador': {m: v / B_BOOT for m, v in frec.items()}},
        'division_mitades': {'R': R_SPLIT, 'frecuencia_elegido': frec_split,
                             'Dice_elegido_en_A': float(sp[:, 0].mean()), 'Dice_elegido_en_B': float(sp[:, 1].mean()),
                             'optimismo_nivel': float((sp[:, 0] - sp[:, 1]).mean()),
                             'ventaja_vs_fcn_en_A': float(sp[:, 2].mean()), 'ventaja_vs_fcn_en_B': float(sp[:, 3].mean()),
                             'optimismo_ventaja_vs_fcn': float((sp[:, 2] - sp[:, 3]).mean())},
        'checkpoint_val_vs_test': {'por_red': val_test,
                                   'media_test_menos_val': float(np.mean([r['test_menos_val'] for r in val_test])),
                                   'redes_con_test_menor': int(sum(r['test_menos_val'] < 0 for r in val_test))},
        'contraste_transunet_fcn': {'validacion': tu_fcn_val, 'test': float(tu_fcn_test),
                                    'retencion_test_sobre_val': float(tu_fcn_test / tu_fcn_val)},
        'seleccion_de_semilla': 'No hay: el estimando promedia las cinco semillas; no se elige la mejor corrida.'}

    res = {'alcance': 'Exploratorio, posterior al test. Test: TOST desde resúmenes (media, d_z, n). Validación: '
                      'conjunto de selección de checkpoint, niveles sesgados al alza.',
           'delta': DELTA, 'alpha_por_lado': ALPHA, 'n_grupos_test': n_test, 'n_grupos_val': ng,
           'tost_test_vs_fcn': tost_test, 'concordancia_test_Holm9': concord_test,
           'tost_validacion': tost_val, 'sensibilidad_validacion_vs_fcn': sens,
           'dependencia_lag1_validacion': dep, 'influencia_validacion': infl,
           'simultaneo_maxT_validacion': {'B_signos': B_FLIP, 'B_boot': B_BOOT, 'c95_max_t': c95, 'contrastes': simult},
           'sesgo_ganador_validacion': ganador_val}
    (OUT / 'inferencia.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')

    f = lambda x: f'{x:.4f}'.replace('.', ',')
    fp = lambda x: (f'{x:.3g}' if x >= 1e-3 else f'{x:.1e}').replace('.', ',')
    L = ['# Tablas de inferencia complementaria', '', 'Generado por `scripts/inferencia_cierre_e2.py`. Exploratorio, '
         'posterior al test. δ = 0,02 Dice; TOST con α = 0,05 por lado (IC90 dentro de ±δ).', '',
         '## TOST en test, red − FCN (t pareado desde media, d_z y n = 122)', '',
         '| Red | Diferencia | DE | IC90 t | p TOST | p TOST Holm(9) | Equivalente (Holm) |', '|---|---:|---:|---:|---:|---:|---|']
    for r in tost_test:
        L.append(f"| {NAMES[r['red']]} | {f(r['media'])} | {f(r['DE'])} | [{f(r['IC90_t'][0])}; {f(r['IC90_t'][1])}] | "
                 f"{fp(r['p_TOST'])} | {fp(r['p_TOST_Holm9'])} | {'Sí' if r['equivalente_Holm9'] else 'No'} |")
    for clave, titulo in [('vs_fcn', 'red − FCN'), ('vs_transunet', 'red − TransUNet')]:
        L += ['', f'## TOST en validación, {titulo}', '',
              '| Red | Diferencia | IC90 t | IC90 bootstrap | p TOST t Holm(9) | p TOST boot Holm(9) | Equivalente t / boot |',
              '|---|---:|---:|---:|---:|---:|---|']
        for r in tost_val[clave]:
            L.append(f"| {NAMES[r['red']]} | {f(r['media'])} | [{f(r['IC90_t'][0])}; {f(r['IC90_t'][1])}] | "
                     f"[{f(r['IC90_bootstrap'][0])}; {f(r['IC90_bootstrap'][1])}] | {fp(r['p_TOST_Holm9'])} | "
                     f"{fp(r['p_TOST_bootstrap_Holm9'])} | {'Sí' if r['equivalente_t_Holm9'] else 'No'} / "
                     f"{'Sí' if r['equivalente_bootstrap_Holm9'] else 'No'} |")
    L += ['', '## Sensibilidad en validación, red − FCN (p Holm(9) por procedimiento)', '',
          '| Red | Media | Mediana | HL | t | Intercambio | Wilcoxon | Signo | Concuerdan | Simetría p | Asimetría |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in sens:
        L.append(f"| {NAMES[r['red']]} | {f(r['media'])} | {f(r['mediana'])} | {f(r['hodges_lehmann'])} | "
                 f"{fp(r['p_t_Holm9'])} | {fp(r['p_intercambio_Holm9'])} | {fp(r['p_wilcoxon_Holm9'])} | "
                 f"{fp(r['p_signo_Holm9'])} | {'Sí' if r['concuerdan'] else 'No'} | "
                 f"{fp(r['simetria']['p_bootstrap_simetrizado'])} | {r['simetria']['asimetria']:.2f}".replace('.', ',') + ' |')
    L += ['', '## Multiplicidad simultánea en validación (max-T por cambio de signo; IC95 simultáneos bootstrap)', '',
          f'Cuantil 95 % del máximo |t| bootstrap: {c95:.3f}.'.replace('.', ',', 1), '',
          '| Red | t | p max-T | IC95 simultáneo |', '|---|---:|---:|---:|']
    for r in simult:
        L.append(f"| {NAMES[r['red']]} | {r['t']:.2f} | {fp(r['p_maxT_paso_unico'])} | "
                 f"[{f(r['IC95_simultaneo'][0])}; {f(r['IC95_simultaneo'][1])}] |".replace('.', ',', 1))
    g = ganador_val
    L += ['', '## Sesgo del ganador en validación', '',
          f"Ganador: {NAMES[g['ganador_validacion']]}, Dice {f(g['Dice_ganador'])}. Bootstrap: sesgo del nivel "
          f"{f(g['bootstrap']['sesgo_nivel_ganador'])}, nivel corregido {f(g['bootstrap']['nivel_corregido'])}; "
          f"frecuencia como ganador {f(g['bootstrap']['frecuencia_ganador'][g['ganador_validacion']])}.",
          f"Mitades (elegir en A, medir en B, R = {R_SPLIT}): optimismo del nivel "
          f"{f(g['division_mitades']['optimismo_nivel'])}; de la ventaja sobre FCN "
          f"{f(g['division_mitades']['optimismo_ventaja_vs_fcn'])}.",
          f"Checkpoint por validación frente a test: test − val medio {f(g['checkpoint_val_vs_test']['media_test_menos_val'])}"
          f" ({g['checkpoint_val_vs_test']['redes_con_test_menor']} de 10 redes con test menor). TransUNet − FCN: "
          f"{f(g['contraste_transunet_fcn']['validacion'])} en validación y {f(g['contraste_transunet_fcn']['test'])} en test."]
    (OUT / 'TABLAS.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('Listo:', OUT / 'inferencia.json')


if __name__ == '__main__':
    main()
