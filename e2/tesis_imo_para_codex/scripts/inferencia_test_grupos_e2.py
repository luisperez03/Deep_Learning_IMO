"""Inferencia complementaria en TEST con los Dice por grupo (completa scripts/inferencia_cierre_e2.py).

  python scripts/inferencia_test_grupos_e2.py

Exploratorio, posterior al test. Lee ../resultados_test_e2_final/<red>_seed<s>.json (50 archivos, campo `groups`,
ignorados por git) y comprueba que reproduce los nueve contrastes de resultados/integracion_e2/analisis.json.
Estimando igual al del libro: Dice por grupo promediado sobre las cinco semillas; contraste = media de la diferencia
pareada por grupo; δ = 0,02; TOST con α = 0,05 por lado.

Hace en test lo que antes solo se pudo en validación: TOST con t y bootstrap (frente a FCN y frente a TransUNet),
simetría para Wilcoxon, cuatro procedimientos con Holm(9), max-T con IC95 simultáneos, sensibilidad sin un grupo o
sin una semilla, dependencia entre grupos vecinos y sesgo del ganador. Compara el TOST con el que salió de los
resúmenes. Salidas: resultados/inferencia_cierre_e2/inferencia_test.json y la sección de test de TABLAS.md (se
reemplaza si ya existe; lo de validación no se toca).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inferencia_cierre_e2 import (ALPHA, B_BOOT, B_FLIP, DELTA, NAMES, OUT, P, R_SPLIT, SEEDS, holm,  # noqa: E402
                                  hodges_lehmann, signflip, simetria, tost_t)

TEST = P.parent / 'resultados_test_e2_final'
MARCA = '<!-- TEST CON GRUPOS: generado por scripts/inferencia_test_grupos_e2.py -->'


def leer_test():
    scores, grupos = {}, None
    for f in sorted(TEST.glob('*_seed*.json')):
        j = json.loads(f.read_text(encoding='utf-8'))
        grupos = grupos or sorted(j['groups'])
        assert sorted(j['groups']) == grupos
        scores.setdefault(j['model'], {})[int(j['seed'])] = np.array([j['groups'][g]['Dice'] for g in grupos])
    assert len(scores) == 10 and all(sorted(v) == SEEDS for v in scores.values())
    return {m: np.stack([v[s] for s in SEEDS]) for m, v in scores.items()}, grupos


def main():
    rng = np.random.default_rng(20261123)
    ana = json.loads((P / 'resultados/integracion_e2/analisis.json').read_text(encoding='utf-8'))
    previo = {r['red']: r for r in json.loads((OUT / 'inferencia.json').read_text(encoding='utf-8'))['tost_test_vs_fcn']}
    scores, grupos = leer_test(); ng = len(grupos)
    gmean = {m: a.mean(0) for m, a in scores.items()}
    disc = max(abs((gmean[c['model']] - gmean['fcn']).mean() - c['difference']) for c in ana['contrasts'])
    assert disc < 1e-12 and ng == ana['friedman']['blocks']
    models = sorted(gmean, key=lambda m: -gmean[m].mean())
    ix = rng.integers(ng, size=(B_BOOT, ng))

    def contraste(a, b):
        d = gmean[a] - gmean[b]; boot = d[ix].mean(1); r = tost_t(d.mean(), d.std(ddof=1), ng)
        lo, hi = np.quantile(boot, [.05, .95])
        r.update({'red': a, 'referencia': b, 'd_z': float(d.mean() / d.std(ddof=1)),
                  'IC90_bootstrap': [float(lo), float(hi)], 'IC95_bootstrap': np.quantile(boot, [.025, .975]).tolist(),
                  'p_TOST_bootstrap': float(max(np.mean(boot <= -DELTA), np.mean(boot >= DELTA))),
                  'equivalente_bootstrap_IC90': bool(lo > -DELTA and hi < DELTA)})
        return r, d

    tost, difs = {'vs_fcn': [], 'vs_transunet': []}, {}
    for ref, clave in [('fcn', 'vs_fcn'), ('transunet_2d', 'vs_transunet')]:
        for m in models:
            if m != ref:
                r, d = contraste(m, ref); tost[clave].append(r)
                if ref == 'fcn':
                    difs[m] = d
        for k in ['p_TOST', 'p_TOST_bootstrap']:
            for r, a in zip(tost[clave], holm([r[k] for r in tost[clave]])):
                r[k + '_Holm9'] = float(a)
        for r in tost[clave]:
            r['equivalente_t_Holm9'] = r['p_TOST_Holm9'] < ALPHA
            r['equivalente_bootstrap_Holm9'] = r['p_TOST_bootstrap_Holm9'] < ALPHA
    comparacion = [{'red': r['red'], 'p_TOST_resumenes': previo[r['red']]['p_TOST'], 'p_TOST_grupos': r['p_TOST'],
                    'IC90_t_resumenes': previo[r['red']]['IC90_t'], 'IC90_t_grupos': r['IC90_t'],
                    'max_dif_abs': float(max(abs(previo[r['red']]['p_TOST'] - r['p_TOST']),
                                             *np.abs(np.subtract(previo[r['red']]['IC90_t'], r['IC90_t']))))}
                   for r in tost['vs_fcn']]

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
    # Comprobación con el libro: intercambio con la misma semilla y Wilcoxon deben coincidir.
    for r, c in zip(sorted(sens, key=lambda x: x['red']), sorted(ana['contrasts'], key=lambda x: x['model'])):
        assert abs(r['p_intercambio'] - c['p_signflip_MC']) < 1e-12 and abs(r['p_wilcoxon'] - c['p_wilcoxon_asymptotic']) < 1e-12

    def lag1(x):
        x = x - x.mean(); return float((x[:-1] * x[1:]).sum() / (x * x).sum())
    dep = []
    for m, d in difs.items():
        obs = lag1(d); nulo = np.array([lag1(rng.permutation(d)) for _ in range(4999)])
        dep.append({'red': m, 'autocorr_lag1': obs, 'p_perm': float((np.sum(abs(nulo) >= abs(obs)) + 1) / 5000)})
    for r, a in zip(dep, holm([r['p_perm'] for r in dep])):
        r['p_perm_Holm9'] = float(a)

    # Sensibilidad: sin un grupo y sin una semilla; decisión t sin ajuste, t con Holm(9) y TOST.
    infl = []
    p_t = np.array([stats.ttest_1samp(d, 0).pvalue for d in difs.values()]); base_holm = holm(p_t) < ALPHA
    for k, (m, d) in enumerate(difs.items()):
        base = p_t[k] < ALPHA; base_tost = tost_t(d.mean(), d.std(ddof=1), ng)['p_TOST'] < ALPHA
        loo = np.array([np.delete(d, i).mean() for i in range(ng)])
        loo_p = np.array([stats.ttest_1samp(np.delete(d, i), 0).pvalue for i in range(ng)])
        loo_tost = np.array([tost_t(np.delete(d, i).mean(), np.delete(d, i).std(ddof=1), ng - 1)['p_TOST'] for i in range(ng)])
        sem = []
        for s in range(5):
            ds = np.delete(scores[m], s, 0).mean(0) - np.delete(scores['fcn'], s, 0).mean(0)
            tt = tost_t(ds.mean(), ds.std(ddof=1), ng)
            sem.append({'sin_semilla': SEEDS[s], 'media': float(ds.mean()), 'p_t': float(stats.ttest_1samp(ds, 0).pvalue),
                        'p_TOST': tt['p_TOST'], 'IC90_t': tt['IC90_t']})
        infl.append({'red': m, 'decision_t_base': bool(base), 'decision_t_Holm_base': bool(base_holm[k]),
                     'equivalente_base': bool(base_tost), 'media_loo_rango': [float(loo.min()), float(loo.max())],
                     'p_TOST_loo_rango': [float(loo_tost.min()), float(loo_tost.max())],
                     'decision_t_cambia_sin_un_grupo': int(np.sum((loo_p < ALPHA) != base)),
                     'TOST_cambia_sin_un_grupo': int(np.sum((loo_tost < ALPHA) != base_tost)),
                     'sin_una_semilla': sem,
                     'decision_t_cambia_sin_una_semilla': int(sum((x['p_t'] < ALPHA) != base for x in sem)),
                     'TOST_cambia_sin_una_semilla': int(sum((x['p_TOST'] < ALPHA) != base_tost for x in sem))})

    D = np.stack([difs[m] for m in difs], 1)
    tobs = D.mean(0) / (D.std(0, ddof=1) / np.sqrt(ng)); maxt = np.empty(B_FLIP)
    for s in range(0, B_FLIP, 1000):
        k = min(1000, B_FLIP - s); Ds = (rng.integers(0, 2, size=(k, ng, 1)) * 2 - 1) * D
        maxt[s:s + k] = np.abs(Ds.mean(1) / (Ds.std(1, ddof=1) / np.sqrt(ng))).max(1)
    Db = D[ix]; tb = np.abs((Db.mean(1) - D.mean(0)) / (Db.std(1, ddof=1) / np.sqrt(ng))).max(1)
    c95 = float(np.quantile(tb, .95)); ee = D.std(0, ddof=1) / np.sqrt(ng)
    simult = [{'red': m, 't': float(tobs[k]), 'p_maxT_paso_unico': float((np.sum(maxt >= abs(tobs[k])) + 1) / (B_FLIP + 1)),
               'IC95_simultaneo': [float(D[:, k].mean() - c95 * ee[k]), float(D[:, k].mean() + c95 * ee[k])]}
              for k, m in enumerate(difs)]

    M = np.stack([gmean[m] for m in models], 1); est = M.mean(0)
    sn, frec = [], {m: 0 for m in models}
    for b in range(B_BOOT):
        mb = M[ix[b]].mean(0); w = int(mb.argmax()); frec[models[w]] += 1; sn.append(mb[w] - est[w])
    split = []
    for _ in range(R_SPLIT):
        perm = rng.permutation(ng); A, Bm = perm[:ng // 2], perm[ng // 2:]; w = int(M[A].mean(0).argmax())
        split.append((models[w], M[A, w].mean() - M[Bm, w].mean()))
    ganador = {'ganador_test': models[0], 'Dice': float(est.max()), 'frecuencia_ganador_bootstrap': {m: v / B_BOOT for m, v in frec.items()},
               'sesgo_nivel_bootstrap': float(np.mean(sn)), 'nivel_corregido': float(est.max() - np.mean(sn)),
               'optimismo_mitades': float(np.mean([s[1] for s in split])),
               'frecuencia_elegido_mitades': {m: sum(s[0] == m for s in split) / R_SPLIT for m in models}}

    res = {'alcance': 'TEST con Dice por grupo, exploratorio y posterior al test (ya observado en E1 y en el cierre).',
           'discrepancia_con_analisis_json': disc, 'n_grupos': ng, 'delta': DELTA, 'tost': tost,
           'comparacion_tost_resumenes': comparacion, 'sensibilidad_vs_fcn': sens, 'dependencia_lag1': dep,
           'influencia': infl, 'simultaneo_maxT': {'c95_max_t': c95, 'contrastes': simult}, 'sesgo_ganador': ganador}
    (OUT / 'inferencia_test.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')

    c = lambda s: s.replace('.', ',')
    f = lambda x: c(f'{x:.4f}')
    fp = lambda x: c(f'{x:.3g}' if x >= 1e-3 else f'{x:.1e}')
    si = lambda b: 'Sí' if b else 'No'
    L = [MARCA, '', '# Test con los Dice por grupo', '',
         f'Generado por `scripts/inferencia_test_grupos_e2.py`. 122 grupos; reproduce `analisis.json` (discrepancia '
         f'{disc:.1g}). Exploratorio, posterior al test.', '']
    for clave, titulo in [('vs_fcn', 'red − FCN'), ('vs_transunet', 'red − TransUNet')]:
        L += [f'## TOST en test, {titulo}', '',
              '| Red | Diferencia | IC90 t | IC90 bootstrap | p TOST t Holm(9) | p TOST boot Holm(9) | Equivalente t / boot |',
              '|---|---:|---:|---:|---:|---:|---|']
        for r in tost[clave]:
            L.append(f"| {NAMES[r['red']]} | {f(r['media'])} | [{f(r['IC90_t'][0])}; {f(r['IC90_t'][1])}] | "
                     f"[{f(r['IC90_bootstrap'][0])}; {f(r['IC90_bootstrap'][1])}] | {fp(r['p_TOST_Holm9'])} | "
                     f"{fp(r['p_TOST_bootstrap_Holm9'])} | {si(r['equivalente_t_Holm9'])} / {si(r['equivalente_bootstrap_Holm9'])} |")
        L.append('')
    L += [c(f"Diferencia máxima con el TOST desde resúmenes (p e IC90): {max(x['max_dif_abs'] for x in comparacion):.1e}") + '.', '',
          '## Sensibilidad en test, red − FCN (p Holm(9) por procedimiento)', '',
          '| Red | Media | Mediana | HL | t | Intercambio | Wilcoxon | Signo | Concuerdan | Simetría p | Asimetría |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in sens:
        L.append(f"| {NAMES[r['red']]} | {f(r['media'])} | {f(r['mediana'])} | {f(r['hodges_lehmann'])} | {fp(r['p_t_Holm9'])} | "
                 f"{fp(r['p_intercambio_Holm9'])} | {fp(r['p_wilcoxon_Holm9'])} | {fp(r['p_signo_Holm9'])} | "
                 f"{si(r['concuerdan'])} | {fp(r['simetria']['p_bootstrap_simetrizado'])} | {c(f'{r['simetria']['asimetria']:.2f}')} |")
    L += ['', '## Sin un grupo o sin una semilla (test, red − FCN)', '',
          '| Red | Media sin un grupo | p TOST sin un grupo | Cambios t / TOST sin un grupo | Media sin una semilla | Cambios t / TOST sin una semilla |',
          '|---|---:|---:|---:|---:|---:|']
    for r in infl:
        ms = [x['media'] for x in r['sin_una_semilla']]
        L.append(f"| {NAMES[r['red']]} | [{f(r['media_loo_rango'][0])}; {f(r['media_loo_rango'][1])}] | "
                 f"[{fp(r['p_TOST_loo_rango'][0])}; {fp(r['p_TOST_loo_rango'][1])}] | {r['decision_t_cambia_sin_un_grupo']} / "
                 f"{r['TOST_cambia_sin_un_grupo']} | [{f(min(ms))}; {f(max(ms))}] | {r['decision_t_cambia_sin_una_semilla']} / "
                 f"{r['TOST_cambia_sin_una_semilla']} |")
    L += ['', '## Multiplicidad simultánea en test (max-T por cambio de signo; IC95 simultáneos bootstrap)', '',
          c(f'Cuantil 95 % del máximo |t| bootstrap: {c95:.3f}') + '.', '', '| Red | t | p max-T | IC95 simultáneo |', '|---|---:|---:|---:|']
    for r in simult:
        L.append(f"| {NAMES[r['red']]} | {c(f'{r['t']:.2f}')} | {fp(r['p_maxT_paso_unico'])} | "
                 f"[{f(r['IC95_simultaneo'][0])}; {f(r['IC95_simultaneo'][1])}] |")
    L += ['', '## Sesgo del ganador en test', '',
          f"Ganador: {NAMES[ganador['ganador_test']]}, Dice {f(ganador['Dice'])}; gana en el "
          f"{c(f'{100 * ganador['frecuencia_ganador_bootstrap'][ganador['ganador_test']]:.1f}')} % de los bootstrap; sesgo del nivel "
          f"{f(ganador['sesgo_nivel_bootstrap'])}, nivel corregido {f(ganador['nivel_corregido'])}; optimismo por mitades "
          f"{f(ganador['optimismo_mitades'])}."]
    tablas = OUT / 'TABLAS.md'
    previo_txt = tablas.read_text(encoding='utf-8').split(MARCA)[0].rstrip() + '\n\n'
    tablas.write_text(previo_txt + '\n'.join(L) + '\n', encoding='utf-8')
    print('Listo:', OUT / 'inferencia_test.json')


if __name__ == '__main__':
    main()
