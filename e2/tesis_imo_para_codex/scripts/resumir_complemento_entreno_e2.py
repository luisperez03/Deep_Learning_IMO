"""Resume el complemento de entrenamiento: sobreajuste, ablación, permutación y HPO.

Lee solo JSON de la carpeta de salida (y, si se da, los result.json y mejores checkpoints de
validación del benchmark original). No toca el test. Escribe RESUMEN.json y RESUMEN.md.
Todo es exploratorio: el test del benchmark ya fue observado.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT/'config/complemento_entreno_e2.json'
SEMILLA_REMUESTREO = 20261123


def leer(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resultados(salida, patron):
    out = []
    for p in sorted(Path(salida).glob(patron)):
        r = leer(p)
        if r.get('completed'):
            r['_dir'] = p.parent
            out.append(r)
    return out


def holm(pvalores):
    orden = sorted(range(len(pvalores)), key=lambda i: pvalores[i])
    ajust = [None]*len(pvalores); previo = 0.
    for rango, i in enumerate(orden):
        previo = max(previo, min(1., (len(pvalores)-rango)*pvalores[i]))
        ajust[i] = previo
    return ajust


def por_grupo(detalle):
    g = defaultdict(list)
    for d in detalle:
        g[d['group_id']].append(d['Dice'])
    return {k: float(np.mean(v)) for k, v in g.items()}


def validacion_original(base_runs, semilla):
    """Detalle por imagen de validación del mejor checkpoint original (solo si existe)."""
    if not base_runs:
        return None
    import torch
    carpeta = Path(base_runs)/f'transunet_2d_seed{semilla}'
    res = carpeta/'result.json'
    if not res.exists():
        return None
    r = leer(res)
    try:
        ckpt = torch.load(carpeta/r['best_file'], map_location='cpu', weights_only=True)
    except Exception as error:  # noqa: BLE001 - se reporta y se sigue sin el bootstrap por grupos
        print(f'No se pudo leer la validación original de la semilla {semilla}: {error}')
        return None
    return {'Dice': r['best_val_group_Dice'], 'validation': ckpt['validation']}


def inferencia_pareada(dif_semilla, dif_grupo, rng):
    from scipy import stats
    d = np.asarray(dif_semilla, dtype=float); n = len(d)
    out = {'n_semillas': n, 'diferencia_media': float(d.mean()) if n else None}
    if n >= 2:
        sd = float(d.std(ddof=1)); se = sd/math.sqrt(n); tcrit = stats.t.ppf(.975, n-1)
        out.update({'de': sd, 'ic95_t': [float(d.mean()-tcrit*se), float(d.mean()+tcrit*se)],
                    'p_t_pareada': float(stats.ttest_1samp(d, 0.).pvalue) if sd > 0 else None})
        if n >= 5 and np.any(d != 0):
            out['p_wilcoxon_exacto'] = float(stats.wilcoxon(d, method='exact').pvalue)
            out['nota_wilcoxon'] = f'Con {n} pares el p mínimo bilateral es {2/2**n:.4f}'
    if dif_grupo is not None:
        g = np.asarray(dif_grupo, dtype=float)
        boots = np.array([g[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)])
        signos = rng.choice([-1., 1.], size=(10000, len(g)))
        nulas = np.abs((signos*g).mean(axis=1))
        out.update({'n_grupos': len(g), 'diferencia_media_grupos': float(g.mean()),
                    'ic95_bootstrap_grupos': [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                    'p_signos_grupos': float((1+np.sum(nulas >= abs(g.mean())))/(10000+1))})
    return out


def resumir_ablacion(salida, cfg, base_runs, rng):
    semillas = cfg['comun']['semillas']
    filas = {(r['variante'], r['semilla']): r for r in resultados(salida, 'abl_*/result.json')}
    referencia_fuente = 'control completo en esta misma carpeta' if all(
        ('completo', s) in filas for s in semillas) else 'benchmark original'
    ref_dice, ref_grupo = {}, {}
    for s in semillas:
        if referencia_fuente.startswith('control'):
            r = filas[('completo', s)]
            ref_dice[s] = r['best_val_group_Dice']
            ref_grupo[s] = por_grupo(leer(r['_dir']/'best_val_predictions.json')['validation'])
        else:
            orig = validacion_original(base_runs, s)
            ref_dice[s] = orig['Dice'] if orig else cfg['referencia_original']['dice_validacion_por_semilla']['transunet_2d'][str(s)]
            if orig:
                ref_grupo[s] = por_grupo(orig['validation'])
    variantes = {}
    for var in cfg['ablacion']['variantes']:
        pares = [(s, filas[(var, s)]) for s in semillas if (var, s) in filas]
        if not pares:
            variantes[var] = {'corridas': 0}
            continue
        dif = [r['best_val_group_Dice']-ref_dice[s] for s, r in pares]
        dif_grupo = None
        if all(s in ref_grupo for s, _ in pares):
            var_g = [por_grupo(leer(r['_dir']/'best_val_predictions.json')['validation']) for _, r in pares]
            grupos = sorted(set.intersection(*[set(v) for v in var_g], *[set(ref_grupo[s]) for s, _ in pares]))
            dif_grupo = [np.mean([vg[g] for vg in var_g]) - np.mean([ref_grupo[s][g] for s, _ in pares])
                         for g in grupos]
        variantes[var] = {'corridas': len(pares),
                          'dice_por_semilla': {s: r['best_val_group_Dice'] for s, r in pares},
                          'dice_medio': float(np.mean([r['best_val_group_Dice'] for _, r in pares])),
                          'epocas': {s: r['epochs_completed'] for s, r in pares},
                          'parametros': pares[0][1]['parameters'],
                          'inferencia': inferencia_pareada(dif, dif_grupo, rng)}
    completas = [v for v in variantes.values() if v.get('corridas') == len(semillas)]
    pv = [v['inferencia'].get('p_t_pareada') for v in completas]
    if completas and all(p is not None for p in pv):
        for v, p in zip(completas, holm(pv)):
            v['inferencia']['p_t_pareada_holm'] = p
    return {'referencia': referencia_fuente, 'referencia_dice_por_semilla': ref_dice, 'variantes': variantes,
            'lectura': 'Diferencia = variante - referencia en Dice de validación. Negativa: quitar el componente empeora.'}


def resumir_permutacion(salida, cfg, base_runs):
    out = {}
    for bloque, spec in cfg['permutacion']['bloques'].items():
        modelo = spec['modelo']
        filas = resultados(salida, f'perm_{modelo}_b*/result.json')
        if not filas:
            out[bloque] = {'B_alcanzado': 0}
            continue
        semilla = cfg['permutacion']['semilla_entrenamiento']
        s_obs, fuente = None, None
        if base_runs and (Path(base_runs)/f'{modelo}_seed{semilla}/result.json').exists():
            s_obs = leer(Path(base_runs)/f'{modelo}_seed{semilla}/result.json')['best_val_group_Dice']
            fuente = 'result.json original en Drive'
        else:
            s_obs = cfg['referencia_original']['dice_validacion_por_semilla'][modelo][str(semilla)]
            fuente = 'auditoria_e2_recibida (copia en la configuración)'
        s_b = np.array([r['best_val_group_Dice'] for r in filas])
        B = len(s_b)
        out[bloque] = {'modelo': modelo, 'B_disenado': spec['B'], 'B_alcanzado': B,
                       's_obs': s_obs, 'fuente_s_obs': fuente,
                       'p': float((1+np.sum(s_b >= s_obs))/(B+1)), 'p_minimo_posible': 1/(B+1),
                       's_b': {'min': float(s_b.min()), 'media': float(s_b.mean()),
                               'mediana': float(np.median(s_b)), 'max': float(s_b.max())},
                       'epocas_medias': float(np.mean([r['epochs_completed'] for r in filas])),
                       'cumple_minimo_guia_costoso': B >= 200}
    return out


def resumir_hpo(salida, cfg):
    filas = resultados(salida, 'hpo_*/result.json')
    modelos = cfg['sobreajuste']['modelos']
    por_t = defaultdict(dict)
    for r in filas:
        por_t[r['intento']['t']][r['modelo']] = r
    T_completo = 0
    while T_completo in por_t and all(m in por_t[T_completo] for m in modelos):
        T_completo += 1
    tabla = {m: [] for m in modelos}
    for t in range(T_completo):
        for m in modelos:
            r = por_t[t][m]
            tabla[m].append({'t': t, 'lr': r['optim']['lr'], 'weight_decay': r['optim'].get('weight_decay'),
                             'dice_val': r['best_val_group_Dice'], 'mejor_epoca': r['best_epoch'],
                             'epocas': r['epochs_completed']})
    ganadores = {}
    for m, filas_m in tabla.items():
        if filas_m:
            g = max(filas_m, key=lambda f: f['dice_val'])
            receta = filas_m[0]['dice_val']
            ganadores[m] = {**g, 'dice_receta_mismo_presupuesto': receta,
                            'mejora_sobre_receta': g['dice_val']-receta,
                            'rango_entre_intentos': max(f['dice_val'] for f in filas_m)-min(f['dice_val'] for f in filas_m)}
    return {'T_completo_para_las_10': T_completo,
            'tareas_completas': len(filas), 'todas_las_configuraciones': tabla, 'ganador_por_validacion': ganadores,
            'advertencia': 'Una semilla y 10 épocas por intento: el ganador lleva ruido de semilla. Exploratorio.'}


def resumir_sobreajuste(salida):
    out = {}
    for p in sorted((Path(salida)/'sobreajuste').glob('*.json')):
        r = leer(p)
        out[r['modelo']] = {k: r[k] for k in ('dice_lote_modo_eval', 'perdida_inicial', 'perdida_final',
                                              'pasa', 'umbral', 'pasos', 'segundos')}
    return out


def markdown(res):
    L = ['# Complemento de entrenamiento E2 — resumen', '',
         'Generado por `scripts/resumir_complemento_entreno_e2.py`. Exploratorio: el test del benchmark ya '
         'fue observado; aquí todo se elige y se mide en validación.', '']
    s = res['sobreajuste']
    L += ['## Sobreajuste a un lote de 4 imágenes', '']
    if s:
        L += ['| red | Dice lote | pérdida inicial | pérdida final | pasa |', '|---|---|---|---|---|']
        L += [f'| {m} | {v["dice_lote_modo_eval"]:.4f} | {v["perdida_inicial"]:.4f} | {v["perdida_final"]:.4f} | '
              f'{"sí" if v["pasa"] else "no"} |' for m, v in s.items()]
    else:
        L.append('Sin corridas.')
    a = res['ablacion']
    L += ['', f'## Ablación de TransUNet (referencia: {a["referencia"]})', '',
          '| variante | corridas | Dice medio | diferencia | IC95 t | p t | p Holm | p Wilcoxon | IC95 grupos |',
          '|---|---|---|---|---|---|---|---|---|']
    for var, v in a['variantes'].items():
        if not v.get('corridas'):
            L.append(f'| {var} | 0 | | | | | | | |'); continue
        i = v['inferencia']

        def f(x):
            return '' if x is None else f'{x:.4f}'

        def ic(x):
            return '' if not x else f'[{x[0]:.4f}; {x[1]:.4f}]'
        L.append(f'| {var} | {v["corridas"]} | {v["dice_medio"]:.4f} | {f(i.get("diferencia_media"))} | '
                 f'{ic(i.get("ic95_t"))} | {f(i.get("p_t_pareada"))} | {f(i.get("p_t_pareada_holm"))} | '
                 f'{f(i.get("p_wilcoxon_exacto"))} | {ic(i.get("ic95_bootstrap_grupos"))} |')
    L += ['', '## Permutación de máscaras en train', '']
    for b, v in res['permutacion'].items():
        if not v.get('B_alcanzado'):
            L.append(f'- {b}: sin réplicas completas.'); continue
        L.append(f'- {b} ({v["modelo"]}): B = {v["B_alcanzado"]} de {v["B_disenado"]}; s_obs = {v["s_obs"]:.4f}; '
                 f's_b media {v["s_b"]["media"]:.4f} (máx. {v["s_b"]["max"]:.4f}); p = {v["p"]:.4f} '
                 f'(mínimo posible {v["p_minimo_posible"]:.4f}).')
    h = res['hpo']
    L += ['', f'## Búsqueda de hiperparámetros (T completo para las 10: {h["T_completo_para_las_10"]})', '']
    if h['ganador_por_validacion']:
        L += ['| red | t ganador | lr | weight decay | Dice val | receta mismo presupuesto | rango |',
              '|---|---|---|---|---|---|---|']
        L += [f'| {m} | {g["t"]} | {g["lr"]:.2e} | {g["weight_decay"]} | {g["dice_val"]:.4f} | '
              f'{g["dice_receta_mismo_presupuesto"]:.4f} | {g["rango_entre_intentos"]:.4f} |'
              for m, g in h['ganador_por_validacion'].items()]
    L += ['', h['advertencia'], '']
    return '\n'.join(L)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--base-runs', help='Carpeta del benchmark original (result.json y mejores checkpoints)')
    p.add_argument('--config', default=str(CONFIG))
    args = p.parse_args(argv)
    cfg = leer(args.config)
    rng = np.random.default_rng(SEMILLA_REMUESTREO)
    res = {'exploratorio': True, 'test_evaluado': False,
           'sobreajuste': resumir_sobreajuste(args.output),
           'ablacion': resumir_ablacion(args.output, cfg, args.base_runs, rng),
           'permutacion': resumir_permutacion(args.output, cfg, args.base_runs),
           'hpo': resumir_hpo(args.output, cfg)}
    salida = Path(args.output)
    (salida/'RESUMEN.json').write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    (salida/'RESUMEN.md').write_text(markdown(res), encoding='utf-8')
    print(markdown(res))


if __name__ == '__main__':
    main()
