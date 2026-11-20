"""Sensibilidad del Dice de test sin los grupos con casi duplicado en train (checklist 11 y 46). Exploratorio.

El test ya se observó: esto no cambia selección, umbral ni ranking oficial. Repite el resumen por grupos quitando los
grupos de test listados en resultados/fuga_embeddings_e2/grupos_casi_duplicado.json.

Fuentes, de más a menos completa (se usa la primera disponible, o la que se pida):
  --resultados-test DIR   los 50 JSON cerrados `{modelo}_seed{semilla}.json` (campo `groups` -> Dice por grupo).
                          Por defecto ../resultados_test_e2_final, que el snapshot no trae.
  --bordes-seed11 JSON    resultados/complementos_locales_e2/bordes_seed11_por_imagen.json: precisión y exhaustividad
                          por imagen de las 10 redes en seed11. Dice = 2PR/(P+R) = 2TP/(2TP+FP+FN), exacto cuando
                          TP>0; con TP=0 ambos son 0 y el Dice también. Es UNA semilla, no el promedio de cinco.
  --simular               genera 50 JSON sintéticos con el formato cerrado y comprueba el cálculo (prueba).

Unidad: grupo. Dice de un modelo = media sobre grupos de la media sobre semillas (igual que evaluar_e2_cerrado.py).
IC 95 % percentil con 5.000 remuestreos de grupos (semilla 20261119), semillas fijas.
Escribe resultados/fuga_embeddings_e2/sensibilidad_<fuente>.json.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
OUT = P / 'resultados/fuga_embeddings_e2'
SEEDS = [11, 23, 37, 53, 71]
MODELOS = ['transunet_2d', 'unet', 'fcn', 'vmunet', 'deeplabv3plus', 'unetplusplus', 'pidnet_s', 'segnext_t',
           'segformer_b0', 'segnet']
SEMILLA_BOOT = 20261119
B = 5000


def leer(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def desde_resultados_test(d):
    """modelo -> (semillas x grupos) y lista de grupos."""
    raw = {m: [leer(Path(d) / f'{m}_seed{s}.json') for s in SEEDS] for m in MODELOS}
    grupos = sorted(raw[MODELOS[0]][0]['groups'])
    return {m: np.array([[r['groups'][g]['Dice'] for g in grupos] for r in raw[m]]) for m in MODELOS}, grupos


def desde_bordes(path):
    filas = leer(path)
    dice = {}
    for f in filas:
        p, r = f['precision'], f['recall']
        dice.setdefault(f['model'], {}).setdefault(f['group_id'], []).append(2 * p * r / (p + r) if p + r else 0.0)
    grupos = sorted(dice[MODELOS[0]])
    return {m: np.array([[np.mean(dice[m][g]) for g in grupos]]) for m in MODELOS}, grupos


def kendall_tau(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return float(s / (n * (n - 1) / 2))


def analizar(scores, grupos, excluir):
    g = np.array(grupos)
    fuera = np.isin(g, list(excluir))
    if fuera.sum() != len(excluir):
        raise ValueError(f'Grupos a excluir que no están en los resultados: {sorted(set(excluir) - set(grupos))}')
    rng = np.random.default_rng(SEMILLA_BOOT)
    idx_resto = rng.integers(0, (~fuera).sum(), size=(B, (~fuera).sum()))
    idx_casi = rng.integers(0, fuera.sum(), size=(B, fuera.sum()))
    filas, medias = [], {}
    for m in MODELOS:
        ga = scores[m].mean(axis=0)  # media sobre semillas, por grupo
        resto, casi = ga[~fuera], ga[fuera]
        medias[m] = (ga, resto)
        dif_boot = casi[idx_casi].mean(1) - resto[idx_resto].mean(1)
        filas.append({'modelo': m, 'Dice_todos': float(ga.mean()), 'Dice_sin_casi_duplicados': float(resto.mean()),
                      'cambio': float(resto.mean() - ga.mean()), 'Dice_solo_grupos_casi_duplicado': float(casi.mean()),
                      'casi_menos_resto': float(casi.mean() - resto.mean()),
                      'casi_menos_resto_IC95': np.quantile(dif_boot, [.025, .975]).tolist()})
    orden_todos = sorted(MODELOS, key=lambda m: -medias[m][0].mean())
    orden_sin = sorted(MODELOS, key=lambda m: -medias[m][1].mean())
    tau = kendall_tau([medias[m][0].mean() for m in MODELOS], [medias[m][1].mean() for m in MODELOS])
    contraste = {}
    for nombre, sel in (('todos', np.ones(len(g), bool)), ('sin_casi_duplicados', ~fuera)):
        d = medias['transunet_2d'][0][sel] - medias['fcn'][0][sel]
        boot = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
        contraste[nombre] = {'diferencia': float(d.mean()), 'IC95': np.quantile(boot, [.025, .975]).tolist(), 'grupos': int(sel.sum())}
    return {'grupos_total': len(grupos), 'grupos_excluidos': sorted(excluir), 'grupos_restantes': int((~fuera).sum()),
            'por_modelo': filas, 'ranking_todos': orden_todos, 'ranking_sin_casi_duplicados': orden_sin,
            'ranking_cambia': orden_todos != orden_sin, 'kendall_tau_rankings': tau,
            'cambio_max_abs': max(abs(f['cambio']) for f in filas),
            'contraste_transunet_menos_fcn': contraste}


def simular(excluir):
    """50 JSON sintéticos con el formato cerrado; comprueba el cálculo contra una cuenta directa."""
    rng = np.random.default_rng(7)
    grupos = sorted(set(excluir) | {f'g_sim_{k:04d}' for k in range(122 - len(excluir))})
    with tempfile.TemporaryDirectory() as tmp:
        verdad = {}
        for i, m in enumerate(MODELOS):
            base = 0.70 + 0.015 * i
            mat = []
            for s in SEEDS:
                vals = {g: float(np.clip(base + (0.05 if g in excluir else 0) + rng.normal(0, .1), 0, 1)) for g in grupos}
                mat.append([vals[g] for g in grupos])
                (Path(tmp) / f'{m}_seed{s}.json').write_text(json.dumps(
                    {'groups': {g: {'Dice': v} for g, v in vals.items()}}), encoding='utf-8')
            verdad[m] = np.array(mat)
        scores, gs = desde_resultados_test(tmp)
        res = analizar(scores, gs, excluir)
    sel = ~np.isin(np.array(gs), list(excluir))
    for f in res['por_modelo']:
        esperado = verdad[f['modelo']].mean(0)[sel].mean()
        assert abs(f['Dice_sin_casi_duplicados'] - esperado) < 1e-12, f
        assert abs(f['Dice_todos'] - verdad[f['modelo']].mean()) < 1e-12, f
        assert f['casi_menos_resto'] > 0  # el efecto simulado (+0,05) se recupera con signo correcto
    assert res['grupos_restantes'] == 122 - len(excluir)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--resultados-test', type=Path, default=P.parent / 'resultados_test_e2_final')
    ap.add_argument('--bordes-seed11', type=Path, default=P / 'resultados/complementos_locales_e2/bordes_seed11_por_imagen.json')
    ap.add_argument('--fuente', choices=['auto', 'resultados_test', 'bordes_seed11', 'simular'], default='auto')
    ap.add_argument('--grupos', type=Path, default=OUT / 'grupos_casi_duplicado.json')
    a = ap.parse_args(argv)
    excluir = set(leer(a.grupos)['test'])
    fuente = a.fuente
    if fuente == 'auto':
        fuente = 'resultados_test' if (a.resultados_test / 'fcn_seed11.json').exists() else 'bordes_seed11'
    if fuente == 'simular':
        res = simular(excluir)
        nota = 'Datos sintéticos: solo prueba que el cálculo es correcto. No son resultados.'
    elif fuente == 'resultados_test':
        res = analizar(*desde_resultados_test(a.resultados_test), excluir)
        nota = 'Cinco semillas, resultados cerrados de test.'
    else:
        res = analizar(*desde_bordes(a.bordes_seed11), excluir)
        nota = ('Solo seed11: Dice por imagen reconstruido de precisión y exhaustividad guardadas (exacto). '
                'No es el Dice oficial de cinco semillas; sirve para ver la dirección y el tamaño del cambio.')
    res = {'fuente': fuente, 'nota': nota, 'estado': 'Exploratorio, posterior al test; no cambia selección ni conclusiones cerradas.',
           **res}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'sensibilidad_{fuente}.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    for f in res['por_modelo']:
        print(f"{f['modelo']:14s} todos {f['Dice_todos']:.4f}  sin {f['Dice_sin_casi_duplicados']:.4f}  "
              f"cambio {f['cambio']:+.4f}  casi-resto {f['casi_menos_resto']:+.4f} {np.round(f['casi_menos_resto_IC95'], 4)}")
    print('ranking cambia:', res['ranking_cambia'], 'tau', round(res['kendall_tau_rankings'], 3))
    print(json.dumps(res['contraste_transunet_menos_fcn'], indent=1))


if __name__ == '__main__':
    sys.exit(main())
