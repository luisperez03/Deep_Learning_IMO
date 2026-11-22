"""Prueba del análisis de la fase 1 y de la regla de selección con datos SINTÉTICOS de efectos conocidos.

No entrena ni lee resultados reales. Genera las 1000 arquitecturas con
  y = 0,80 + 0,010·log2(w1/16) − 0,0015·log2(w1/16)² + 0,004·log2(w2/16) − 0,012·b1 − 0,006·b1·b2 + ruido(0,005)
y comprueba que el resumidor recupera lo que se puso: b1 y w1 son los factores de mayor varianza, b1 negativo, interacción b1:b2
negativa, w3 y b3 sin efecto (p de Holm alto), término cuadrático negativo en w1 y Wilcoxon a favor de no usar
dropout. Después arma una fase 2 sintética y comprueba la regla de empate por parámetros y el TOST.

Uso: python scripts/prueba_resumen_busqueda_sintetica.py
"""
import json
from pathlib import Path

import numpy as np

import busqueda_arquitectura_e2 as B
import resumir_busqueda_arquitectura_e2 as R

OUT = B.PROJECT/'resultados/busqueda_arquitectura_e2/prueba_resumen_sintetica.json'


def main():
    cfg = json.loads(B.CONFIG.read_text(encoding='utf-8'))
    rng = np.random.default_rng(1)
    filas = []
    for a in B.espacio(cfg):
        l1, l2 = np.log2(a['w'][0]/16), np.log2(a['w'][1]/16); b1, b2, _ = a['b']
        y = .80 + .010*l1 - .0015*l1**2 + .004*l2 - .012*b1 - .006*b1*b2 + rng.normal(0, .005)
        filas.append({**a, 'best_val_group_Dice': y, 'parameters': sum(a['w'])*100, 'seconds_total': sum(a['w']),
                      'history': [{'val_loss': .3, 'train_loss': .2}]})
    ml = R.modelo_lineal(filas)
    fac = {f['factor']: f for f in ml['factores']}
    coef = {c['termino']: c for c in ml['coeficientes']}
    tend = {t['termino']: t for t in R.tendencia(filas)}
    wil = R.wilcoxon_dropout(filas)
    checks = {
        'b1 y w1 son los dos factores de mayor varianza, como en la fórmula':
            sorted(fac, key=lambda k: -fac[k]['proporcion_varianza'])[:2] == ['b1', 'w1'],
        'b3 sin efecto (p de Holm > 0,05)': fac['b3']['p_holm'] > .05,
        'b1 negativo con IC95 bajo cero': coef['b1']['ic95'][1] < 0,
        'b1:b2 negativo con IC95 bajo cero': coef['b1:b2']['ic95'][1] < 0,
        'w3 sin efecto (p de Holm > 0,05)': fac['w3']['p_holm'] > .05,
        'cuadrático de w1 negativo (rendimientos decrecientes)': tend['log2_w1^2']['estimado'] < 0 and tend['log2_w1^2']['p'] < .05,
        'Wilcoxon: sin dropout rinde más (mediana > 0, p < 0,05)': wil['mediana_sin_menos_con'] > 0 and wil['p'] < .05,
        'R2 alto': ml['r2'] > .8,
    }
    # Fase 2 sintética: A mejor por poco (0,005) pero con más parámetros que B; C peor por 0,05.
    def corridas(arq, w, media, params):
        return [{'arq': arq, 'w': list(w), 'b': [0, 0, 0], 'semilla': s, 'best_val_group_Dice': media + e,
                 'parameters': params, 'macs': 1, 'seconds_total': 1., 'epochs_completed': 20, 'brecha_train_val': .01,
                 '_dir': Path('.')} for s, e in zip((11, 23, 37, 53, 71), (.002, -.001, .0, .001, -.002))]
    runs = corridas('a1', (256, 256, 256), .855, 900) + corridas('a2', (32, 32, 32), .850, 100) \
        + corridas('a3', (16, 32, 64), .800, 50)
    fin = {'referencia': 'a1', 'heredada': 'a3', '_top': ['a1', 'a2']}
    f2 = R.fase2_familia(runs, fin, cfg, {'a1': .8, 'a2': .79, 'a3': .7}, cfg['seleccion']['delta'])
    checks['regla 14.5: empate técnico gana la de menos parámetros'] = f2['ganador']['arq'] == 'a2'
    checks['TOST: a2 equivalente a la mejor, a3 no'] = f2['conjunto_equivalente_tost'] == ['a1', 'a2']
    checks['ganador frente a heredada: supera por δ o más'] = \
        f2['comparacion_heredada']['lectura'] == 'el ganador supera a la heredada por δ o más'
    out = {'que_es': 'Prueba con datos sintéticos; ningún número es un resultado', 'comprobaciones': checks,
           'todas': all(checks.values())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for k, v in checks.items():
        print('OK  ' if v else 'FALLA', k)
    if not out['todas']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
