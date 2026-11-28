"""Análisis de potencia del E3 (protocolo, §Potencia). Solo planificación: no usa ningún resultado del modelo original.

Qué calcula, por cada rival j de la familia (m = 10):
- La media de Dice por semilla: promedio de imágenes dentro de cada grupo y luego promedio entre los grupos (la
  misma regla de selección del E2).
- σ̂_d: desviación estándar entre semillas de la diferencia pareada base − rival, donde la base es la U-Net
  compacta del E2, que es el esqueleto del modelo original. Es el valor de planificación de la variabilidad del
  contraste original − rival; el del original real solo se conoce después de correrlo.
- R necesario para detectar δ = 0,02 con potencia 0,90 y Holm (peor caso, α/m bilateral):
  R = ⌈(z_{1−α/(2m)} + z_{0,90})² · σ̂_d² / δ²⌉.
- La diferencia mínima detectable con R = 10: MDD = (z_{1−α/(2m)} + z_{0,90}) · σ̂_d / √R.
- La potencia conjunta: el producto de las potencias por contraste, suponiéndolas independientes.

Aproximación normal: con 5 semillas, σ̂_d es imprecisa. Por eso también se reporta el R con la cota superior del
IC del 80 % de σ̂_d (chi cuadrado con 4 gl). Se hace con validación (planificación) y se muestra test solo como
contraste, porque el test ya se observó en el E2.

Uso: python potencia_e3.py --e2 <carpeta tesis_imo_para_codex> --salida potencia_e3.json
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats

ALFA, DELTA, POTENCIA, M, R_PLAN = 0.05, 0.02, 0.90, 10, 10
BASE = 'unet'


def media_por_semilla(d):
    """Media por semilla: imágenes dentro del grupo, luego grupos."""
    return d.groupby(['model', 'seed', 'group_id']).Dice.mean().groupby(['model', 'seed']).mean().unstack('model')


def potencia_normal(sigma, r, z_a):
    if sigma <= 0:
        return 1.0
    return float(stats.norm.cdf(DELTA * math.sqrt(r) / sigma - z_a))


def analizar(d):
    tabla = media_por_semilla(d)
    z_a, z_b = stats.norm.ppf(1 - ALFA / (2 * M)), stats.norm.ppf(POTENCIA)
    k = (z_a + z_b) ** 2
    filas = []
    for rival in tabla.columns:
        dif = (tabla[BASE] - tabla[rival]) if rival != BASE else None
        if dif is None:
            # Contraste original − U-Net compacta: la variabilidad base se aproxima con la DE de la U-Net sola × √2.
            sigma = float(tabla[BASE].std(ddof=1)) * math.sqrt(2)
            fuente = 'DE de la U-Net compacta × √2 (sin pareja)'
        else:
            sigma = float(dif.std(ddof=1))
            fuente = 'DE de la diferencia pareada U-Net compacta − rival'
        n = int(tabla[BASE].notna().sum())
        sigma_sup = sigma * math.sqrt((n - 1) / stats.chi2.ppf(0.10, n - 1))  # cota superior del IC 80 %
        filas.append({
            'rival': rival, 'media_rival': float(tabla[rival].mean()), 'media_base': float(tabla[BASE].mean()),
            'semillas': n, 'sigma_d': sigma, 'fuente_sigma': fuente, 'sigma_d_IC80_sup': sigma_sup,
            'R_necesario': math.ceil(k * sigma ** 2 / DELTA ** 2),
            'R_necesario_IC80_sup': math.ceil(k * sigma_sup ** 2 / DELTA ** 2),
            'MDD_R10': (z_a + z_b) * sigma / math.sqrt(R_PLAN),
            'potencia_R10': potencia_normal(sigma, R_PLAN, z_a),
        })
    conjunta = math.prod(f['potencia_R10'] for f in filas)
    return {'z_alfa_Holm_peor_caso': z_a, 'z_potencia': z_b, 'filas': filas, 'potencia_conjunta_R10': conjunta}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--e2', required=True)
    p.add_argument('--salida', required=True)
    a = p.parse_args()
    fuente = Path(a.e2) / 'resultados/errores_atajos_e2/por_imagen_corridas.csv'
    d = pd.read_csv(fuente)
    out = {'fuente': str(fuente.relative_to(Path(a.e2))), 'alfa': ALFA, 'delta': DELTA, 'potencia_objetivo': POTENCIA,
           'm': M, 'R_plan': R_PLAN, 'base': BASE,
           'validacion': analizar(d[d.split == 'val']), 'test_solo_contraste': analizar(d[d.split == 'test'])}
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    for parte in ['validacion', 'test_solo_contraste']:
        print(f'\n{parte}: potencia conjunta con R = 10: {out[parte]["potencia_conjunta_R10"]:.3f}')
        print(f'{"rival":15s} {"media":>6s} {"σ̂_d":>7s} {"R":>4s} {"R(IC80)":>8s} {"MDD10":>7s} {"pot10":>6s}')
        for f in out[parte]['filas']:
            print(f'{f["rival"]:15s} {f["media_rival"]:6.4f} {f["sigma_d"]:7.4f} {f["R_necesario"]:4d} '
                  f'{f["R_necesario_IC80_sup"]:8d} {f["MDD_R10"]:7.4f} {f["potencia_R10"]:6.3f}')


if __name__ == '__main__':
    main()
