"""Tablas y figura del complemento de entrenamiento para el libro (filas 37, 56 y 61 del checklist).

  python scripts/tablas_complemento_entreno_e2.py

Lee resultados/complemento_entreno_e2/RESUMEN.json (de scripts/resumir_complemento_entreno_e2.py) y corridas.csv (de
scripts/tabla_corridas_complemento_e2.py). Escribe en la misma carpeta TABLA_ABLACION.md, TABLA_HPO.md,
TABLA_PERMUTACION.md y permutacion_fcn.png. No recalcula ninguna prueba: solo da formato a lo que el resumen ya trae.
Todo se mide en validación; el test no se carga.
"""
import csv
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / 'resultados' / 'complemento_entreno_e2'
NOMBRES = {'fcn': 'FCN compacta', 'segnet': 'SegNet compacta', 'unet': 'U-Net compacta',
           'deeplabv3plus': 'DeepLabv3+ R18', 'unetplusplus': 'U-Net++ R18', 'pidnet_s': 'PIDNet-S',
           'segnext_t': 'SegNeXt-T', 'segformer_b0': 'SegFormer-B0', 'transunet_2d': 'TransUNet',
           'vmunet': 'VM-UNet'}
VARIANTES = {'sin_transformer': 'Sin Transformer', 'sin_saltos': 'Sin conexiones de salto', 'sin_dice': 'BCE sola (sin Dice)'}


def n(x, d=3):
    return f'{float(x):.{d}f}'.replace('.', ',').replace('-', '−')


def ic(par, d=3):
    return f'[{n(par[0], d)}; {n(par[1], d)}]'


def p(x):
    return '< 0,001' if x < 0.001 else n(x, 3)


def escribir(nombre, titulo, nota, filas):
    (CARPETA / nombre).write_text('\n'.join([f'# {titulo}', '', nota, ''] + filas) + '\n', encoding='utf-8')


def ablacion(res):
    a = res['ablacion']
    ref = a['referencia_dice_por_semilla']
    media_ref = sum(ref.values()) / len(ref)
    filas = ['| Variante | Parámetros | Dice val medio | Diferencia | IC95 t (semillas) | p t, Holm | p Wilcoxon '
             '| IC95 bootstrap (grupos) | Épocas |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
             f'| TransUNet completa (benchmark) | {res["_parametros_completa"] / 1e6:.1f} M'.replace('.', ',')
             + f' | {n(media_ref)} | — | — | — | — | — | 30 máx. |']
    for k, v in a['variantes'].items():
        i = v['inferencia']
        ep = sorted(v['epocas'].values())
        filas.append(f'| {VARIANTES[k]} | {v["parametros"] / 1e6:.1f} M'.replace('.', ',')
                     + f' | {n(v["dice_medio"])} | {n(i["diferencia_media"])} | {ic(i["ic95_t"])} | '
                     f'{p(i["p_t_pareada_holm"])} | {n(i["p_wilcoxon_exacto"], 4)} | '
                     f'{ic(i["ic95_bootstrap_grupos"])} | {ep[0]}–{ep[-1]} |')
    escribir('TABLA_ABLACION.md', 'Ablación de TransUNet: Dice de validación por grupo',
             'Generado por `scripts/tablas_complemento_entreno_e2.py` desde `RESUMEN.json`. Cinco semillas pareadas '
             '(11, 23, 37, 53, 71). Diferencia = variante − completa. Con 5 pares, el p mínimo del Wilcoxon bilateral '
             'es 0,0625. El bootstrap remuestrea los 122 grupos de validación.', filas)


def hpo(res):
    h = res['hpo']
    filas = ['| Red | Receta (t0) | t1: lr × 0,1 | t2: lr × 9,3 | t3: lr × 1,19, sin wd | Mejor − receta |',
             '|---|---:|---:|---:|---:|---:|']
    orden = sorted(h['todas_las_configuraciones'], key=lambda m: -max(c['dice_val'] for c in h['todas_las_configuraciones'][m]))
    for m in orden:
        cs = h['todas_las_configuraciones'][m]
        mejor = max(c['dice_val'] for c in cs)
        celdas = [f'**{n(c["dice_val"])}**' if c['dice_val'] == mejor else n(c['dice_val']) for c in cs]
        filas.append(f'| {NOMBRES[m]} | ' + ' | '.join(celdas) + f' | {n(h["ganador_por_validacion"][m]["mejora_sobre_receta"])} |')
    escribir('TABLA_HPO.md', 'Búsqueda de hiperparámetros: Dice de validación por intento',
             'Generado por `scripts/tablas_complemento_entreno_e2.py` desde `RESUMEN.json`. Una semilla (11), máximo 10 '
             'épocas por intento, mismo espacio relativo para las 10 redes: cada intento multiplica la tasa de '
             'aprendizaje de la receta y fija el weight decay (1e-4 en t1 y t2, 0 en t3). En negrita, el mejor intento '
             'de cada red. Ordenado por el mejor Dice.', filas)


def permutacion(res, corridas):
    v = res['permutacion']['pipeline_fcn']
    s_b = [float(r['dice_val']) for r in corridas if r['corrida'].startswith('perm_fcn_')]
    ep = [int(r['epocas']) for r in corridas if r['corrida'].startswith('perm_fcn_')]
    filas = ['| Pipeline | B | Dice val real | Nula: mín. | media | mediana | máx. | p | Épocas por réplica |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
             f'| FCN compacta | {len(s_b)} | {n(v["s_obs"])} | {n(min(s_b))} | {n(v["s_b"]["media"])} | '
             f'{n(v["s_b"]["mediana"])} | {n(max(s_b))} | {n(v["p"], 4)} | {min(ep)}–{max(ep)} (media {n(v["epocas_medias"], 1)}) |']
    escribir('TABLA_PERMUTACION.md', 'Permutación de máscaras en train',
             'Generado por `scripts/tablas_complemento_entreno_e2.py` desde `RESUMEN.json` y `corridas.csv`. p = (1 + '
             '#{nula ≥ real}) / (B + 1); con B = 200 el mínimo posible es 1/201 = 0,0050.', filas)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(s_b, bins=25, color='#8c8c8c', edgecolor='white')
    ax.axvline(v['s_obs'], color='#c0392b', lw=2)
    ax.text(v['s_obs'] - 0.01, ax.get_ylim()[1] * 0.9, f'real {n(v["s_obs"])}', ha='right', color='#c0392b')
    ax.set_xlabel('Dice de validación por grupo')
    ax.set_ylabel('réplicas')
    ax.set_title(f'FCN con máscaras de train permutadas (B = {len(s_b)})')
    ax.set_xlim(0, 0.9)
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(CARPETA / 'permutacion_fcn.png', dpi=150)


def main():
    res = json.loads((CARPETA / 'RESUMEN.json').read_text(encoding='utf-8'))
    # La variante BCE sola solo cambia la pérdida: su conteo de parámetros es el de la red completa.
    res['_parametros_completa'] = res['ablacion']['variantes']['sin_dice']['parametros']
    corridas = list(csv.DictReader((CARPETA / 'corridas.csv').open(encoding='utf-8')))
    ablacion(res)
    hpo(res)
    permutacion(res, corridas)
    print('Tablas y figura en', CARPETA)


if __name__ == '__main__':
    main()
