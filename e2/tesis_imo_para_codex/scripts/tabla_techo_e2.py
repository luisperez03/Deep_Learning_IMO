"""Tabla del techo humano y γ por red para el libro (fila 63 del checklist).

  python scripts/tabla_techo_e2.py

Lee resultados/techo_e2/techo_reanotacion.json (`techo_reanotacion_e2.py medir`) y gamma.json
(`techo_reanotacion_e2.py gamma --techo ...`). Agrega, desde resultados/validacion_por_imagen_e2/, el Dice de cada red
en las MISMAS 50 imágenes del techo (media de sus 5 semillas), para comparar sobre la misma muestra. Escribe resultados/techo_e2/TABLA.md. Solo da formato: no recalcula.
"""
import json
from pathlib import Path

CARPETA = Path(__file__).resolve().parents[1] / 'resultados' / 'techo_e2'
NOMBRES = {'fcn': 'FCN compacta', 'segnet': 'SegNet compacta', 'unet': 'U-Net compacta',
           'deeplabv3plus': 'DeepLabv3+ R18', 'unetplusplus': 'U-Net++ R18', 'pidnet_s': 'PIDNet-S',
           'segnext_t': 'SegNeXt-T', 'segformer_b0': 'SegFormer-B0', 'transunet_2d': 'TransUNet',
           'vmunet': 'VM-UNet'}


def n(x):
    return f'{x:.3f}'.replace('.', ',')


def dice_mismas(ids):
    out = {}
    for f in sorted((CARPETA.parent / 'validacion_por_imagen_e2').glob('*_val_por_imagen.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        por = {x['id']: x['Dice'] for x in d['detail']}
        out.setdefault(d['model'], []).append(sum(por[i] for i in ids) / len(ids))
    return {m: sum(v) / len(v) for m, v in out.items() if len(v) == 5}


def main():
    t = json.loads((CARPETA / 'techo_reanotacion.json').read_text(encoding='utf-8'))
    g = json.loads((CARPETA / 'gamma.json').read_text(encoding='utf-8'))
    ter = t['por_tercil_area']
    ids = [x['id'] for x in t['por_imagen']]
    mismas = dice_mismas(ids)
    lin = ['# Techo humano y γ', '',
           'Generado por `scripts/tabla_techo_e2.py` desde `techo_reanotacion.json` y `gamma.json`. Techo: Dice entre '
           'una segunda anotación independiente y la máscara del conjunto en 50 imágenes de validación (una por grupo), '
           'con la geometría y la métrica de los modelos. γ = (techo − s_red)/(techo − s_clásico): fracción del margen '
           'sobre el clásico que la red aún no cubre.', '',
           f"**Techo:** Dice {n(t['Dice_techo'])} (IC95 bootstrap por grupos {n(t['IC95_bootstrap_grupos'][0])}–"
           f"{n(t['IC95_bootstrap_grupos'][1])}); mediana {n(t['Dice_mediano'])}; IoU {n(t['IoU_medio'])}; por tercil de "
           f"área (pequeño, medio, grande): {n(ter['1'])}, {n(ter['2'])}, {n(ter['3'])}. Clásico: "
           f"{n(g['s_base']['val'])} en validación y {n(g['s_base']['test'])} en test.", '',
           '| Red | Dice val (122 grupos) | Dice en las 50 del techo | Distancia al techo (mismas 50) | γ val | γ test |',
           '|---|---:|---:|---:|---:|---:|']
    filas = sorted(g['gamma_con_techo_medido'], key=lambda r: r['gamma_val'])
    for r in filas:
        s = g['s_k']['val'][r['modelo']]
        m = mismas[r['modelo']]
        lin.append(f"| {NOMBRES[r['modelo']]} | {n(s)} | {n(m)} | {n(t['Dice_techo'] - m)} | {n(r['gamma_val'])} | "
                   f"{n(r['gamma_test'])} |")
    (CARPETA / 'TABLA.md').write_text('\n'.join(lin) + '\n', encoding='utf-8')
    print(CARPETA / 'TABLA.md')


if __name__ == '__main__':
    main()
