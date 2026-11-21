"""Tabla por imagen de la prueba de sobreajuste (bloque 1 de colab/Complemento_Entrenamiento_E2.ipynb, fila 41).

  python scripts/tabla_sobreajuste_e2.py
  python scripts/tabla_sobreajuste_e2.py --json-dir <salida>/sobreajuste      # regenera el CSV desde los JSON
  python scripts/tabla_sobreajuste_e2.py --mascaras resultados/preentrenamiento/masks_binary   # regenera áreas

Entradas en resultados/complemento_entreno_e2/sobreajuste/:
- resumen_sobreajuste.csv: un renglón por red, sacado de los JSON de `complemento_entreno_e2.py sobreajuste`
  (Dice por imagen en modo evaluación, pérdida al paso 1, 200 y 500). Con --json-dir se rehace desde esos JSON.
- areas_lote.csv: fracción de píxeles del rótulo en la máscara binaria original de las 4 imágenes del lote. Con
  --mascaras se rehace desde las máscaras (están en el paquete v2, no en el repositorio).
Escribe TABLA.md en la misma carpeta. No cambia el umbral (0,90) ni la regla, fijados en
config/complemento_entreno_e2.json antes de correr.
"""
import argparse
import csv
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / 'resultados' / 'complemento_entreno_e2' / 'sobreajuste'
NOMBRES = {'fcn': 'FCN compacta', 'segnet': 'SegNet compacta', 'unet': 'U-Net compacta',
           'deeplabv3plus': 'DeepLabv3+ R18', 'unetplusplus': 'U-Net++ R18', 'pidnet_s': 'PIDNet-S',
           'segnext_t': 'SegNeXt-T', 'segformer_b0': 'SegFormer-B0', 'transunet_2d': 'TransUNet',
           'vmunet': 'VM-UNet'}


def num(x, d=3):
    return f'{float(x):.{d}f}'.replace('.', ',')


def desde_json(carpeta):
    filas, imagenes = [], None
    for p in sorted(Path(carpeta).glob('*.json')):
        d = json.loads(p.read_text(encoding='utf-8'))
        if imagenes is None:
            imagenes = d['imagenes']
        if d['imagenes'] != imagenes:
            raise SystemExit(f'{p.name}: lote distinto al de las demás redes')
        paso200 = next(c['perdida'] for c in d['curva'] if c['paso'] == 200)
        filas.append({'modelo': d['modelo'], 'pasos': d['pasos'],
                      **{f'dice_{i}': f'{v:.4f}' for i, v in zip(imagenes, d['dice_por_imagen'])},
                      'dice_lote': f"{d['dice_lote_modo_eval']:.4f}", 'pasa': str(d['pasa']).lower(),
                      'perdida_inicial': f"{d['perdida_inicial']:.4f}", 'perdida_paso_200': f'{paso200:.4f}',
                      'perdida_final': f"{d['perdida_final']:.4f}", 'segundos': f"{d['segundos']:.1f}"})
    if len(filas) != len(NOMBRES):
        raise SystemExit(f'Se esperaban {len(NOMBRES)} redes y hay {len(filas)}')
    with (CARPETA / 'resumen_sobreajuste.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def areas(mascaras, imagenes):
    import numpy as np
    from PIL import Image
    with (CARPETA / 'areas_lote.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['imagen', 'alto', 'ancho', 'fraccion_rotulo'])
        for i in imagenes:
            m = np.array(Image.open(Path(mascaras) / f'{i}.png'))
            w.writerow([i, m.shape[0], m.shape[1], f'{(m > 0).mean():.4f}'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json-dir')
    ap.add_argument('--mascaras')
    a = ap.parse_args()
    if a.json_dir:
        desde_json(a.json_dir)
    filas = list(csv.DictReader((CARPETA / 'resumen_sobreajuste.csv').open(encoding='utf-8')))
    imagenes = [k[len('dice_'):] for k in filas[0] if k.startswith('dice_image_')]
    if a.mascaras:
        areas(a.mascaras, imagenes)
    area = {r['imagen']: float(r['fraccion_rotulo'])
            for r in csv.DictReader((CARPETA / 'areas_lote.csv').open(encoding='utf-8'))}
    orden = sorted(filas, key=lambda r: -float(r['dice_lote']))
    cab = ' | '.join(f'{i} ({num(area[i] * 100, 1)} %)' for i in imagenes)
    lin = ['# Sobreajuste a un lote: Dice por imagen', '',
           'Generado por `scripts/tabla_sobreajuste_e2.py` desde `resumen_sobreajuste.csv` y `areas_lote.csv`. '
           'Entre paréntesis, el área del rótulo como fracción de la imagen original. Umbral fijado antes: Dice '
           'medio del lote ≥ 0,90 tras 500 pasos. La pérdida es la de la receta de cada red (en PIDNet-S suma '
           'términos auxiliares), así que se lee dentro de cada fila y no entre redes.', '',
           f'| Red | {cab} | Dice del lote | Pasa | Pérdida paso 1 → 200 → 500 |',
           '|---|' + '---:|' * len(imagenes) + '---:|:---:|---:|']
    for r in orden:
        dices = ' | '.join(num(r[f'dice_{i}']) for i in imagenes)
        perd = ' → '.join(num(r[k]) for k in ('perdida_inicial', 'perdida_paso_200', 'perdida_final'))
        lin.append(f"| {NOMBRES[r['modelo']]} | {dices} | {num(r['dice_lote'])} | "
                   f"{'sí' if r['pasa'] == 'true' else '**no**'} | {perd} |")
    (CARPETA / 'TABLA.md').write_text('\n'.join(lin) + '\n', encoding='utf-8')
    print(CARPETA / 'TABLA.md')


if __name__ == '__main__':
    main()
