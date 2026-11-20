"""Extrae el Dice por imagen de validación guardado en el mejor checkpoint de cada corrida.

scripts/experimento_e2_colab.py guarda en best_epoch_*.pt la clave 'validation'
(lista de {id, group_id, Dice, IoU, mIoU, accuracy}) de la época elegida. Este
script la vuelca a un JSON por corrida, verificando el SHA-256 del checkpoint
contra result.json y que la media por grupos reproduzca best_val_group_Dice.
Necesita torch (Colab o el entorno local del proyecto); no entrena ni evalúa.

Uso:
  python extraer_validacion_por_imagen.py --corridas /ruta/experimento --salida resultados/validacion_por_imagen_e2
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for bloque in iter(lambda: f.read(1 << 20), b''):
            h.update(bloque)
    return h.hexdigest()


def media_por_grupo(detalle):
    grupos = defaultdict(list)
    for fila in detalle:
        grupos[fila['group_id']].append(fila['Dice'])
    return float(np.mean([np.mean(v) for v in grupos.values()]))


def extraer(carpeta_corrida):
    import torch
    resultado = json.loads((carpeta_corrida / 'result.json').read_text(encoding='utf-8'))
    if not resultado.get('completed'):
        raise RuntimeError(f'Corrida incompleta: {carpeta_corrida}')
    mejor = carpeta_corrida / resultado['best_file']
    real = sha256(mejor)
    if real != resultado['best_sha256']:
        raise RuntimeError(f'SHA-256 distinto en {mejor}')
    guardado = torch.load(mejor, map_location='cpu', weights_only=False)
    detalle = guardado['validation']
    media = media_por_grupo(detalle)
    if abs(media - resultado['best_val_group_Dice']) > 1e-9:
        raise RuntimeError(f'La media por grupos no reproduce result.json en {carpeta_corrida}')
    return {'model': resultado['model'], 'seed': resultado['seed'], 'split': 'val',
            'best_file': resultado['best_file'], 'best_sha256': real,
            'epoch': int(guardado['epoch']) + 1, 'val_group_Dice': media,
            'detail': [{k: fila[k] for k in ('id', 'group_id', 'Dice', 'IoU')} for fila in detalle]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corridas', required=True, help='carpeta con <modelo>_seed<semilla>/result.json')
    parser.add_argument('--salida', required=True)
    args = parser.parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    hechas = 0
    for resultado in sorted(Path(args.corridas).glob('*_seed*/result.json')):
        datos = extraer(resultado.parent)
        destino = salida / f"{datos['model']}_seed{datos['seed']}_val_por_imagen.json"
        destino.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding='utf-8')
        hechas += 1
        print(f"{destino.name}: {len(datos['detail'])} imágenes, Dice grupos {datos['val_group_Dice']:.5f}")
    if not hechas:
        raise SystemExit('No se encontró ningún result.json')


if __name__ == '__main__':
    main()
