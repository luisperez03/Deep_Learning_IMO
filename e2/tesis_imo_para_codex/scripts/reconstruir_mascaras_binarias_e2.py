"""Reconstruye resultados/preentrenamiento/masks_binary/ desde las máscaras originales y la verifica por SHA-256.

El snapshot del proyecto no lleva las 1.451 máscaras binarias que piden los manifiestos (`binary_mask_path`). Este
script las regenera con la misma función de `preparar_base.py` (`binary_mask`: color distinto de negro -> 1) y
guarda cada PNG solo si su SHA-256 coincide con `binary_mask_sha256` del manifiesto; si los bytes difieren pero los
píxeles son idénticos a lo esperado no puede saberse sin la máscara original, así que se reporta como falla y no se
escribe. No toca imágenes, manifiestos ni particiones.

  python scripts/reconstruir_mascaras_binarias_e2.py --voc RUTA/VOC2012 [--destino RUTA]

`--destino` por defecto es la carpeta del manifiesto (resultados/preentrenamiento/masks_binary). Las máscaras no se
versionan: son datos derivados de imágenes de terceros.
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inicio_voc import palette  # noqa: E402
from preparar_base import PROJECT, binary_mask, read_csv  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--voc', type=Path, required=True)
    ap.add_argument('--destino', type=Path, default=PROJECT / 'resultados/preentrenamiento/masks_binary')
    a = ap.parse_args(argv)
    colores = palette(a.voc / 'labelmap.txt')
    a.destino.mkdir(parents=True, exist_ok=True)
    ok, fallas = 0, []
    for split in ('train', 'val', 'test'):
        for row in read_csv(PROJECT / 'resultados/preentrenamiento' / f'{split}.csv'):
            origen = a.voc / row['original_mask_path'].split('VOC2012/', 1)[1]
            if hashlib.sha256(origen.read_bytes()).hexdigest() != row['original_mask_sha256']:
                fallas.append((row['id'], 'máscara original distinta'))
                continue
            with Image.open(origen) as im:
                b = binary_mask(im, colores)
            buf = io.BytesIO()
            Image.fromarray(b).save(buf, format='PNG')
            datos = buf.getvalue()
            if hashlib.sha256(datos).hexdigest() != row['binary_mask_sha256']:
                fallas.append((row['id'], 'SHA-256 de la binaria distinto'))
                continue
            (a.destino / Path(row['binary_mask_path']).name).write_bytes(datos)
            ok += 1
    print(json.dumps({'escritas_y_verificadas': ok, 'fallas': fallas[:20], 'n_fallas': len(fallas),
                      'destino': str(a.destino)}, ensure_ascii=False, indent=1))
    return 0 if not fallas else 1


if __name__ == '__main__':
    raise SystemExit(main())
