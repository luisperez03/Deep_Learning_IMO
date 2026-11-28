"""Genera la tabla del libro desde los mismos metadatos que consume el dashboard."""
import csv
import hashlib
import json
from pathlib import Path

E3 = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for modelo in ('fcn', 'unet', 'segnet'):
        ruta = E3 / 'artifacts' / modelo / 'metadata.json'
        m = json.loads(ruta.read_text(encoding='utf-8'))
        rows.append({'modelo': modelo, 'arquitectura': m['arquitectura'], 'semilla': m['semilla'],
                     'estado': m['estado'], 'aprueba': m['ida_y_vuelta']['aprueba'],
                     'torchscript_respaldo_aprueba': (m.get('respaldo_torchscript') or {}).get('aprueba', False),
                     'dice_val_media_5': m['media_val_configuracion'],
                     'de_val_5': m['de_val_configuracion'], 'dice_val_artefacto': m['val_dice_memoria'],
                     'error_prob_max': m['ida_y_vuelta']['error_abs_max'],
                     'mascaras_iguales': m['ida_y_vuelta']['coincidencia_mascara'],
                     'diferencia_dice': m['ida_y_vuelta']['diferencia_dice'],
                     'imagenes_val': m['ida_y_vuelta']['imagenes'], 'onnx_mib': m['onnx_mb'],
                     'parametros': m['parametros'], 'metadata_sha256': hashlib.sha256(ruta.read_bytes()).hexdigest()})
    destino = E3 / 'data/compactas_a0498_exportacion.csv'
    with destino.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(destino)


if __name__ == '__main__':
    main()
