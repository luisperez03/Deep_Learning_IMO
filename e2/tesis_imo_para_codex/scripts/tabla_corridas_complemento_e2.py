"""Una fila por corrida del complemento de entrenamiento (ablación, permutación y HPO).

  python scripts/tabla_corridas_complemento_e2.py --output <carpeta del complemento en Drive> [--destino corridas.csv]

Lee los result.json de abl_*, perm_* y hpo_* y escribe un CSV con lo que el RESUMEN.json no guarda por corrida:
el Dice de validación de cada réplica de la permutación (la distribución nula completa, no solo min/media/máx.)
y los segundos de GPU de cada corrida. Solo lee JSON; no carga pesos ni toca el test.
"""
import argparse
import csv
import json
from pathlib import Path

CAMPOS = ['corrida', 'dice_val', 'mejor_epoca', 'epocas', 'segundos', 'test_evaluado']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True)
    p.add_argument('--destino', default=None, help='Por defecto <output>/corridas.csv')
    a = p.parse_args(argv)
    salida = Path(a.output)
    filas = []
    for patron in ('abl_*/result.json', 'perm_*/result.json', 'hpo_*/result.json'):
        for f in sorted(salida.glob(patron)):
            r = json.loads(f.read_text(encoding='utf-8'))
            if not r.get('completed'):
                continue
            filas.append({'corrida': f.parent.name, 'dice_val': f"{r['best_val_group_Dice']:.4f}",
                          'mejor_epoca': r['best_epoch'], 'epocas': r['epochs_completed'],
                          'segundos': f"{r['seconds_total']:.0f}",
                          'test_evaluado': str(bool(r.get('test_evaluated'))).lower()})
    destino = Path(a.destino) if a.destino else salida/'corridas.csv'
    with destino.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)
    print(destino, len(filas), 'corridas')


if __name__ == '__main__':
    main()
