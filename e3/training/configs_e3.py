"""Configuraciones del E3 derivadas del E2 (protocolo v2, §2). Escribe en e3/config/:

- `experimento_e3_rivales.json`: la de `config/experimento_e2.json` con un solo cambio, SegFormer-B0 con la
  configuración ganadora de la búsqueda equitativa del cierre del E2 (intento t = 2: lr = 6e-5 · 10^u₂, wd = 1e-4).
  La receta lo dejaba mal ajustado (Dice de validación 0,660 frente a 0,782 con la ganadora,
  `resultados/complemento_entreno_e2/RESUMEN.md`). Es una corrección del bloque A. Las otras 9 redes conservan la
  receta: la búsqueda (1 semilla, 10 épocas) no las mueve más de δ = 0,02.
- `hpo_original_t0.json` a `hpo_original_t3.json`: la misma búsqueda aplicada al original, con su receta (la de la
  U-Net compacta: Adam, lr 1e-3) y el mismo presupuesto por intento (10 épocas, paciencia 3, semilla 11). Es
  exploratoria, como en los rivales: no cambia la configuración confirmatoria del original.

La secuencia de intentos se reproduce con el mismo generador que `scripts/complemento_entreno_e2.py` (secuencia_hpo:
default_rng(20261122), u ~ U(-1, 1), wd en {0; 1e-4; 1e-2}) y se verifica contra los valores publicados.

Uso: python configs_e3.py --e2 <carpeta tesis_imo_para_codex>
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

SALIDA = Path(__file__).resolve().parents[1] / 'config'
T = 4


def secuencia_hpo(semilla=20261122, t_disenado=8):
    """Copia de secuencia_hpo del E2: el intento 0 es la receta; los demás, multiplicador de lr y weight decay."""
    rng = np.random.default_rng(semilla)
    intentos = [{'t': 0, 'exponente_lr': 0.0, 'weight_decay': None}]
    for t in range(1, t_disenado):
        intentos.append({'t': t, 'exponente_lr': float(rng.uniform(-1, 1)),
                         'weight_decay': float(rng.choice([0.0, 1e-4, 1e-2]))})
    return intentos


def aplicar(receta, intento):
    r = dict(receta)
    r['lr'] = float(receta['lr'] * 10 ** intento['exponente_lr'])
    if intento['weight_decay'] is not None:
        r['weight_decay'] = intento['weight_decay']
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e2', required=True)
    a = ap.parse_args()
    base = json.loads((Path(a.e2) / 'config/experimento_e2.json').read_text(encoding='utf-8'))
    seq = secuencia_hpo()
    SALIDA.mkdir(exist_ok=True)

    rivales = copy.deepcopy(base)
    seg = aplicar(base['models']['segformer_b0'], seq[2])
    # Verificación contra la tabla publicada del E2 (lr 5,57e-4, wd 1e-4).
    assert abs(seg['lr'] - 5.57e-4) < 1e-6 and seg['weight_decay'] == 1e-4, seg
    rivales['models']['segformer_b0'] = seg
    rivales['experiment'] = 'e3_rivales_v2'
    rivales['seeds'] = [11, 23, 37, 53, 71, 89, 97, 101, 103, 107]
    rivales['nota_e3'] = ('Igual a experimento_e2.json salvo segformer_b0: configuración ganadora (t = 2) de la búsqueda '
                          'equitativa del cierre del E2. Corrección del bloque A del E3.')
    (SALIDA / 'experimento_e3_rivales.json').write_text(json.dumps(rivales, ensure_ascii=False, indent=2), encoding='utf-8')

    for intento in seq[:T]:
        c = copy.deepcopy(base)
        c['models']['unet'] = aplicar(base['models']['unet'], intento)
        c.update({'experiment': f'e3_hpo_original_t{intento["t"]}', 'epochs': 10, 'patience': 3, 'seeds': [11],
                  'nota_e3': 'Búsqueda exploratoria del original con el presupuesto de la del E2 (10 épocas, paciencia 3, '
                             'semilla 11). models.unet es la receta del original.'})
        (SALIDA / f'hpo_original_t{intento["t"]}.json').write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding='utf-8')
        print(intento['t'], c['models']['unet'])
    print('SegFormer-B0:', seg)


if __name__ == '__main__':
    main()
