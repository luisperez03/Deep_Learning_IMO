"""Compara pesos y métricas de un humo continuo y otro pausado, no datos de test."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directorio', type=Path)
    args = parser.parse_args()
    result = {}
    for arm in ['dbnet', 'dbnet_at']:
        run = f'unetat_{arm}_seed11'
        # Archivos locales de confianza producidos por REPRODUCIR_HUMO_ORIGINAL.ps1.
        a = torch.load(args.directorio / 'reanudada' / run / 'last.pt', map_location='cpu', weights_only=False)
        b = torch.load(args.directorio / 'continua' / run / 'last.pt', map_location='cpu', weights_only=False)
        assert a['fingerprint'] == b['fingerprint'], 'Configuraciones distintas'
        assert a['epoch'] == b['epoch'] == 1, 'Deben ser dos épocas'
        assert a['model'].keys() == b['model'].keys()
        assert all(torch.equal(v, b['model'][k]) for k, v in a['model'].items()), 'Pesos distintos'
        histories = [[{k: v for k, v in h.items() if not k.startswith('seconds')}
                      for h in checkpoint['history']] for checkpoint in [a, b]]
        assert histories[0] == histories[1], 'Métricas distintas'
        result[arm] = {'pesos_identicos': True, 'metricas_identicas': True, 'epocas': 2}
    (args.directorio / 'verificacion_reanudacion.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
