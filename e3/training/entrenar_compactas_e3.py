"""Amplía a0498 con las cinco semillas E3; reutiliza exactamente el entrenador E2."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e2', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--smoke-voc', help='Solo humo CPU: dos imágenes, una época, salida separada')
    ap.add_argument('--smoke-masks', help='Raíz de máscaras existentes, solo para humo local')
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.e2) / 'scripts'))
    import busqueda_arquitectura_e2 as B
    import torch
    e3 = Path(__file__).resolve().parents[1]
    train, val = B.C.partition('train'), B.C.partition('val')
    proto = json.loads((Path(args.e2) / 'config/experimento_e2.json').read_text())
    if args.smoke_voc:
        train, val = [B.C.reubicar(rows[:2], args.smoke_voc, args.smoke_masks) for rows in [train, val]]
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('Se requiere GPU para la campaña')
        device = torch.device('cuda')
    ident = B.identidad(SimpleNamespace(local=bool(args.smoke_voc)), train, val)
    ident['extension_e3'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for family in ['fcn', 'unet', 'segnet']:
        template = json.loads((e3 / f'config/plantilla_{family}_a0498.json').read_text())['tarea']
        assert template['arq'] == 'a0498' and template['w'] == [64, 64, 64]
        assert template['b'] == [0, 1, 0] and template['dropout_p'] == .2
        for seed in ([89] if args.smoke_voc else [89, 97, 101, 103, 107]):
            task = copy.deepcopy(template)
            task.update(semilla=seed, id=f'{family}_a0498_s{seed}')
            if args.smoke_voc:
                task.update(epocas=1, prueba_local=True)
            B.entrenar(task, out, proto, ident, device, train, val)


if __name__ == '__main__':
    main()
