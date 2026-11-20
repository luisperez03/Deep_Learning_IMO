"""Prueba autorizada de cinco pasos; no es entrenamiento comparativo ni piloto.

Ejecutar por separado para cada modelo. Solo lee train y no guarda pesos.
Rechaza iniciar por debajo de la reserva de RAM; esta reserva no garantiza
ausencia de picos de memoria ni controla temperatura.
"""
import argparse
import importlib.util
import json
import time
import platform
from datetime import datetime, timezone
from pathlib import Path
from preflight import memory_available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['fcn', 'segnet', 'unet'], default='unet')
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[1] / 'resultados/preentrenamiento'
    report = {'model': args.model, 'steps_completed': 0, 'training_comparison_started': False,
              'free_ram_before_gib': memory_available(), 'status': 'pending',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'python': platform.python_version(), 'device': 'cpu',
              'minimum_free_ram_gib': 2.0, 'test_accessed': False}
    def save():
        (out / f'carga_{args.model}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    def enough():
        ram = memory_available()
        return ram is not None and ram >= 2.0
    if not enough():
        report['status'] = 'blocked_free_ram_below_2_GiB'
        save()
        return 2
    if importlib.util.find_spec('torch') is None:
        report['status'] = 'blocked_torch_not_installed'
        save()
        return 2
    import torch
    import numpy as np
    from dataset_ready import partition, load_example
    from modelos_base import MODELS
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(11)
    np.random.seed(11)
    # Unpooling puede no ofrecer determinismo estricto según backend. Conservar
    # las advertencias en el log, como en el runner de Colab; no prometer bit a bit.
    torch.use_deterministic_algorithms(True, warn_only=True)
    report['determinism'] = 'seeded_warn_only; conservar advertencias del backend'
    model = MODELS[args.model]()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    report.update(torch_version=torch.__version__, seed=11, cpu_threads=2,
                  parameters=sum(p.numel() for p in model.parameters()), seconds_per_step=[])
    rows = partition('train')[:5]
    from inicio_voc import sha
    project = Path(__file__).resolve().parents[1]
    report['source_hashes'] = {name: sha(project/name) for name in [
        'scripts/modelos_base.py', 'scripts/dataset_ready.py',
        'scripts/prueba_carga.py', 'resultados/preentrenamiento/train.csv',
        'resultados/preentrenamiento/normalization_train.json']}
    report['image_ids'] = []
    report['free_ram_after_each_step_gib'] = []
    try:
        for row in rows:
            if not enough():
                report['status'] = 'stopped_low_free_ram'
                break
            start = time.perf_counter()
            x, y, valid = load_example(row)
            x = torch.from_numpy(x.copy())[None]
            y = torch.from_numpy(y.astype(np.float32))[None, None]
            valid = torch.from_numpy(valid.copy())[None, None]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if logits.shape != y.shape:
                raise ValueError('Forma de salida incorrecta')
            # Se excluye el relleno artificial: no es fondo observado.
            p, t = torch.sigmoid(logits)[valid], y[valid]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], t)
            dice_loss = 1 - (2 * (p*t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
            loss = 0.5 * (bce + dice_loss)
            if not torch.isfinite(loss):
                raise ValueError('Pérdida no finita')
            loss.backward()
            optimizer.step()
            report['steps_completed'] += 1
            report['seconds_per_step'].append(time.perf_counter() - start)
            report['image_ids'].append(row['id'])
            report['free_ram_after_each_step_gib'].append(memory_available())
        else:
            report['status'] = 'completed_smoke_only'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
    report['free_ram_after_gib'] = memory_available()
    save()
    return 0 if report['status'] == 'completed_smoke_only' else 2


if __name__ == '__main__':
    raise SystemExit(main())
