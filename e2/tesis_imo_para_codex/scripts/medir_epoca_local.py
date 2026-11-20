"""Una época FCN + validación para presupuesto; no inicia el piloto ni lee test."""
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from preflight import memory_available


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'resultados/medicion_local'
    out.mkdir(exist_ok=True)
    report = {'status': 'pending', 'model': 'fcn', 'seed': 11, 'train_steps': 0,
              'val_images': 0, 'test_accessed': False, 'minimum_free_ram_gib': 2,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'ram_samples_gib': []}
    def save():
        (out/'epoca_fcn.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    def check():
        ram = memory_available()
        if ram is None or ram < 2:
            raise MemoryError(f'Reserva insuficiente: {ram} GiB libres')
        report['ram_samples_gib'].append(ram)
    try:
        check()
        import torch
        import numpy as np
        from collections import defaultdict
        from colab_runner import setup, tensors, loss_fn, environment
        from dataset_ready import partition, metrics
        from modelos_base import SmallFCN
        from inicio_voc import sha
        setup(11); torch.set_num_interop_threads(1)
        report['environment'] = environment()
        report['determinism'] = 'seeded_warn_only'
        report['source_hashes'] = {p: sha(root/p) for p in [
            'scripts/medir_epoca_local.py', 'scripts/modelos_base.py',
            'scripts/colab_runner.py', 'scripts/dataset_ready.py',
            'resultados/preentrenamiento/train.csv', 'resultados/preentrenamiento/val.csv',
            'resultados/preentrenamiento/normalization_train.json']}
        train, val = partition('train'), partition('val')
        for row in train + val:
            if sha(root.parent/row['image_path']) != row['image_sha256']:
                raise ValueError('Imagen modificada: '+row['id'])
            if sha(root/row['binary_mask_path']) != row['binary_mask_sha256']:
                raise ValueError('Máscara modificada: '+row['id'])
        device = torch.device('cpu')
        model = SmallFCN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        report.update(batch_size=1, size=256, threads=2, lr=0.001,
                      parameters=sum(p.numel() for p in model.parameters()))
        random.Random(11*100000).shuffle(train)
        model.train(); start = time.perf_counter(); total_loss = 0
        for row in train:
            check()
            x, y, valid = tensors(row, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y, valid)
            if not torch.isfinite(loss):
                raise ValueError('Pérdida no finita')
            loss.backward(); optimizer.step()
            total_loss += loss.item(); report['train_steps'] += 1
            if report['train_steps'] % 100 == 0:
                print('Train:', report['train_steps'], '/', len(train), flush=True)
                save()
        report['train_seconds'] = time.perf_counter()-start
        report['train_loss'] = total_loss/len(train)
        # Validación sin gradientes: nunca actualiza pesos ni estadísticas.
        model.eval(); start = time.perf_counter(); groups = defaultdict(list); detail = []
        with torch.no_grad():
            for row in val:
                check()
                x, y, valid = tensors(row, device)
                scores = metrics(torch.sigmoid(model(x))[0, 0].numpy(),
                                 y[0, 0].numpy(), valid[0, 0].numpy())
                groups[row['group_id']].append(scores['Dice'])
                detail.append({'id': row['id'], 'group_id': row['group_id'], **scores})
                report['val_images'] += 1
        report['val_seconds'] = time.perf_counter()-start
        report['val_group_Dice'] = float(np.mean([np.mean(v) for v in groups.values()]))
        report['epoch_seconds'] = report['train_seconds']+report['val_seconds']
        report['status'] = 'completed_one_epoch_budget_only'
        report['minimum_sampled_free_ram_gib'] = min(report['ram_samples_gib'])
        (out/'val_predictions.json').write_text(json.dumps(detail, indent=2), encoding='utf-8')
        torch.save(model.state_dict(), out/'fcn_one_epoch.pt')
    except Exception as exc:
        report['status'] = 'stopped_low_memory' if isinstance(exc, MemoryError) else 'failed'
        report['error'] = repr(exc)
    save()
    print(json.dumps({k:v for k,v in report.items() if k not in ['ram_samples_gib','source_hashes']}, indent=2))
    return 0 if report['status'] == 'completed_one_epoch_budget_only' else 2


if __name__ == '__main__':
    raise SystemExit(main())
