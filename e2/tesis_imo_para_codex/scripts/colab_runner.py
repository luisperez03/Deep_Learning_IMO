"""Piloto portable: train/val únicamente, checkpoints por época y semillas comunes.

No modifica particiones ni accede al test. Las adaptaciones no son réplicas de
los artículos. Las advertencias de determinismo se conservan en el cuaderno:
semillas no garantizan igualdad bit a bit entre equipos/versiones.
"""
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import PIL
import torch
from dataset_ready import load_example, metrics, partition
from modelos_base import MODELS


def environment():
    return {'torch': str(torch.__version__), 'numpy': np.__version__,
            'pillow': PIL.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Algunas operaciones (p. ej. unpooling) dependen del backend. No ocultar
    # advertencias ni prometer determinismo estricto cuando no esté disponible.
    torch.use_deterministic_algorithms(True, warn_only=True)


def tensors(row, device):
    x, y, valid = load_example(row)
    return (torch.from_numpy(x.copy())[None].to(device),
            torch.from_numpy(y.astype(np.float32))[None, None].to(device),
            torch.from_numpy(valid.copy())[None, None].to(device))


def loss_fn(logits, target, valid):
    if logits.shape != target.shape:
        raise ValueError('La salida no coincide con la máscara')
    p, t = torch.sigmoid(logits)[valid], target[valid]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], t)
    dice = 1 - (2 * (p*t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
    return (bce + dice) / 2


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def smoke(output):
    """Cinco pasos/modelo; prueba funcional, NO medición de calidad predictiva."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = partition('train')[:5]
    results = []
    for name, constructor in MODELS.items():
        setup(11)
        model = constructor().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        times = []
        for row in rows:
            sync(device); start = time.perf_counter()
            x, y, v = tensors(row, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y, v)
            if not torch.isfinite(loss):
                raise ValueError('Pérdida no finita')
            loss.backward(); optimizer.step()
            sync(device); times.append(time.perf_counter() - start)
        results.append({'model': name, 'steps': len(times), 'seconds_per_step': times,
                        'parameters': sum(p.numel() for p in model.parameters()),
                        'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20
                        if device.type == 'cuda' else None})
        print(results[-1], flush=True)
        del model, optimizer, x, y, v, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report = {'environment': environment(), 'results': results, 'seed': 11,
              'quality_metrics': 'No se evalúan: son cinco pasos de prueba'}
    (output/'smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


@torch.no_grad()
def evaluate(model, rows, device, guard=None):
    model.eval(); grouped = defaultdict(list); detail = []
    for row in rows:
        if guard:
            guard()
        x, y, v = tensors(row, device)
        scores = metrics(torch.sigmoid(model(x))[0, 0].cpu().numpy(),
                         y[0, 0].cpu().numpy(), v[0, 0].cpu().numpy())
        grouped[row['group_id']].append(scores['Dice'])
        detail.append({'id': row['id'], 'group_id': row['group_id'], **scores})
    return float(np.mean([np.mean(values) for values in grouped.values()])), detail


def run(name, seed, output, epochs=30, lr=0.001, patience=5,
        identity=None, guard=None, max_epochs_this_call=None, force_cpu=False):
    """Reanuda al final de la última época guardada; igual presupuesto/modelo.

    Un corte dentro de una época repite esa época desde el checkpoint anterior.
    Cargar checkpoints solo de esta ejecución privada: torch.load puede ejecutar
    código cuando weights_only=False. No aceptar checkpoints de terceros.
    """
    if name not in MODELS or epochs < 1:
        raise ValueError('Configuración inválida')
    setup(seed)
    device = torch.device('cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    root = Path(__file__).resolve().parents[1]
    # La identidad incluye todos los archivos del paquete, por tanto código,
    # particiones, imágenes y máscaras. Cambios rechazan una reanudación silenciosa.
    package_digest = identity or hashlib.sha256((root.parent/'package_manifest.json').read_bytes()).hexdigest()
    config = {'model': name, 'seed': seed, 'epochs': epochs, 'lr': lr,
              'patience': patience, 'batch_size': 1, 'size': 256,
              'package_manifest_sha256': package_digest, 'environment': environment(),
              'execution_device': str(device)}
    dest = Path(output)/f'{name}_seed{seed}_lr{lr:g}'
    dest.mkdir(parents=True, exist_ok=True)
    cfg = dest/'config.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError('Cambió configuración, datos o entorno. Usar otra carpeta de resultados.')
    cfg.write_text(json.dumps(config, indent=2), encoding='utf-8')
    model = MODELS[name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train, val = partition('train'), partition('val')
    start_epoch, best, stale, history = 0, -1.0, 0, []
    checkpoint = dest/'last.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start_epoch, best, stale, history = saved['epoch']+1, saved['best'], saved['stale'], saved['history']
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    end_epoch = epochs if max_epochs_this_call is None else min(epochs, start_epoch+max_epochs_this_call)
    for epoch in range(start_epoch, end_epoch):
        if stale >= patience:
            break
        model.train(); order = list(train)
        random.Random(seed * 100000 + epoch).shuffle(order)
        sync(device); start = time.perf_counter(); total_loss = 0
        for row in order:
            if guard:
                guard()
            x, y, v = tensors(row, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y, v)
            if not torch.isfinite(loss):
                raise ValueError('Pérdida no finita')
            loss.backward(); optimizer.step(); total_loss += loss.item()
        score, details = evaluate(model, val, device, guard=guard)
        improved = score > best
        if improved:
            best, stale = score, 0
            torch.save(model.state_dict(), dest/'best.tmp')
            os.replace(dest/'best.tmp', dest/'best.pt')
            (dest/'best_val_predictions.json').write_text(json.dumps(details), encoding='utf-8')
        else:
            stale += 1
        sync(device)
        history.append({'epoch': epoch+1, 'train_loss': total_loss/len(train),
                        'val_group_Dice': score, 'seconds': time.perf_counter()-start})
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best': best, 'stale': stale, 'history': history,
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None},
                   dest/'last.tmp')
        os.replace(dest/'last.tmp', checkpoint)
        (dest/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(name, seed, history[-1], flush=True)
    result = {'model': name, 'seed': seed, 'lr': lr, 'best_val_group_Dice': best,
              'epochs_completed': len(history), 'test_evaluated': False,
              'parameters': sum(p.numel() for p in model.parameters()),
              'completed': len(history) >= epochs or stale >= patience}
    (dest/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
