"""Test final del piloto de tasa fija en la GPU de Colab."""
import csv
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import PIL
import torch

from dataset_ready import load_example, metrics, partition
from modelos_base import MODELS
from preparar_base import PROJECT


def write_csv(path, records):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def check(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'El hash difiere del registro: {path}')


@torch.inference_mode()
def main():
    dest = PROJECT / 'resultados/test_final_lr0001'
    if dest.exists():
        raise FileExistsError(f'Ya existe una evaluación final en {dest}')
    if not torch.cuda.is_available():
        raise RuntimeError('Selecciona el kernel Colab GPU T4 antes de ejecutar el test.')
    device = torch.device('cuda')
    prep = PROJECT / 'resultados/preentrenamiento'
    freeze = json.loads((prep / 'freeze_manifest.json').read_text())['artifacts']
    for name in ['test.csv', 'train.csv', 'val.csv', 'normalization_train.json']:
        check(prep / name, freeze[name])
    rows = partition('test', allow_test=True)
    groups = sorted({row['group_id'] for row in rows})
    assert len(rows) == len({row['id'] for row in rows}) == 218 and len(groups) == 122
    for part in ['train', 'val']:
        assert set(groups).isdisjoint(row['group_id'] for row in partition(part))
    for row in rows:
        check(PROJECT.parent / row['image_path'], row['image_sha256'])
        check(PROJECT / row['binary_mask_path'], row['binary_mask_sha256'])

    archive = PROJECT / 'piloto_T4_final.tar.gz'
    archive_hash = '90bc33ff5e0647c5d99bf34ec4ff703d23b843e61ca06effc6074479981e4443'
    check(archive, archive_hash)
    prefix = 'IMO_colab_resultados_v1/continuacion_cpu_gpu/'
    networks = {}
    sources = []
    with tarfile.open(archive) as bundle:
        summary = json.load(bundle.extractfile(prefix + 'resumen.json'))
        pairs = [(r['model'], r['seed']) for r in summary]
        assert len(pairs) == 15 and set(pairs) == {(m, s) for m in MODELS for s in [11, 23, 37, 53, 71]}
        for record in summary:
            name, seed = record['model'], record['seed']
            assert record['completed'] and record['lr'] == 0.001
            relative = f'{name}_seed{seed}_lr0.001/best.pt'
            if (seed == 11 or (seed == 23 and name == 'fcn')):
                source = PROJECT / 'resultados/piloto_local_v1' / relative
                raw = source.read_bytes()
                origin = str(source.relative_to(PROJECT))
            else:
                origin = prefix + relative
                raw = bundle.extractfile(origin).read()
            net = MODELS[name]()
            net.load_state_dict(torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True))
            networks[name, seed] = net.to(device).eval()
            sources.append(dict(model=name, seed=seed, file=origin, sha256=hashlib.sha256(raw).hexdigest()))
    print('Verificados 15 checkpoints, 218 imágenes y 122 grupos.', flush=True)
    detail = []
    by_group = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        x, target, valid = load_example(row)
        batch = torch.from_numpy(x.copy())[None].to(device)
        for (model, seed), network in networks.items():
            prob = network(batch)[0, 0].sigmoid().cpu().numpy()
            if not np.isfinite(prob).all():
                raise ValueError('Predicción no finita')
            scores = metrics(prob, target, valid)
            detail.append(dict(model=model, seed=seed, id=row['id'], group_id=row['group_id'], **scores))
            by_group[model, seed, row['group_id']].append(scores)
        if index % 50 == 0:
            print(f'{index}/218 imágenes evaluadas con los 15 modelos.', flush=True)

    runs = []
    models = []
    draws = np.random.default_rng(20261107).integers(len(groups), size=(10000, len(groups)))
    for name in MODELS:
        dice_groups = []
        for seed in [11, 23, 37, 53, 71]:
            group_scores = [by_group[name, seed, g] for g in groups]
            means = {key: [np.mean([s[key] for s in scores]) for scores in group_scores]
                     for key in ('Dice', 'IoU', 'mIoU', 'accuracy')}
            dice_groups.append(means['Dice'])
            runs.append(dict(model=name, seed=seed, **{k: float(np.mean(v)) for k, v in means.items()}))
        avg_by_group = np.mean(dice_groups, axis=0)
        low, high = np.quantile(avg_by_group[draws].mean(axis=1), [0.025, 0.975])
        model_runs = [r for r in runs if r['model'] == name]
        models.append(dict(model=name, **{k: float(np.mean([r[k] for r in model_runs])) for k in means},
                           Dice_SD_semillas=float(np.std([r['Dice'] for r in model_runs], ddof=1)),
                           Dice_IC95_grupos_inf=float(low), Dice_IC95_grupos_sup=float(high)))
    meta = dict(rows=len(rows), groups=len(groups), threshold=0.5, lr=0.001,
                checkpoint_selection='best.pt por Dice de validación', device=torch.cuda.get_device_name(device),
                cuda=torch.version.cuda, torch=str(torch.__version__),
                python=sys.version, numpy=np.__version__, pillow=PIL.__version__, archive_sha256=archive_hash,
                frozen_files=freeze, checkpoints=sources, bootstrap=dict(unit='group_id', draws=10000, seed=20261107))
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        ready = Path(tmp) / 'evaluacion'
        ready.mkdir()
        write_csv(ready / 'por_imagen.csv', detail)
        write_csv(ready / 'por_corrida.csv', runs)
        write_csv(ready / 'resumen.csv', models)
        (ready / 'procedencia.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        ready.rename(dest)
    print(json.dumps(models, indent=2))


if __name__ == '__main__':
    main()
