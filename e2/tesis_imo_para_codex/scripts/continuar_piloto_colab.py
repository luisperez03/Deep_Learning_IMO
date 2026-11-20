"""Continúa en la GPU el piloto local incluido en el paquete de continuación."""
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = Path(__file__).resolve().parent
    base = Path('/content/imo_workspace')
    project = base / 'tesis_imo_para_codex'
    output = Path('/content/IMO_colab_resultados_v1')
    dest = output / 'continuacion_cpu_gpu'
    dest.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), 'Se requiere una GPU.'
    identity = read(source / 'identity.json')
    manifest = base / 'package_manifest.json'
    for name in ('train.csv', 'val.csv', 'normalization_train.json'):
        path = Path('resultados/preentrenamiento') / name
        assert digest(project / path) == identity[str(path).replace('/', chr(92))]
    for name in ('modelos_base.py', 'dataset_ready.py', 'preparar_base.py', 'inicio_voc.py'):
        assert digest(project / 'scripts' / name) == identity['scripts' + chr(92) + name]
    runner_path = source / 'colab_runner.py'
    assert digest(runner_path) == identity['scripts' + chr(92) + 'colab_runner.py']
    sys.path.insert(0, str(project / 'scripts'))
    spec = importlib.util.spec_from_file_location('runner_continuacion', runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    env = runner.environment()
    gpu_identity = hashlib.sha256((digest(manifest) + digest(runner_path)).encode()).hexdigest()
    local = source / 'segnet_seed23_lr0.001'
    cpu_config = read(local / 'config.json')
    assert cpu_config['package_manifest_sha256'] == digest(source / 'identity.json')
    cfg = dict(cpu_config, environment=env, execution_device='cuda',
               package_manifest_sha256=gpu_identity)
    migration = {'source': cpu_config, 'destination': cfg,
                 'last_cpu_epoch': read(local / 'history.json')[-1]['epoch'],
                 'source_checkpoint_sha256': digest(local / 'last.pt')}
    provenance = dest / 'migracion.json'
    if provenance.exists() and read(provenance) != migration:
        raise ValueError('La procedencia o el entorno difieren de la continuación anterior.')
    provenance.write_text(json.dumps(migration, indent=2))
    partial = dest / local.name
    if not partial.exists():
        stage = dest / (local.name + '.preparacion')
        shutil.copytree(local, stage, dirs_exist_ok=True)
        (stage / 'config.json').write_text(json.dumps(cfg, indent=2))
        stage.rename(partial)
    results = read(source / 'resumen.json')
    finished = {(r['model'], r['seed']) for r in results if r['completed']}
    print('Corridas CPU completas:', sorted(finished), flush=True)
    print('Continuación en', env['device'], 'desde época CPU', migration['last_cpu_epoch'], flush=True)
    smoke_path = output / 'smoke.json'
    if not smoke_path.exists() or read(smoke_path)['environment'] != env:
        runner.smoke(output)
    assert {r['model'] for r in read(smoke_path)['results'] if r['steps'] == 5} == set(runner.MODELS)
    for seed in [11, 23, 37, 53, 71]:
        for name in ['fcn', 'segnet', 'unet']:
            if (name, seed) in finished:
                continue
            print('Ejecutando', name, 'semilla', seed, flush=True)
            result = runner.run(name, seed, dest, identity=gpu_identity)
            results.append(result)
            (dest / 'resumen.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
