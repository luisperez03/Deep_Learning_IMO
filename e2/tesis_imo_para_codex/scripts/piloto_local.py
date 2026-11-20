"""Orquesta 15 corridas CPU independientes; reanuda checkpoints por época.

La primera época de semilla 11 mide cada modelo y se conserva dentro del piloto.
El presupuesto es orientativo: no pronostica early stopping ni temperatura.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'resultados/piloto_local_v1'
CONFIG = ROOT/'config/piloto_local.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def guard(minimum):
    from preflight import memory_available
    ram = memory_available()
    if ram is None or ram < minimum:
        raise MemoryError(f'RAM libre: {ram} GiB; se necesitan {minimum}. Cierra ChatGPT y reanuda.')


def freeze():
    """Verifica datos reales y fija código/entorno, sin tocar el conjunto test."""
    files = [CONFIG, ROOT/'requirements-cpu.lock.txt']
    files += [ROOT/'scripts'/name for name in ['piloto_local.py', 'colab_runner.py',
              'modelos_base.py', 'dataset_ready.py', 'preparar_base.py', 'inicio_voc.py']]
    prep = ROOT/'resultados/preentrenamiento'
    frozen = json.loads((prep/'freeze_manifest.json').read_text())
    groups = {}; seen = set()
    for name in ['train.csv', 'val.csv', 'normalization_train.json']:
        if sha(prep/name) != frozen['artifacts'][name]:
            raise ValueError('Preparación congelada modificada: '+name)
        files.append(prep/name)
    for split in ['train', 'val']:
        with (prep/f'{split}.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row['id'] in seen or groups.setdefault(row['group_id'], split) != split:
                raise ValueError('Fuga entre particiones')
            seen.add(row['id'])
            for path, expected in [(ROOT.parent/row['image_path'], row['image_sha256']),
                                   (ROOT/row['binary_mask_path'], row['binary_mask_sha256'])]:
                if sha(path) != expected:
                    raise ValueError('Dato modificado: '+str(path))
    identity = {str(p.relative_to(ROOT)): sha(p) for p in files}
    # Los hashes de train/val incluyen los hashes de cada imagen y máscara.
    identity['installed_packages'] = subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True).strip().splitlines()
    identity['python'] = sys.version
    target = OUT/'identity.json'
    if target.exists() and json.loads(target.read_text()) != identity:
        raise ValueError('Cambió código, entorno o datos. No mezclar con el piloto existente.')
    target.write_text(json.dumps(identity, sort_keys=True, indent=2), encoding='utf-8')
    return sha(target)


def summarize(cfg):
    rows = []
    for seed in cfg['seeds']:
        for name in cfg['models']:
            path = OUT/f'{name}_seed{seed}_lr{cfg["lr"]:g}'/'result.json'
            if path.exists():
                rows.append(json.loads(path.read_text()))
    (OUT/'resumen.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['fcn', 'segnet', 'unet'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--one-epoch', action='store_true')
    parser.add_argument('--measure-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    if cfg['test_allowed'] or cfg['cpu_threads'] != 2 or cfg['batch_size'] != 1 or cfg['image_size'] != 256:
        raise ValueError('Perfil no implementado; no ignorar ajustes del archivo')
    guard(cfg['minimum_free_ram_gib'])
    if args.worker:
        from colab_runner import run
        run(args.worker, args.seed, OUT, epochs=cfg['epochs'], lr=cfg['lr'],
            patience=cfg['patience'], identity=sha(OUT/'identity.json'),
            guard=lambda: guard(cfg['minimum_free_ram_gib']),
            max_epochs_this_call=1 if args.one_epoch else None, force_cpu=True)
        return
    # Bloqueo liberado por Windows incluso al cerrar la consola: no hay que borrar
    # archivos de bloqueo manualmente tras una interrupción.
    import msvcrt
    with (OUT/'execution.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('Ya existe una ejecución del piloto en esta carpeta')
        freeze()
        print('Integridad verificada. Test reservado. Ctrl+C detiene; repetir archivo reanuda.', flush=True)
        def child(name, seed, one=False):
            guard(cfg['minimum_free_ram_gib'])
            command = [sys.executable, '-u', str(Path(__file__)), '--worker', name, '--seed', str(seed)]
            if one:
                command += ['--one-epoch']
            print(f'\nModelo {name}, semilla {seed}', flush=True)
            logpath = OUT/f'{name}_seed{seed}.log'
            with logpath.open('a', encoding='utf-8') as log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding='utf-8', errors='replace') as proc:
                    try:
                        for line in proc.stdout:
                            print(line, end='', flush=True); log.write(line); log.flush()
                        code = proc.wait()
                    except KeyboardInterrupt:
                        proc.terminate(); proc.wait()
                        raise
            summarize(cfg)
            if code:
                raise RuntimeError(f'{name} detenido. Consultar {logpath}; no se inicia otro modelo.')
        # No repetir mediciones completas al reanudar.
        for name in cfg['models']:
            history = OUT/f'{name}_seed11_lr{cfg["lr"]:g}'/'history.json'
            if not history.exists():
                child(name, 11, one=True)
        seconds = {name: json.loads((OUT/f'{name}_seed11_lr{cfg["lr"]:g}'/'history.json').read_text())[0]['seconds']
                   for name in cfg['models']}
        budget = {'measured_first_epoch_seconds': seconds,
                  'projection_hours_without_early_stopping': sum(seconds.values())*len(cfg['seeds'])*cfg['epochs']/3600,
                  'note': 'Proyección, no garantía. No mide temperatura. Incluye épocas ya ejecutadas.'}
        (OUT/'presupuesto.json').write_text(json.dumps(budget, indent=2), encoding='utf-8')
        print(json.dumps(budget, indent=2), flush=True)
        if not args.measure_only:
            for seed in cfg['seeds']:
                for name in cfg['models']:
                    result = OUT/f'{name}_seed{seed}_lr{cfg["lr"]:g}'/'result.json'
                    if not result.exists() or not json.loads(result.read_text()).get('completed'):
                        child(name, seed)
        print('Resultados guardados en '+str(OUT), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/'ultima_interrupcion.txt').write_text(repr(exc), encoding='utf-8')
        print('DETENIDO:', repr(exc), flush=True)
        raise SystemExit(2)
