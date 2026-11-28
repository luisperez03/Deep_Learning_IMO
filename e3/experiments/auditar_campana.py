"""Inventario por rutas explícitas, sin inferir carpetas ni cargar imágenes de test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

SEEDS = [11, 23, 37, 53, 71, 89, 97, 101, 103, 107]
OTHERS = ['transunet_2d', 'pidnet_s', 'vmunet', 'segnext_t', 'deeplabv3plus', 'unetplusplus']


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campana', type=Path, required=True)
    ap.add_argument('--benchmark', type=Path, required=True)
    ap.add_argument('--compactas', type=Path, required=True)
    args = ap.parse_args()
    targets = []
    for s in SEEDS:
        for arm in ['dbnet', 'dbnet_at']:
            targets.append((f'unetat_{arm}', s, args.campana / 'original' / f'unetat_{arm}_seed{s}'))
        for family in ['fcn', 'unet', 'segnet']:
            root = args.compactas if s in SEEDS[:5] else args.campana / 'compactas'
            targets.append((family, s, root / 'fase2' / f'{family}_a0498_s{s}'))
        for model in OTHERS + ['segformer_b0']:
            session = 'A' if model in OTHERS[:2] else 'B'
            if model == 'segformer_b0': session = 'C'
            root = args.benchmark if s in SEEDS[:5] and model != 'segformer_b0' else args.campana / f'rivales/sesion_{session}'
            targets.append((model, s, root / f'{model}_seed{s}'))
    rows = []
    for model, seed, directory in targets:
        row = dict(modelo=model, semilla=seed, directorio=str(directory), estado='falta_resultado', dice_val='')
        rp = directory / 'result.json'
        if rp.exists():
            try:
                r = json.loads(rp.read_text())
                cfg = json.loads((directory / 'config.json').read_text())
                assert cfg['fingerprint'] == r['fingerprint']
                assert r.get('completed') is True and r.get('seed', r.get('semilla')) == seed
                assert r.get('model', r.get('familia')) == model
                if model in ['fcn', 'unet', 'segnet']:
                    assert r['arq'] == 'a0498'
                    assert cfg['tarea']['w'] == [64,64,64] and cfg['tarea']['b'] == [0,1,0]
                if model.startswith('unetat_'):
                    assert cfg['identity']['opencv'] == '4.11.0'
                if model == 'segformer_b0':
                    recipe = cfg['config']['models'][model]
                    assert recipe['optimizer']=='AdamW' and abs(recipe['lr']-.0005573638936676344)<1e-14
                    assert recipe['weight_decay']==.0001
                name = r['best_file']
                assert Path(name).name == name
                assert sha(directory / name) == r['best_sha256']
                row.update(estado='verificado', dice_val=r['best_val_group_Dice'])
            except (KeyError, ValueError, AssertionError, OSError) as e:
                row['estado'] = 'invalido:' + type(e).__name__
        rows.append(row)
    args.campana.mkdir(parents=True, exist_ok=True)
    with (args.campana / 'inventario_campana.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = dict(esperadas=len(rows), verificadas=sum(r['estado']=='verificado' for r in rows),
                  test_evaluado=False, pendientes=[r for r in rows if r['estado']!='verificado'])
    report['lista_para_revision_pretest'] = report['verificadas'] == report['esperadas']
    (args.campana / 'estado_campana.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['lista_para_revision_pretest'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
