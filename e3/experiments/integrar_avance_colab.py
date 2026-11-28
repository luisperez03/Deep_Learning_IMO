"""Integra resultados completos de un paquete parcial sin atribuirles métricas de test."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campana', type=Path, required=True)
    p.add_argument('--salida', type=Path, required=True)
    a = p.parse_args()
    rows, curves = [], []
    for path in sorted(a.campana.rglob('result.json')):
        r = json.loads(path.read_text()); folder = path.parent
        c = json.loads((folder/'config.json').read_text())
        assert r['completed'] and c['fingerprint'] == r['fingerprint']
        assert Path(r['best_file']).name == r['best_file']
        assert sha(folder/r['best_file']) == r['best_sha256']
        if r['model'].startswith('unetat_'):
            assert c['identity']['opencv'] == '4.11.0'
        elif r['model'] == 'segformer_b0':
            recipe = c['config']['models']['segformer_b0']
            assert recipe['optimizer'] == 'AdamW' and abs(recipe['lr']-.0005573638936676344) < 1e-14
            assert recipe['weight_decay'] == .0001
        history = json.loads((folder/'history.json').read_text())
        assert len(history) == r['epochs_completed']
        assert abs(max(x['val_group_Dice'] for x in history)-r['best_val_group_Dice']) < 1e-8
        rows.append(dict(modelo=r['model'], semilla=r['seed'], dice_val=r['best_val_group_Dice'],
                         epocas=r['epochs_completed'], segundos=sum(x['seconds'] for x in history),
                         checkpoint=str(path.parent.relative_to(a.campana)/r['best_file']).replace('\\','/'),
                         sha256=r['best_sha256']))
        curves.extend(dict(modelo=r['model'], semilla=r['seed'], epoca=x['epoch'],
                           perdida=x['train_loss'], dice_val=x['val_group_Dice']) for x in history)
    assert rows and len({(r['modelo'],r['semilla']) for r in rows}) == len(rows)
    summaries = []
    for name in sorted({r['modelo'] for r in rows}):
        runs = sorted([r for r in rows if r['modelo']==name], key=lambda r:(r['dice_val'],r['semilla']))
        scores = [r['dice_val'] for r in runs]
        summaries.append(dict(modelo=name, n=len(runs), esperadas=10, media=statistics.mean(scores),
                              de=statistics.stdev(scores) if len(scores)>1 else None,
                              minimo=min(scores), maximo=max(scores), horas=sum(r['segundos'] for r in runs)/3600,
                              semilla_representativa=runs[(len(runs)-1)//2]['semilla'],
                              estado='completo' if len(runs)==10 else 'parcial'))
    paired = {name:{r['semilla']:r['dice_val'] for r in rows if r['modelo']==name}
              for name in ['unetat_dbnet','unetat_dbnet_at']}
    seeds = sorted(set(paired['unetat_dbnet']) & set(paired['unetat_dbnet_at']))
    diffs = [paired['unetat_dbnet_at'][s]-paired['unetat_dbnet'][s] for s in seeds]
    result = dict(corridas=len(rows), resumen=summaries, test_evaluado=False,
                  diferencia_val_at_menos_base=statistics.mean(diffs) if diffs else None,
                  semillas_pareadas=seeds,
                  alcance='Mejor validación por corrida; selección de época. No evaluación confirmatoria de test.')
    a.salida.mkdir(parents=True,exist_ok=True)
    for name, records in [('avance_colab_corridas.csv',rows),('avance_colab_curvas.csv',curves)]:
        with (a.salida/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (a.salida/'avance_colab.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
