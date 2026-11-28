"""Predicciones por imagen en validación para la campaña íntegra, sin selección por test."""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e2', type=Path, required=True)
    ap.add_argument('--e3', type=Path, required=True)
    ap.add_argument('--voc', type=Path, required=True)
    ap.add_argument('--campana', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true', help='Solo prueba técnica, dos imágenes, salida separada')
    args = ap.parse_args()
    sys.path[:0] = [str(args.e2/'scripts'), str(args.e3), str(args.e3/'training')]
    import numpy as np
    import torch
    import cv2
    from dataset_ready import load_example, partition, metrics
    from modelos_e2 import MODELS_E2, main_logits
    from compactas_e2 import construir_red
    from src.original_model import UNetAT
    assert cv2.__version__=='4.11.0'
    state = json.loads((args.campana/'estado_campana.json').read_text())
    if args.smoke:
        assert state.get('smoke') is True
    else:
        assert state['lista_para_revision_pretest'], 'Completar y auditar primero las 120 corridas'
    with (args.campana/'inventario_campana.csv').open(encoding='utf-8') as f:
        runs=list(csv.DictReader(f))
    assert (args.smoke or len(runs)==120) and all(r['estado']=='verificado' for r in runs)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    rows=partition('val')
    if args.smoke: rows=rows[:2]
    data=[(row,*load_example(row,data_root=args.voc)) for row in rows]
    out=args.campana/('evaluacion_validacion_humo' if args.smoke else 'evaluacion_validacion'); out.mkdir(exist_ok=True)
    summaries=[]
    for run in runs:
        folder=Path(run['directorio']); name=run['modelo']; seed=int(run['semilla'])
        result=json.loads((folder/'result.json').read_text())
        cfg=json.loads((folder/'config.json').read_text())
        checkpoint=folder/result['best_file']; assert sha(checkpoint)==result['best_sha256']
        destination=out/f'{name}_s{seed}.csv'
        stamp=destination.with_suffix('.json')
        identity={'checkpoint_sha256':result['best_sha256'],'script':sha(Path(__file__)),
                  'data':sha(args.e2/'resultados/preentrenamiento/val.csv'),
                  'normalizacion':sha(args.e2/'resultados/preentrenamiento/normalization_train.json'),
                  'opencv':cv2.__version__,'torch':torch.__version__, 'smoke':args.smoke,
                  'dispositivo':torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}
        if stamp.exists() and destination.exists():
            previous=json.loads(stamp.read_text())
            assert previous['identity']==identity and previous['csv_sha256']==sha(destination)
            summaries.append(previous); continue
        torch.manual_seed(20261118)
        if name.startswith('unetat_'):
            model=UNetAT()
        elif name in ['fcn','unet','segnet']:
            t=cfg['tarea']; model=construir_red(name,t['w'],t['b'],t['dropout_p'])
        else:
            model=MODELS_E2[name]()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
        model=model.to(device).eval(); predictions=[]; groups=defaultdict(list)
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type=='cuda' else []):
            torch.manual_seed(20261118)
            for row,x,y,v in data:
                tensor=torch.from_numpy(x[None]).to(device)
                if name.startswith('unetat_'):
                    prob=model.predict_mask(tensor,name.removeprefix('unetat_'),valida=v[None])[0].astype(np.float32)
                else:
                    prob=main_logits(model(tensor)).sigmoid()[0,0].cpu().numpy()
                scores=metrics(prob,y,v)
                predictions.append(dict(modelo=name,semilla=seed,id=row['id'],group_id=row['group_id'],**scores))
                groups[row['group_id']].append(scores['Dice'])
        with destination.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
        value=float(np.mean([np.mean(g) for g in groups.values()]))
        record={'modelo':name,'semilla':seed,'identity':identity,'csv_sha256':sha(destination),
                'imagenes':len(predictions),'grupos':len(groups),'dice_val':value,
                'dice_guardado':result['best_val_group_Dice'],
                'diferencia_dice':value-result['best_val_group_Dice'],'test_evaluado':False}
        stamp.write_text(json.dumps(record,indent=2),encoding='utf-8'); summaries.append(record)
        print(name,seed,'Dice',value,'delta',record['diferencia_dice'],flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    (out/'resumen.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')


if __name__=='__main__':main()
