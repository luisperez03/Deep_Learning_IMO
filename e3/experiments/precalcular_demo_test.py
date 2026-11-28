"""Demostración visual de test autorizada por el usuario; no selecciona ni ajusta modelos con test."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--e2',type=Path,required=True);p.add_argument('--voc',type=Path,required=True)
    p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--compactas',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modelos',nargs='+',default=['transunet_2d','fcn','unet','segnet','deeplabv3plus','unetplusplus','pidnet_s','segnext_t','segformer_b0'])
    p.add_argument('--device',default='cpu');a=p.parse_args()
    e3=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(a.e2/'scripts'),str(e3/'training')]
    import numpy as np
    import torch
    from PIL import Image
    from dataset_ready import partition,load_example,metrics
    from modelos_e2 import MODELS_E2,main_logits
    from seleccion_compactas import seleccionar,cargar
    from exportar_onnx import elegir_semilla
    torch.set_num_threads(2)
    rows=partition('test',allow_test=True)
    data=[]
    for row in rows:
        assert sha(a.voc/row['image_path'])==row['image_sha256']
        assert sha(a.e2/row['binary_mask_path'])==row['binary_mask_sha256']
        data.append((row,*load_example(row,data_root=a.voc)))
    a.output.mkdir(parents=True,exist_ok=True)
    for name in a.modelos:
        out=a.output/name;out.mkdir(exist_ok=True)
        try:
            compact=name in ['fcn','unet','segnet']
            if compact:
                choice=seleccionar(a.compactas,name,json.loads((e3/'config/hashes_compactas.json').read_text()))
                checkpoint=Path(choice['checkpoint'])
                arch='a0498';seed=choice['semilla']
            else:
                choice=elegir_semilla(a.benchmark,name);checkpoint=choice['checkpoint'];seed=choice['semilla']
                result=json.loads((checkpoint.parent/'result.json').read_text())
                assert sha(checkpoint)==result['best_sha256']
                arch='E2 histórico'
            identity=dict(checkpoint_sha256=sha(checkpoint),test_csv_sha256=sha(a.e2/'resultados/preentrenamiento/test.csv'),
                          script_sha256=sha(Path(__file__)),torch=torch.__version__,device=a.device)
            old=out/'manifest.json'
            if old.exists():
                old_data=json.loads(old.read_text())
                if old_data.get('identity')==identity and len(old_data.get('casos',{}))==len(rows):
                    if all(sha(out/item['archivo'])==item['sha256'] for item in old_data['casos'].values()):
                        print(name,'ya verificado',flush=True);continue
            torch.manual_seed(20261118)
            model=cargar(choice,name) if compact else MODELS_E2[name]()
            if not compact:model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
            model=model.to(a.device).eval()
            meta=dict(modelo=name,semilla=seed,arquitectura=arch,identity=identity,
                      seleccion='mediana de Dice de validación; no seleccionada por test',
                      uso='Demostración visual solicitada; no evaluación confirmatoria nueva del E3',casos={})
            devices=[torch.cuda.current_device()] if a.device.startswith('cuda') else []
            with torch.no_grad(),torch.random.fork_rng(devices=devices):
                torch.manual_seed(20261118)
                for index,(row,x,y,v) in enumerate(data):
                    prob=main_logits(model(torch.from_numpy(x[None]).to(a.device))).sigmoid()[0,0].cpu().numpy()
                    assert prob.shape==y.shape and np.isfinite(prob).all()
                    mask=(prob>=.5)&v
                    file=out/(row['id']+'.png');Image.fromarray(mask.astype('uint8')*255).save(file)
                    meta['casos'][row['id']]=dict(archivo=file.name,sha256=sha(file),image_sha256=row['image_sha256'],
                                                 mask_sha256=row['binary_mask_sha256'],**metrics(mask,y,v))
                    if (index+1)%50==0:print(name,index+1,'/',len(rows),flush=True)
            temp=out/'manifest.tmp';temp.write_text(json.dumps(meta,indent=2),encoding='utf-8');temp.replace(old)
            print(name,'completo',len(rows),'casos',flush=True)
            del model
            if a.device.startswith('cuda'):torch.cuda.empty_cache()
        except Exception as error:
            (out/'error.json').write_text(json.dumps({'modelo':name,'error':str(error)},indent=2),encoding='utf-8')
            print(name,'ERROR',repr(error),flush=True)


if __name__=='__main__':main()
