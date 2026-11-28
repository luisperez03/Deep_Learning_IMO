"""Demostración del original con semilla seleccionada por validación, sin ajustar con test."""
import argparse
import json
import sys
from pathlib import Path
from precalcular_demo_test import sha


def main():
    p=argparse.ArgumentParser()
    for arg in ['e2','voc','campana','output']:
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args(); e3=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(e3),str(a.e2/'scripts')]
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from dataset_ready import partition,load_example,metrics
    from src.original_model import UNetAT
    assert cv2.__version__=='4.11.0'
    torch.set_num_threads(2)
    runs=[]
    for path in (a.campana/'original').glob('unetat_dbnet_at_seed*/result.json'):
        r=json.loads(path.read_text());assert r['completed']
        checkpoint=path.parent/r['best_file'];assert sha(checkpoint)==r['best_sha256']
        runs.append((r,checkpoint))
    assert len(runs)==10
    r,checkpoint=sorted(runs,key=lambda x:(x[0]['best_val_group_Dice'],x[0]['seed']))[4]
    model=UNetAT().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
    a.output.mkdir(parents=True,exist_ok=True)
    meta=dict(modelo='original',semilla=r['seed'],arquitectura='UNet-AT · C1+C2',
              seleccion='Mediana inferior de validación entre diez semillas',
              identity=dict(checkpoint_sha256=sha(checkpoint),script_sha256=sha(Path(__file__)),opencv=cv2.__version__),
              uso='Exploración visual de test; no ajuste ni comparación confirmatoria',casos={})
    for index,row in enumerate(partition('test',allow_test=True)):
        assert sha(a.voc/row['image_path'])==row['image_sha256']
        assert sha(a.e2/row['binary_mask_path'])==row['binary_mask_sha256']
        x,y,v=load_example(row,data_root=a.voc)
        mask=model.predict_mask(torch.from_numpy(x[None]),'dbnet_at',valida=v[None])[0]&v
        dest=a.output/(row['id']+'.png');Image.fromarray(mask.astype('uint8')*255).save(dest)
        meta['casos'][row['id']]=dict(archivo=dest.name,sha256=sha(dest),image_sha256=row['image_sha256'],
                                    mask_sha256=row['binary_mask_sha256'],**metrics(mask,y,v))
        if (index+1)%50==0:print(index+1,'/ 218',flush=True)
    temp=a.output/'manifest.tmp';temp.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    temp.replace(a.output/'manifest.json');print('Original integrado, semilla',r['seed'],flush=True)


if __name__=='__main__':main()
