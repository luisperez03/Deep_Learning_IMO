"""Carga diferida y métricas verificables, sin importar frameworks ni crear modelos."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from preparar_base import PROJECT,read_csv,letterbox

def partition(name,allow_test=False):
    if name not in ('train','val','test'):raise ValueError('Partición desconocida')
    if name=='test' and not allow_test:raise PermissionError('Test reservado; requiere autorización explícita de evaluación final')
    return read_csv(PROJECT/'resultados/preentrenamiento'/f'{name}.csv')

def load_example(row,data_root=None):
    """Devuelve X CHW float32, Y uint8 0/1 y zona válida sin padding.

    Solo carga una imagen por llamada. No mezcla train y test, no hace aumentos
    ni ajusta media/desviación con la imagen actual: usa exclusivamente train.
    """
    data_root=Path(data_root) if data_root else PROJECT.parent
    norm=json.loads((PROJECT/'resultados/preentrenamiento/normalization_train.json').read_text())
    with Image.open(data_root/row['image_path']) as im:im,valid=letterbox(im.convert('RGB'),norm['size'])
    with Image.open(PROJECT/row['binary_mask_path']) as mask:mask,_=letterbox(mask.convert('L'),norm['size'],True)
    y=np.asarray(mask,dtype=np.uint8)
    if not np.isin(y,[0,1]).all():raise ValueError('Máscara debe contener 0/1, no 0/255')
    x=np.asarray(im,dtype=np.float32)/255
    x=(x-np.asarray(norm['mean'],dtype=np.float32))/np.maximum(np.asarray(norm['std'],dtype=np.float32),1e-8)
    x[~valid]=0
    return x.transpose(2,0,1),y,valid

def metrics(prediction,target,valid=None):
    """Dice/IoU por imagen; excluir padding. Vacío contra vacío recibe 1.

    La inferencia estadística posterior se agrupa por grupo de similitud, nunca
    trata los píxeles como réplicas independientes.
    """
    p=np.asarray(prediction)>=0.5;t=np.asarray(target).astype(bool)
    if p.shape!=t.shape:raise ValueError('Formas diferentes')
    v=np.ones(t.shape,dtype=bool) if valid is None else np.asarray(valid,dtype=bool)
    if v.shape!=t.shape or not v.any():raise ValueError('Zona válida incorrecta')
    p=p[v];t=t[v]
    tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum());tn=int((~p&~t).sum())
    dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0
    iou=tp/(tp+fp+fn) if tp+fp+fn else 1.0
    bg=tn/(tn+fp+fn) if tn+fp+fn else 1.0
    return {'Dice':dice,'IoU':iou,'mIoU':(iou+bg)/2,'accuracy':(tp+tn)/len(t)}
