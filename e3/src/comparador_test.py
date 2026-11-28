"""Comparación visual de casos del test local, sin cargar frameworks de entrenamiento."""
import base64
import csv
import hashlib
import io
import json
import os
import re
import secrets
from pathlib import Path

import numpy as np
from PIL import Image

from . import data as D


def _dentro(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Ruta fuera del paquete de test')
    return path


def catalogo():
    e2=os.environ.get('E3_TEST_E2');root=os.environ.get('E3_TEST_RAIZ')
    if not e2 or not root:
        raise FileNotFoundError('El paquete de test no está conectado a esta instancia del tablero.')
    e2,root=Path(e2),Path(root)
    with (e2/'resultados/preentrenamiento/test.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    out={}
    for row in rows:
        ident=row['id']
        if row['split']!='test' or not re.fullmatch(r'[A-Za-z0-9_-]+',ident) or ident in out:
            raise ValueError('Manifiesto de test inválido')
        out[ident]={**row,'imagen':_dentro(root,row['image_path']),
                    'mascara':_dentro(e2,row['binary_mask_path'])}
    if not out:raise ValueError('El paquete de test está vacío')
    return out


def elegir(anterior=None):
    ids=list(catalogo())
    if len(ids)>1 and anterior in ids:ids.remove(anterior)
    return secrets.choice(ids)


def referencia():
    row=D.redes().sort_values(['dice_val','modelo'],ascending=[False,True]).iloc[0]
    return {'modelo':row.modelo,'nombre':row.nombre,'dice_val':float(row.dice_val),
            'criterio':'Mayor Dice medio de validación del benchmark E2; selección E3 pendiente.'}


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def _url(im):
    im=im.copy();im.thumbnail((640,640))
    buffer=io.BytesIO();im.save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')


def _overlay(image, mask, color):
    array=np.asarray(image.convert('RGB')).copy()
    array[mask]=(array[mask]*.5+np.array(color)*.5).astype('uint8')
    return _url(Image.fromarray(array))


def caso(ident, modelos):
    rows=catalogo()
    if ident not in rows:raise ValueError('La imagen no pertenece al test registrado')
    row=rows[ident]
    raw=row['imagen'].read_bytes();truth_raw=row['mascara'].read_bytes()
    if _sha(raw)!=row['image_sha256'] or _sha(truth_raw)!=row['binary_mask_sha256']:
        raise ValueError('Imagen o máscara distinta del paquete de test verificado')
    with Image.open(io.BytesIO(raw)) as im:image=im.convert('RGB')
    with Image.open(io.BytesIO(truth_raw)) as im:truth=np.asarray(im.convert('L'))>0
    if truth.shape!=(image.height,image.width):raise ValueError('La máscara real no coincide con la imagen')
    result={'id':ident,'group_id':row['group_id'],'imagen':_url(image),
            'real':_overlay(image,truth,[36,190,120]),'modelos':[], 'referencia':referencia()}
    cache=Path(os.environ.get('E3_DEMO_TEST',str(D.dir_modelos()/'demo_test')))
    allowed=set(D.nombres())-{'baseline_clasico'}
    for model in dict.fromkeys(modelos):
        if model not in allowed:raise ValueError('Modelo desconocido')
        entry={'modelo':model,'nombre':D.nombres()[model],'estado':'pendiente',
               'mensaje':'Predicción de este modelo aún no integrada.'}
        manifest=cache/model/'manifest.json'
        if manifest.is_file():
            try:
                meta=json.loads(manifest.read_text(encoding='utf-8'))
                assert meta['modelo']==model
                item=meta['casos'].get(ident)
                if not item:
                    result['modelos'].append(entry);continue
                assert item['image_sha256']==row['image_sha256'] and item['mask_sha256']==row['binary_mask_sha256']
                path=_dentro(cache/model,item['archivo']);mask_bytes=path.read_bytes()
                assert _sha(mask_bytes)==item['sha256']
                with Image.open(io.BytesIO(mask_bytes)) as im:
                    assert im.size==(256,256)
                    scale=256/max(image.size);w=max(1,round(image.width*scale));h=max(1,round(image.height*scale))
                    x,y=(256-w)//2,(256-h)//2
                    mask=np.asarray(im.crop((x,y,x+w,y+h)).resize(image.size,Image.Resampling.NEAREST))>0
                entry.update(estado='ok',superposicion=_overlay(image,mask,[255,123,38]),
                             dice=float(item['Dice']),iou=float(item['IoU']),semilla=meta['semilla'],
                             arquitectura=meta['arquitectura'],mensaje='Predicción precalculada con pesos verificados.')
            except (KeyError,ValueError,AssertionError,OSError):
                entry.update(estado='error',mensaje='La predicción no supera la comprobación de integridad.')
        result['modelos'].append(entry)
    return result
