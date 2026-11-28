import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import comparador_test as C

ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture
def paquete(monkeypatch):
    monkeypatch.setenv('E3_TEST_E2',str(ROOT/'tesis_imo_para_codex'))
    monkeypatch.setenv('E3_TEST_RAIZ',str(ROOT))
    return C.catalogo()


def test_seleccion_solo_test_sin_repeticion_inmediata(paquete,monkeypatch):
    assert len(paquete)==218 and all(r['split']=='test' for r in paquete.values())
    monkeypatch.setattr(C.secrets,'choice',lambda choices:choices[0])
    first=C.elegir();second=C.elegir(first)
    assert first in paquete and second in paquete and first!=second
    assert C.referencia()['modelo']=='transunet_2d'


def test_no_acepta_imagen_fuera_del_manifiesto(paquete):
    with pytest.raises(ValueError):C.caso('../../archivo',[])
    with pytest.raises(ValueError):C.caso('image_0001',[])


def test_compara_mask_verificada_y_rechaza_alteracion(paquete,tmp_path,monkeypatch):
    ident=next(iter(paquete));row=paquete[ident]
    folder=tmp_path/'fcn';folder.mkdir()
    path=folder/f'{ident}.png';Image.fromarray(np.zeros((256,256),dtype='uint8')).save(path)
    meta=dict(modelo='fcn',semilla=53,arquitectura='a0498',casos={ident:dict(archivo=path.name,
              sha256=hashlib.sha256(path.read_bytes()).hexdigest(),image_sha256=row['image_sha256'],
              mask_sha256=row['binary_mask_sha256'],Dice=0.,IoU=0.)})
    (folder/'manifest.json').write_text(json.dumps(meta))
    monkeypatch.setenv('E3_DEMO_TEST',str(tmp_path))
    case=C.caso(ident,['transunet_2d','fcn'])
    assert case['modelos'][0]['estado']=='pendiente'
    assert case['modelos'][1]['estado']=='ok'
    assert case['modelos'][1]['semilla']==53
    assert case['real'].startswith('data:image/png;base64,')
    path.write_bytes(b'alterado')
    assert C.caso(ident,['fcn'])['modelos'][0]['estado']=='error'


def test_callback_comparacion_http(paquete):
    import app
    client=app.server.test_client();client.get('/_dash-layout')
    result=client.post('/_dash-update-component',json={
        'output':'test-comparacion.children',
        'outputs':{'id':'test-comparacion','property':'children'},
        'inputs':[{'id':'test-caso-id','property':'data','value':next(iter(paquete))},
                  {'id':'test-modelos','property':'value','value':['fcn','unet']}],
        'state':[], 'changedPropIds':['test-caso-id.data']})
    assert result.status_code==200
    content=json.dumps(result.get_json(),ensure_ascii=False)
    assert 'TransUNet' in content and 'Ganador de referencia E2' in content
    assert 'Máscara real' in content and 'data:image/png;base64,' in content


def test_sin_paquete_muestra_estado(monkeypatch):
    monkeypatch.delenv('E3_TEST_E2',raising=False)
    monkeypatch.delenv('E3_TEST_RAIZ',raising=False)
    with pytest.raises(FileNotFoundError):C.catalogo()
