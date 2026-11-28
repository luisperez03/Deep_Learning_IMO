"""Pruebas reales de los tres artefactos y contrato de preprocesamiento E2."""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

E3 = Path(__file__).resolve().parents[1]
ROOT = E3.parents[1]
sys.path.insert(0, str(E3))
from src import data as D, inference


@pytest.mark.parametrize('modelo', ['fcn', 'unet'])
def test_inferencia_real_y_tabla_coherente(modelo):
    meta = inference.metadatos(modelo)
    assert meta['estado'] == 'ok' and meta['arquitectura'] == 'a0498'
    assert meta['ida_y_vuelta']['imagenes'] == 218
    assert meta['ida_y_vuelta']['coincidencia_mascara'] == 1
    raw = (ROOT / 'VOC2012/JPEGImages/image_0006.jpg').read_bytes()
    out = inference.predecir(modelo, raw)
    assert out['estado'] == 'ok' and out['semilla'] == meta['semilla']
    with Image.open(io.BytesIO(raw)) as im:
        original_size = im.size
    for key in ('mascara', 'superposicion'):
        with Image.open(io.BytesIO(base64.b64decode(out[key].split(',')[1]))) as im:
            assert im.size == original_size
    fila = next(r for r in D.tabla_artefactos() if r['Modelo'] == D.nombres()[modelo])
    assert fila['Dice val. artefacto'] == round(meta['val_dice_memoria'], 4)
    assert fila['Semilla'] == meta['semilla']


def test_preprocesamiento_coincide_con_arnes_e2():
    sys.path.insert(0, str(ROOT / 'tesis_imo_para_codex/scripts'))
    from dataset_ready import partition, load_example
    for row in partition('val')[:8]:
        raw = (ROOT / row['image_path']).read_bytes()
        x, caja = inference.preprocesar(raw)
        ref, _, valid = load_example(row, data_root=ROOT)
        np.testing.assert_array_equal(x[0], ref)
        x0, y0, x1, y1 = caja
        assert valid.sum() == (x1 - x0) * (y1 - y0)


def test_segnet_rechaza_onnx_y_conserva_respaldo_verificado():
    import hashlib
    meta = inference.metadatos('segnet')
    assert not meta['ida_y_vuelta']['aprueba'] and meta['artefacto'] is None
    assert inference._sesion('segnet') is None
    respaldo = meta['respaldo_torchscript']
    assert respaldo['aprueba'] and respaldo['imagenes'] == 218 and respaldo['mascaras_iguales']
    ruta = D.dir_modelos() / 'segnet' / respaldo['archivo']
    assert hashlib.sha256(ruta.read_bytes()).hexdigest() == respaldo['sha256']


def test_no_habilita_artefacto_fallido(tmp_path, monkeypatch):
    (tmp_path / 'fcn').mkdir()
    (tmp_path / 'fcn/metadata.json').write_text(json.dumps({'estado': 'ida_y_vuelta_fallida', 'artefacto': 'fcn.onnx'}))
    monkeypatch.setenv('E3_MODELOS_ONNX', str(tmp_path))
    inference._sesion.cache_clear()
    assert inference._sesion('fcn') is None
    inference._sesion.cache_clear()


def test_callback_prediccion_http():
    import app
    client = app.server.test_client()
    client.get('/_dash-layout')
    raw = (ROOT / 'VOC2012/JPEGImages/image_0006.jpg').read_bytes()
    contents = 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')
    respuesta = client.post('/_dash-update-component', json={
        'output': 'salida-prediccion.children',
        'outputs': {'id': 'salida-prediccion', 'property': 'children'},
        'inputs': [{'id': 'upload', 'property': 'contents', 'value': contents},
                   {'id': 'sel-modelo-vivo', 'property': 'value', 'value': 'fcn'}],
        'state': [{'id': 'upload', 'property': 'filename', 'value': 'imagen.jpg'}],
        'changedPropIds': ['sel-modelo-vivo.value']})
    assert respuesta.status_code == 200
    texto = json.dumps(respuesta.get_json())
    assert 'a0498' in texto and 'data:image/png;base64' in texto
