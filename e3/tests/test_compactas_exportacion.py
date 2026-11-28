"""Validación de pesos reales, selección y equivalencia del unpool exportable."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

E3 = Path(__file__).resolve().parents[1]
ROOT = E3.parents[1]
sys.path.insert(0, str(E3 / 'training'))
from seleccion_compactas import seleccionar, cargar
from unpool_exportable import unpool, adaptar_segnet

PESOS = ROOT / 'avance_e3_recuperado_20261124/artefactos_compactas_a0498'
HASHES = json.loads((E3 / 'config/hashes_compactas.json').read_text())
torch.set_num_threads(2)


@pytest.mark.parametrize('familia', ['fcn', 'unet', 'segnet'])
def test_seleccion_carga_y_forward(familia):
    elegida = seleccionar(PESOS, familia, HASHES)
    assert elegida['arquitectura'] == 'a0498'
    assert elegida['val_dice'] == sorted(elegida['val_por_semilla'].values())[2]
    modelo = cargar(elegida, familia)
    with torch.no_grad():
        y = modelo(torch.zeros(1, 3, 256, 256))
    assert y.shape == (1, 1, 256, 256) and torch.isfinite(y).all()


def test_rechaza_hash_distinto():
    falso = dict(HASHES, fcn_a0498_s11='0' * 64)
    with pytest.raises(ValueError, match='Peso alterado'):
        seleccionar(PESOS, 'fcn', falso)


def test_unpool_exacto_con_indices_y_activaciones_distintas():
    torch.manual_seed(11)
    ref = torch.randn(2, 7, 32, 48)
    pooled, indices = torch.nn.functional.max_pool2d(ref, 2, return_indices=True)
    x = torch.randn_like(pooled)
    original = torch.nn.functional.max_unpool2d(x, indices, 2, output_size=ref.shape)
    assert torch.equal(original, unpool(x, indices, ref))


def test_segnet_equivalente_y_onnx(tmp_path):
    import onnxruntime as ort
    eleccion = seleccionar(PESOS, 'segnet', HASHES)
    original = cargar(eleccion, 'segnet')
    adaptada = adaptar_segnet(original)
    torch.manual_seed(23)
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y0, y1 = original(x), adaptada(x)
    assert torch.equal(y0, y1)
    destino = tmp_path / 'segnet.onnx'
    torch.onnx.export(adaptada, x, destino, opset_version=17, input_names=['imagen'],
                      output_names=['logits'], dynamo=False)
    op = ort.SessionOptions()
    op.intra_op_num_threads = 2
    sesion = ort.InferenceSession(str(destino), sess_options=op, providers=['CPUExecutionProvider'])
    actual = sesion.run(None, {'imagen': x.numpy()})[0]
    np.testing.assert_allclose(actual, y0.numpy(), atol=1e-4, rtol=1e-4)
