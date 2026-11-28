"""Respaldo CPU verificado cuando ONNX no cumple la tolerancia. No habilita ONNX en la app."""
import hashlib
import time
from pathlib import Path

import numpy as np
import torch


def exportar_respaldo(modelo, filas, destino, metrics):
    from exportar_onnx import dice_por_grupo, filas_meta

    class Probabilidad(torch.nn.Module):
        def __init__(self, red):
            super().__init__()
            self.red = red

        def forward(self, x):
            return torch.sigmoid(self.red(x))

    modelo = Probabilidad(modelo.cpu().eval()).eval()
    ruta = Path(destino) / 'segnet_torchscript.pt'
    x0 = torch.from_numpy(filas[0][0][None])
    script = torch.jit.trace(modelo, x0, check_trace=True)
    torch.jit.save(script, str(ruta))
    restaurado = torch.jit.load(str(ruta), map_location='cpu').eval()
    originales, exportadas = [], []
    error, iguales = 0.0, True
    inicio = time.perf_counter()
    with torch.no_grad():
        for i, (x, _, _, _) in enumerate(filas):
            tensor = torch.from_numpy(x[None])
            a, b = modelo(tensor).numpy()[0, 0], restaurado(tensor).numpy()[0, 0]
            error = max(error, float(np.abs(a - b).max()))
            iguales = iguales and np.array_equal(a >= 0.5, b >= 0.5)
            originales.append(a)
            exportadas.append(b)
            if (i + 1) % 50 == 0:
                print(f'TorchScript: {i + 1}/{len(filas)} imágenes', flush=True)
    dice_a = dice_por_grupo(originales, filas_meta(filas), metrics)
    dice_b = dice_por_grupo(exportadas, filas_meta(filas), metrics)
    return {'archivo': ruta.name, 'sha256': hashlib.sha256(ruta.read_bytes()).hexdigest(),
            'formato': 'TorchScript', 'imagenes': len(filas), 'error_abs_max': error,
            'mascaras_iguales': bool(iguales), 'diferencia_dice': dice_b - dice_a,
            'aprueba': bool(error < 1e-4 and iguales), 'segundos_verificacion': time.perf_counter() - inicio,
            'uso': 'CPU con PyTorch; no habilitado en el dashboard ONNX ni probado en Render'}
