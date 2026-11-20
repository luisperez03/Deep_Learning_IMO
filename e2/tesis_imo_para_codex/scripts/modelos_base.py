"""Adaptaciones pequeñas, desde cero: no réplicas de los artículos originales.

Tres niveles (16, 32, 64 canales) para limitar memoria. Cada salida es un
logit por píxel: positivo favorece etiqueta; sigmoid lo convierte en probabilidad.
No se usa BatchNorm porque el lote local será de una imagen.
"""
import torch
from torch import nn
from torch.nn import functional as F


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(b, b, 3, padding=1), nn.ReLU())


class SmallFCN(nn.Module):
    """Reconstrucción desde mapa grueso, sin saltos laterales (familia FCN)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(3, 16), nn.MaxPool2d(2),
                                      block(16, 32), nn.MaxPool2d(2), block(32, 64))
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        return F.interpolate(self.head(self.features(x)), size=x.shape[-2:],
                             mode='bilinear', align_corners=False)


class SmallUNet(nn.Module):
    """Recupera detalle concatenando características del encoder."""
    def __init__(self):
        super().__init__()
        self.e1, self.e2, self.e3 = block(3, 16), block(16, 32), block(32, 64)
        self.d2, self.d1 = block(96, 32), block(48, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        d = self.d2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='nearest'), b], 1))
        d = self.d1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='nearest'), a], 1))
        return self.head(d)


class SmallSegNet(nn.Module):
    """Reconstruye posiciones con los índices del max pooling, sin concatenar."""
    def __init__(self):
        super().__init__()
        self.e1, self.e2, self.e3 = block(3, 16), block(16, 32), block(32, 64)
        self.reduce = nn.Conv2d(64, 32, 1)
        self.d2, self.d1 = block(32, 16), block(16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        p, i1 = F.max_pool2d(a, 2, return_indices=True)
        b = self.e2(p)
        p, i2 = F.max_pool2d(b, 2, return_indices=True)
        c = self.reduce(self.e3(p))
        d = self.d2(F.max_unpool2d(c, i2, 2, output_size=b.shape))
        d = self.d1(F.max_unpool2d(d, i1, 2, output_size=a.shape))
        return self.head(d)


MODELS = {'fcn': SmallFCN, 'segnet': SmallSegNet, 'unet': SmallUNet}
