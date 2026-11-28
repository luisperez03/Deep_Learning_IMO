"""Arquitecturas compactas E2; copia fiel de definiciones, sin importar runners.
Procedencia y hashes en config/fuentes_compactas.json. No es el mecanismo original E3.
"""
import torch
from torch import nn
from torch.nn import functional as F

def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(b, b, 3, padding=1), nn.ReLU())

def _drop(activo, p):
    return nn.Dropout(p) if activo else nn.Identity()

class FCNBusqueda(nn.Module):
    """SmallFCN con anchos (w1, w2, w3) y dropout tras cada bloque. Mismo orden de creación de capas."""

    def __init__(self, w, b, p=0.2):
        super().__init__()
        w1, w2, w3 = w
        self.e1, self.e2, self.e3 = block(3, w1), block(w1, w2), block(w2, w3)
        self.head = nn.Conv2d(w3, 1, 1)
        self.r1, self.r2, self.r3 = (_drop(x, p) for x in b)

    def forward(self, x):
        a = self.r1(self.e1(x))
        c = self.r2(self.e2(F.max_pool2d(a, 2)))
        c = self.r3(self.e3(F.max_pool2d(c, 2)))
        return F.interpolate(self.head(c), size=x.shape[-2:], mode='bilinear', align_corners=False)

class UNetBusqueda(nn.Module):
    """SmallUNet con anchos (w1, w2, w3); el decodificador espeja los anchos y el dropout."""

    def __init__(self, w, b, p=0.2):
        super().__init__()
        w1, w2, w3 = w
        self.e1, self.e2, self.e3 = block(3, w1), block(w1, w2), block(w2, w3)
        self.d2, self.d1 = block(w3 + w2, w2), block(w2 + w1, w1)
        self.head = nn.Conv2d(w1, 1, 1)
        self.r1, self.r2, self.r3 = (_drop(x, p) for x in b)
        self.s2, self.s1 = _drop(b[1], p), _drop(b[0], p)  # espejos de los niveles 2 y 1

    def forward(self, x):
        a = self.r1(self.e1(x))
        m = self.r2(self.e2(F.max_pool2d(a, 2)))
        c = self.r3(self.e3(F.max_pool2d(m, 2)))
        d = self.s2(self.d2(torch.cat([F.interpolate(c, size=m.shape[-2:], mode='nearest'), m], 1)))
        d = self.s1(self.d1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='nearest'), a], 1)))
        return self.head(d)

class SegNetBusqueda(nn.Module):
    """SmallSegNet con anchos (w1, w2, w3); reducción w3 -> w2 y decodificador espejo con unpooling."""

    def __init__(self, w, b, p=0.2):
        super().__init__()
        w1, w2, w3 = w
        self.e1, self.e2, self.e3 = block(3, w1), block(w1, w2), block(w2, w3)
        self.reduce = nn.Conv2d(w3, w2, 1)
        self.d2, self.d1 = block(w2, w1), block(w1, w1)
        self.head = nn.Conv2d(w1, 1, 1)
        self.r1, self.r2, self.r3 = (_drop(x, p) for x in b)
        self.s2, self.s1 = _drop(b[1], p), _drop(b[0], p)

    def forward(self, x):
        a = self.r1(self.e1(x))
        p, i1 = F.max_pool2d(a, 2, return_indices=True)
        m = self.r2(self.e2(p))
        p, i2 = F.max_pool2d(m, 2, return_indices=True)
        c = self.reduce(self.r3(self.e3(p)))
        d = self.s2(self.d2(F.max_unpool2d(c, i2, 2, output_size=m.shape)))
        d = self.s1(self.d1(F.max_unpool2d(d, i1, 2, output_size=a.shape)))
        return self.head(d)

CLASES = {'fcn': FCNBusqueda, 'unet': UNetBusqueda, 'segnet': SegNetBusqueda}

def construir_red(familia, w, b, p):
    return CLASES[familia](tuple(w), tuple(b), p)
