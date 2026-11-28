"""Equivalencia de max_unpool2d para pools 2x2, stride 2, sin índices repetidos."""
import copy
import torch
from compactas_e2 import SegNetBusqueda
from torch.nn import functional as F


def unpool(x, indices, referencia):
    # Los índices de MaxPool son offsets espaciales dentro de cada canal.
    base = torch.zeros_like(referencia).flatten(2)
    return base.scatter(2, indices.flatten(2), x.flatten(2)).reshape_as(referencia)


class SegNetExportable(SegNetBusqueda):
    def forward(self, x):
        a = self.r1(self.e1(x))
        p, i1 = F.max_pool2d(a, 2, return_indices=True)
        m = self.r2(self.e2(p))
        p, i2 = F.max_pool2d(m, 2, return_indices=True)
        c = self.reduce(self.r3(self.e3(p)))
        d = self.s2(self.d2(unpool(c, i2, m)))
        d = self.s1(self.d1(unpool(d, i1, a)))
        return self.head(d)


def adaptar_segnet(modelo):
    nuevo = copy.deepcopy(modelo)
    nuevo.__class__ = SegNetExportable
    return nuevo.eval()
