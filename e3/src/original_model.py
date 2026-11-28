"""UNetAT: la U-Net compacta del benchmark E2 con el mecanismo de DBNet-AT. Modelo original del Entregable 3.

Que es. Se porta el mecanismo de DBNet (Liao et al., 2020, AAAI, "Real-time Scene Text Detection with
Differentiable Binarization") y el de DBNet-AT (entregable_3/diseno.md, seccion 3) a la U-Net compacta de E2
(modelos_base.SmallUNet: tres niveles de 16, 32 y 64 canales, sin BatchNorm). Se entrena desde cero, sin pesos
preentrenados, en el mismo pipeline de E2.

Arquitectura (la de SmallUNet, reescrita aqui, mas dos cabezas):
    e1 = bloque(3 -> 16)             resolucion completa
    e2 = bloque(16 -> 32)            1/2
    e3 = bloque(32 -> 64)            1/4   (el mapa mas profundo: su paso, 4 px, es g0 de C1)
    d2 = bloque(96 -> 32)            sube e3 y concatena e2
    d1 = bloque(48 -> 16)            sube d2 y concatena e1
    cabeza_prob, cabeza_umbral: dos cabezas sobre los 16 canales de d1 que dan los logits de P y de T.
bloque(a, b) = Conv3x3(a,b) ReLU Conv3x3(b,b) ReLU. Las capas del codificador y del decodificador se crean en
el mismo orden que en SmallUNet, asi que con la misma semilla global tienen los mismos pesos iniciales que la
U-Net de E2; las cabezas se crean al final.

Formulacion (DBNet, seccion 3 del articulo):
    P = sigmoid(z_P)                       mapa de probabilidad del nucleo
    T = sigmoid(z_T)                       mapa de umbral aprendido
    B = sigmoid(k (P - T)),  k = 50        binarizacion diferenciable (aproxima el escalon H(P - T))
Derivada de la binarizacion, la que hace entrenable al umbral:
    dB/dP =  k B (1 - B)
    dB/dT = -k B (1 - B)
La derivada es grande solo donde B ~ 0,5, es decir P ~ T: ahi el gradiente de la perdida empuja a P y a T a
separarse. Lejos del cruce B esta saturado y casi no hay gradiente.

Perdida (perdida_db), la del codigo oficial de DBNet y la de E2 (entregable_2/codigo/modelos.py, DBNet.perdida):
    L = 5 Ls + 10 Lt + Lb
    Ls = BCE de P con minado de negativos dificiles 1:3 (OHEM)
    Lt = L1 entre T y su blanco, promediada dentro de la banda Gd del mapa de umbral
    Lb = Dice entre B y el nucleo
Los pesos 5, 10 y 1 coinciden con diseno.md 3.1 ("5 Ls + 10 Lt + Lb"). El articulo escribe BCE para Lb con alfa = 1;
el codigo oficial usa Dice y pesa 5 la BCE, y E2 sigue el codigo.

Las etiquetas (nucleo, banda y blanco de T) las produce src.geometria_at.at_objetivos (C1 si variante='dbnet_at',
regla de DBNet si variante='dbnet'). El posproceso (binarizar P, filtrar, expandir con C2) es predict_mask.
"""
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import geometria_at as G

K_BINARIZACION = 50.0


def bloque(a, b):
    """Dos convoluciones 3x3 con ReLU (el bloque de SmallUNet)."""
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())


class UNetAT(nn.Module):
    """U-Net compacta con cabezas de probabilidad (P) y de umbral (T) y binarizacion B = sigmoid(k (P - T)).

    Parametros:
        k: pendiente de la binarizacion aproximada (50, como DBNet).
        cabeza: '1x1' (una convolucion 1x1 por cabeza: 17 parametros cada una, paridad de capacidad con la U-Net
            de E2) o '3x3' (conv 3x3 de 16 canales, ReLU y conv 1x1: mas capacidad en cada cabeza).
        semilla: si se da, los pesos iniciales salen de esa semilla sin tocar el generador global; si es None se
            usa el generador global (el arnes de E2 lo siembra con setup_seed antes de crear el modelo).
    forward: en entrenamiento devuelve un dict con 'prob' (P), 'thresh' (T), 'binary' (B) y 'logit_prob' (z_P, que
        la perdida necesita para la BCE estable); en eval devuelve solo P, de forma (B, 1, H, W).
    """

    def __init__(self, k=K_BINARIZACION, cabeza="1x1", semilla=None):
        super().__init__()
        if cabeza not in ("1x1", "3x3"):
            raise ValueError("cabeza debe ser '1x1' o '3x3'")
        self.k = float(k)
        self.cabeza_tipo = cabeza
        # fork_rng(devices=[]) guarda y restaura el generador de la CPU: crear el modelo con semilla no altera el global
        contexto = torch.random.fork_rng(devices=[]) if semilla is not None else nullcontext()
        with contexto:
            if semilla is not None:
                torch.manual_seed(int(semilla))
            # mismo orden de creacion que SmallUNet: e1, e2, e3, d2, d1
            self.e1, self.e2, self.e3 = bloque(3, 16), bloque(16, 32), bloque(32, 64)
            self.d2, self.d1 = bloque(96, 32), bloque(48, 16)
            self.cabeza_prob = self._cabeza(cabeza)
            self.cabeza_umbral = self._cabeza(cabeza)

    @staticmethod
    def _cabeza(tipo):
        if tipo == "1x1":
            return nn.Conv2d(16, 1, 1)
        return nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))

    # ------------------------------------------------------------------ validacion de la entrada
    @staticmethod
    def _validar(x):
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"la entrada debe ser un tensor, no {type(x).__name__}")
        if x.dim() != 4:
            raise ValueError(f"se espera (B, 3, H, W); llego {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"se esperan 3 canales; llegaron {x.shape[1]}")
        if not x.is_floating_point():
            raise ValueError(f"se espera un tensor flotante, no {x.dtype}")
        h, w = x.shape[-2:]
        if h % 4 or w % 4:
            raise ValueError(f"alto y ancho deben ser multiplos de 4 (dos reducciones a la mitad); llego {h} x {w}")

    # ------------------------------------------------------------------ red
    def _caracteristicas(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        d = self.d2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="nearest"), b], 1))
        return self.d1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode="nearest"), a], 1))

    def forward(self, x):
        self._validar(x)
        h = self._caracteristicas(x)
        zp = self.cabeza_prob(h)
        if not self.training:
            return torch.sigmoid(zp)
        p, t = torch.sigmoid(zp), torch.sigmoid(self.cabeza_umbral(h))
        return {"prob": p, "thresh": t, "binary": torch.sigmoid(self.k * (p - t)), "logit_prob": zp}

    # ------------------------------------------------------------------ inferencia
    @torch.no_grad()
    def predict_proba(self, x):
        """P (B, 1, H, W), sin importar el modo del modulo."""
        self._validar(x)
        return torch.sigmoid(self.cabeza_prob(self._caracteristicas(x)))

    @torch.no_grad()
    def predict_mask(self, x, variante="dbnet_at", valida=None):
        """Mascara binaria (B, H, W) bool: P pasa por el posproceso de la variante (src.geometria_at.VARIANTES).
        'dbnet_at' = C1 + C2 (expansion consistente en inglete); 'dbnet' = el posproceso de DBNet (r' = 1,5).
        valida: bool (B, H, W) con la zona valida (sin relleno del letterbox); por defecto, toda la imagen."""
        if variante not in G.VARIANTES:
            raise ValueError(f"variante desconocida: {variante}; opciones: {sorted(G.VARIANTES)}")
        p = self.predict_proba(x)[:, 0].cpu().numpy()
        v = np.ones(p.shape, bool) if valida is None else np.asarray(valida, bool).reshape(p.shape)
        post = G.post_de(variante)
        return np.stack([post(p[i], v[i]) for i in range(len(p))])

    def perdida(self, salida, objetivos, **pesos):
        return perdida_db(salida, objetivos, **pesos)


# ---------------------------------------------------------------------- perdida (copia de la de E2)
def bce_ohem(logits, gt, mask, razon=3):
    """BCE con minado de negativos dificiles 1:razon (E2, modelos.bce_ohem; codigo oficial de DBNet)."""
    perd = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    pos = (gt > 0.5) & (mask > 0.5)
    neg = (gt <= 0.5) & (mask > 0.5)
    n_pos = int(pos.sum())
    # sin positivos (imagen sin rotulo) se toman 100 negativos, como en las implementaciones oficiales
    n_neg = min(int(neg.sum()), razon * n_pos if n_pos else 100)
    perd_neg = torch.topk(perd[neg], n_neg).values if n_neg else perd.new_zeros(0)
    return (perd[pos].sum() + perd_neg.sum()) / (n_pos + n_neg + 1e-6)


def dice_perdida(p, t, m):
    """1 - Dice enmascarado, con p, t y m del mismo tamano (E2, modelos.dice_perdida)."""
    p, t = p * m, t * m
    return 1 - (2 * (p * t).sum() + 1e-6) / ((p * p).sum() + (t * t).sum() + 1e-6)


def perdida_db(salida, objetivos, peso_s=5.0, peso_t=10.0, peso_b=1.0):
    """L = 5 Ls + 10 Lt + Lb, la de DBNet segun el codigo oficial y E2.

    salida: dict de UNetAT en modo entrenamiento. objetivos: dict de tensores (B, 1, H, W) con las claves de
    geometria_at.at_objetivos (db_gt, db_mask, db_umbral, db_umbral_mask)."""
    ls = bce_ohem(salida["logit_prob"], objetivos["db_gt"], objetivos["db_mask"])
    lb = dice_perdida(salida["binary"], objetivos["db_gt"], objetivos["db_mask"])
    m = objetivos["db_umbral_mask"]
    lt = (torch.abs(salida["thresh"] - objetivos["db_umbral"]) * m).sum() / (m.sum() + 1e-6)
    return peso_s * ls + peso_t * lt + peso_b * lb
