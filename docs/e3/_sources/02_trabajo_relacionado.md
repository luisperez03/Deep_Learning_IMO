(sec-trabajo-relacionado)=
# Trabajo relacionado

## Segmentación semántica

La U-Net {cite:p}`ronneberger2015unet` es el esqueleto de codificador-decodificador con conexiones de salto que usa el
modelo original (versión compacta de tres niveles). SegFormer {cite:p}`xie2021segformer` es el paper guía de la
familia de transformadores del benchmark (variante B0) y la referencia más cercana al problema de objetos pequeños.
Las diez redes comparadas son FCN, SegNet y U-Net compactas, DeepLabv3+ R18, UNet++ R18, PIDNet-S, SegNeXt-T,
SegFormer-B0, TransUNet y VM-UNet (`experiments/protocolo.md`, §5).

## Representaciones de núcleo encogido

PSENet {cite:p}`wang2019psenet` y DBNet {cite:p}`liao2020dbnet` detectan texto en escena prediciendo un núcleo
encogido de cada región y expandiéndolo en inferencia. DBNet agrega una cabeza de umbral y una binarización
aproximada diferenciable. UNet-AT toma esa representación como fuente transferida (`experiments/procedencia.md`).

## Datos

HAZMAT-13 {cite:p}`sharifi2021hazmat` es el conjunto de rótulos de materiales peligrosos usado aquí
(`data/DATA_STATEMENT.md`).

## Estadística de la comparación

Wilcoxon {cite:p}`wilcoxon1945`, Holm {cite:p}`holm1979`, δ de Cliff {cite:p}`cliff1993` y la comparación de varios
clasificadores {cite:p}`demsar2006` fundamentan las pruebas del protocolo.

```{admonition} Pendiente
:class: warning
Falta la revisión de novedad con verificación externa ({ref}`sec-novedad`); hasta cerrarla, este capítulo no
afirma ausencia de trabajo previo sobre el mecanismo.
```
