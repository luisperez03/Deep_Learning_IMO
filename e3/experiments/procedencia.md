# Procedencia del mecanismo de UNet-AT

> **Trazabilidad.** Escrito el 2026-11-23. Huellas calculadas con Python (`hashlib.sha256`) sobre los archivos del
> árbol de trabajo en esa fecha, antes del congelamiento con la etiqueta `e3-protocolo`; hay que recalcularlas al
> congelar. Fuentes: `experiments/protocolo.md`, `MODELO_ORIGINAL_ESTADO.md`, `src/original_model.py`.

## Fuente transferida

La representación procede de DBNet (Liao et al., 2020, AAAI, doi:10.1609/aaai.v34i07.6812), diseñada para detección de
texto en escena. No se usa su código ni sus pesos: el mecanismo se reescribió desde cero sobre el arnés del E2.

## Qué se conserva de DBNet

- Cabeza de probabilidad del núcleo $P$, cabeza de umbral $T$ y binarización aproximada $B=\sigma(k(P-T))$ con $k=50$.
- Pérdida $5\mathcal{L}_s + 10\mathcal{L}_t + \mathcal{L}_b$ con BCE con minería de negativos difíciles 1:3, L1 en la
  banda y Dice sobre $B$.
- La regla de encogimiento de DBNet (`φ ≡ 1`) como brazo `dbnet` y su posproceso, idéntico al del E2.

## Qué se modifica

| Pieza | DBNet | UNet-AT |
|---|---|---|
| Esqueleto | ResNet con FPN, preentrenado | U-Net compacta del E2 (16, 32, 64), sin preentrenamiento |
| Cabezas | Cabeza de DBNet | Convolución 1×1 por cabeza (17 parámetros más que la U-Net compacta) |
| Encogimiento (C1) | Distancia fija por área y perímetro | Atenuada por el grosor: `φ(g) = g²/(g² + g0²)`, `g0 = 4` px |
| Expansión (C2) | Distancia fija (unclip) con bordes redondeados | Punto fijo `d = ρ(M(K', d))` con inglete propio (límite 2) y ajuste de 0,75 px |
| Inglete | Biblioteca de recorte de polígonos | Implementación propia (`inglete_poligono`), validada contra la biblioteca (IoU de 0,96 a 1,00) |
| Tarea | Detección de texto (cajas) | Segmentación binaria de rótulos |

## SHA-256 de los módulos del original

| Archivo | SHA-256 |
|---|---|
| `src/geometria_at.py` | `7dba12fbba9333b04c4ce1ae96f305cdcb7c0865337d722b8c77f3538fd4676c` |
| `src/original_model.py` | `395340738602521a528e3c14cc662e0cb3f737954d1b055a9e82c0b605fdcbff` |
| `training/entrenar_original.py` | `fde07a4f2176f5fdb1549b508605df903041e4e15bcca3951714f131f8ecb1a8` |

Verificar: `python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <archivo>`.

## Lo que falta para congelar

- Declaración de novedad con verificación de un tercero (`experiments/novedad/`).
- Recalcular las huellas al congelar el protocolo.
