# Costo computacional de las 10 arquitecturas

> **Trazabilidad.** Generado el 2026-11-22 con `scripts/medir_costo_e2.py` (parámetros, FLOPs, memoria, latencia) y
> `scripts/costo_total_e2.py` (costo de GPU registrado y cota de energía). Redes instanciadas con inicialización
> aleatoria (el costo no depende de los pesos). Entrada: 1 imagen RGB de 256×256, batch 1, `torch.no_grad`.
> Entorno: torch 2.14.1 CPU, Python 3.12, Windows 11, AMD Ryzen 5 PRO 6650U (6 núcleos, 12 hilos lógicos), 14,8 GiB de RAM.
> Por qué importa: el costo es el contrapeso del Dice en la comparación de redes (fila 44) y el gasto total es la base
> de cualquier afirmación de costo o energía (fila 45).

## Qué se midió y cómo

- **Parámetros:** `sum(p.numel())` de la red instanciada (coincide con el `parameters` de `analisis.json`).
- **FLOPs:** `torch.utils.flop_counter.FlopCounterMode` en una pasada de inferencia (1 MAC = 2 FLOPs). Cuenta convoluciones,
  matmul, bmm y atención; **no cuenta** normalizaciones, activaciones ni interpolaciones, así que es una cota inferior
  homogénea, no un conteo exhaustivo.
- **Memoria:** cada red corre en un proceso nuevo. Se reportan el pico del proceso (`PeakWorkingSetSize`, incluye torch y
  la construcción del modelo) y el *incremento* del conjunto de trabajo en la primera inferencia, muestreado cada 1 ms
  sobre el modelo ya cargado. Es memoria de CPU del proceso, no de activaciones exactas: una cota inferior del pico de
  activaciones.
- **Latencia:** 5 pasadas de calentamiento y 30 repeticiones; mediana e IQR. Dos configuraciones de hilos fijadas con
  `torch.set_num_threads`: 1 y 6 (los núcleos físicos). Los 30 tiempos de cada corrida están en los JSON.

## Resultados (CPU, 256×256, batch 1)

| Modelo | Parámetros | GFLOPs | Latencia 1 hilo, mediana (IQR) ms | Latencia 6 hilos, mediana (IQR) ms | Pico proceso MiB | Δ inferencia MiB |
|---|---|---|---|---|---|---|
| FCN | 72.145 | 1,27 | 50,9 (3,4) | 22,5 (7,6) | 290 | 14 |
| SegNet | 85.761 | 2,11 | 73,2 (5,0) | 24,8 (2,7) | 313 | 67 |
| U-Net | 118.273 | 3,68 | 127,9 (9,3) | 39,3 (3,7) | 319 | 44 |
| DeepLabv3+ | 12.329.297 | 9,12 | 277,2 (19,3) | 86,0 (6,5) | 403 | 38 |
| UNet++ | 15.970.449 | 31,94 | 812,3 (34,1) | 320,9 (32,8) | 495 | 101 |
| PIDNet-S | 7.716.291 | 3,13 | 89,8 (8,8) | 53,2 (5,7) | 323 | 15 |
| SegNeXt-T | 4.225.121 | 3,13 | 202,1 (16,5) | 129,1 (27,2) | 364 | 27 |
| SegFormer-B0 | 3.714.401 | 3,49 | 215,9 (15,9) | 115,2 (68,2) | 392 | 57 |
| TransUNet 2D | 105.322.001 | 76,77 | 2057,8 (95,9) | 737,6 (80,8) | 797 | 96 |
| VM-UNet | 44.273.833 (solo conteo) | — | — | — | — | — |

Tabla completa con ambos hilos y todos los campos: `costo_e2.csv`, `costo_e2.md`; un JSON por red y configuración
(`<modelo>_cpu_h<hilos>.json`) con el hardware, los 30 tiempos y los FLOPs por operador.

## Lectura

- **Los FLOPs no ordenan la latencia.** PIDNet-S, SegNeXt-T y SegFormer-B0 tienen casi los mismos FLOPs (3,1–3,5 G) pero
  con 1 hilo tardan 90, 202 y 216 ms: los operadores sin FLOPs contados (normalizaciones, atención, interpolación) y la
  intensidad aritmética pesan. UNet++ hace 32 G y tarda 812 ms; U-Net hace 3,7 G y tarda 128 ms.
- **Los parámetros tampoco.** SegFormer-B0 (3,7 M) es más lento que DeepLabv3+ (12,3 M) con 1 hilo. TransUNet tiene el
  mayor costo en las tres medidas (105 M, 77 G, 2,1 s con 1 hilo).
- **Frente al Dice:** las tres redes pequeñas (FCN, SegNet, U-Net) cuestan entre 1,3 y 3,7 G y 23–40 ms con 6 hilos.
  Cualquier comparación costo-precisión debe leerse con los intervalos del Dice, no con puntos.
- **Escalado con hilos:** 6 hilos aceleran entre 1,6× (SegNeXt-T) y 3,2× (DeepLabv3+, SegNet, U-Net); las redes con
  muchos operadores pequeños escalan peor.

## Límites (declarar tal cual)

1. **CPU, no la GPU del entrenamiento.** Entrenamiento en L4 y test en T4: estas latencias no son las de ninguna de
   las dos y no se pueden mezclar con los segundos de época. Sirven para comparar redes entre sí en un mismo hardware.
2. **Máquina compartida.** Las mediciones corrieron mientras otras tareas usaban el equipo; el IQR de varias redes con
   6 hilos es amplio (p. ej. SegFormer-B0, 68 ms). Las medianas ordenan las redes, pero una diferencia menor al IQR no es
   una diferencia.
3. **VM-UNet no corre en CPU:** exige CUDA y `mamba-ssm` y el repositorio prohíbe el reemplazo por GRU. Solo se declara su
   número de parámetros (44.273.833, de `analisis.json`). FLOPs, memoria y latencia quedan pendientes de la medición en
   Colab (`COLAB_MEDIR_COSTO.md`, mismo script con `--dispositivo cuda`). Aun medido allí, `FlopCounterMode` no cuenta el
   escaneo selectivo de Mamba (kernel propio): sus FLOPs serían una cota inferior.
4. Una sola resolución (256), batch 1 e inicialización aleatoria; la memoria excluye la de activaciones de entrenamiento.
5. Los FLOPs son de la misma herramienta para todas, pero no coinciden con los que reportan los autores de cada red (otra
   resolución, otra herramienta, otros operadores contados).

## Costo total y energía (fila 45): `costo_total_e2.md` / `.json`

Se reconstruyó solo lo que quedó registrado en el repo (horas de GPU), sin estimar lo demás:

| Concepto | GPU | Corridas | Horas | kWh (cota TDP) |
|---|---|---|---|---|
| Entrenamiento de las 50 corridas (segundos de época) | NVIDIA L4 | 50 | 16,40 | 1,18 |
| Diagnóstico: evaluación limpia | NVIDIA L4 | 50 | 0,86 | 0,06 |
| Diagnóstico: robustez | NVIDIA L4 | 50 | 0,57 | 0,04 |
| Diagnóstico: interpretabilidad | NVIDIA L4 | 15 | 0,26 | 0,02 |
| Complemento: ablación | NVIDIA L4 | 15 | 7,41 | 0,53 |
| Complemento: permutación del pipeline | NVIDIA L4 | 200 | 3,80 | 0,27 |
| Complemento: búsqueda de hiperparámetros | NVIDIA L4 | 40 | 3,88 | 0,28 |
| Complemento: sobreajuste a un lote | NVIDIA L4 | 10 | 0,12 | 0,01 |
| **Total registrado** | | | **33,30** | **2,40** |

- **Horas** = segundos que cada corrida guardó (épocas, etapas de diagnóstico), no tiempo de pared de la sesión de Colab.
  Sin registro y por tanto fuera de la suma: evaluación en test (T4), réplica de SegFormer (T4), carga y espera de las
  sesiones, extracción de Dice por imagen de validación e instalación de dependencias. El total es un mínimo.
- **Energía** = TDP publicado × horas (L4: 72 W; T4: 70 W). Es una **cota superior de la energía de la GPU** a plena
  potencia, no una medición: la GPU rara vez trabaja a TDP, y la cifra no incluye CPU, memoria, red ni refrigeración del
  centro de datos. Con esa salvedad, el trabajo registrado no pasa de unos 2,4 kWh de GPU. No se calcula huella de
  carbono: no hay base declarada para la intensidad de la red eléctrica de la región de Colab.
- El complemento se asigna a la L4 por la estimación de `estimacion_computo.json`; las corridas no guardan la GPU.
- Entrenamiento por red (5 semillas cada una): TransUNet 3,46 h, VM-UNet 2,91 h, PIDNet-S 2,29 h, SegNeXt-T 2,04 h,
  SegFormer-B0 1,60 h, DeepLabv3+ 1,03 h, UNet++ 1,00 h, FCN 0,80 h, U-Net 0,74 h, SegNet 0,53 h.

## Reproducir

```
python scripts/medir_costo_e2.py medir --modelos fcn,segnet,unet,deeplabv3plus,unetplusplus,pidnet_s,segnext_t,segformer_b0,transunet_2d,vmunet
python scripts/medir_costo_e2.py tabla
python scripts/costo_total_e2.py
```

(`ml_collections` es necesario para TransUNet; está en `requirements-e2.txt`.)
