# Resultados originales de las siete redes nuevas

**Cotejo del 20/11/2026.** Cada cifra se leyó en el PDF de arXiv de la versión indicada y se comprobó con `cotejar_resultados_originales.py` (pdftotext): **21 de 21 afirmaciones confirmadas en la página indicada** (`cotejo_resultados_originales.json`). "p." es la página del PDF de arXiv; "T." es la tabla del artículo. Las cifras de DeepLabv3+ coinciden con un cotejo independiente previo sobre la misma variante (Xception en VOC 2012 y Cityscapes). Para UNet++ se usa la versión de IEEE TMI 2020 (arXiv 1912.05074), que es la que cita el benchmark, y no la de DLMIA 2018.

Son cifras de **los artículos**, con sus datasets, resoluciones y preentrenamientos. No son comparables entre sí ni con el Dice de HAZMAT-13, y sus costos no son los de nuestras adaptaciones (estos, medidos, están en `libro/modelos_e2.md`).

## Tabla

| Red (versión) | Dataset y métrica | Resultado de cabecera | Varianza | Costo reportado |
|---|---|---|---|---|
| DeepLabv3+ (1802.02611v3) | PASCAL VOC 2012 test y Cityscapes test; mIoU | VOC: 87,8 con Xception; 89,0 con preentrenamiento JFT-300M (T. 6, p. 13). Cityscapes: 82,1 (T. 7b, p. 14) | No reporta | No reporta parámetros. 54,17 B Multiply-Adds con convolución separable y OS 16 en VOC val (T. 5, p. 12) |
| UNet++ (1912.05074v2, TMI) | EM, Cell, Nuclei, Brain Tumor, Liver (2D) y Lung Nodule (3D); IoU | Sin supervisión profunda: 88,92 / 91,03 / 92,44 / 90,86 / 82,51, contra 88,30 / 88,73 / 90,57 / 89,21 / 79,90 de U-Net (T. IV, p. 6) | **Sí**: media ± DE de **20 corridas independientes**, con prueba t de dos muestras contra U-Net (T. IV, p. 6). Ej.: Nuclei 92,44 ± 1,20 | 9,0 M parámetros (U-Net 7,8 M) (T. IV, p. 6). No reporta FLOPs |
| PIDNet-S (2206.02066v3) | Cityscapes val y test; CamVid test; mIoU y FPS | Cityscapes: **val 78,8, test 78,6**, 93,2 FPS en RTX 3090 a 2048 × 1024 (T. 6, p. 8). CamVid: 80,1 a 153,7 FPS, con preentrenamiento en Cityscapes (T. 5, p. 7) | No reporta | 7,6 M parámetros; 47,6 GFLOPs a resolución completa (T. 6, p. 8) |
| SegNeXt-T (2209.08575v1) | ADE20K, Cityscapes, COCO-Stuff; mIoU escala única / multiescala | ADE20K 41,1 / 42,2; Cityscapes 79,8 / 81,4; COCO-Stuff 38,7 / 39,1 (T. 9, p. 9). Cityscapes en tiempo real, 768 × 1536: 78,0 (T. 11, p. 9) | No reporta | 4,3 M parámetros; 6,6 GFLOPs a 512 × 512 y 50,5 a 2048 × 1024 (T. 9, p. 9) |
| SegFormer-B0 (2105.15203v3) | ADE20K, Cityscapes, COCO-Stuff; mIoU escala única / multiescala | ADE20K 37,4 / 38,0; Cityscapes 76,2 / 78,1; COCO-Stuff 35,6 (T. 1a, p. 7). Cityscapes con lado corto 512: 71,9 a 47,6 FPS (T. 2, p. 7) | No reporta | 3,8 M parámetros (3,4 codificador + 0,4 decodificador); 8,4 GFLOPs en ADE20K y 125,5 en Cityscapes (T. 1a, p. 7) |
| TransUNet (2102.04306v1) | Synapse (8 órganos en TC) y ACDC (RM cardiaca); DSC y HD95 | Synapse: DSC 77,48, HD95 31,69 mm (T. 1, p. 6). ACDC: DSC 89,71 (T. 5, p. 10) | No reporta | **No reporta** parámetros ni FLOPs en esta versión |
| VM-UNet (2402.02491v2) | ISIC17, ISIC18, Synapse; mIoU, DSC, Acc, Spe, Sen, HD95 | ISIC17: mIoU 80,23, DSC 89,03; ISIC18: mIoU 81,35, DSC 89,71 (T. I, p. 5). Synapse: DSC 81,08, HD95 19,21 (T. II, p. 6). Sin preentrenamiento, ISIC17 DSC 87,38 (T. III, p. 6) | No reporta | Configuración base {2,2,2,2-2,2,2,1}: 27,43 M parámetros y 4,11 GFLOPs (T. V, p. 6) |

## Lectura

- **Varianza.** Solo uno de los siete (UNet++ en TMI) repite corridas: 20 por configuración, con desviación estándar y prueba t. Los otros seis dan una cifra única sin semilla ni intervalo.
- **Costo.** Cinco reportan parámetros y cinco reportan FLOPs u operaciones, cada uno a su propia resolución de entrada. TransUNet v1 no reporta ninguno de los dos.
- **Datasets.** Ninguno se evalúa en rótulos ni señales de mercancía peligrosa: cuatro son de escenas urbanas o generales (VOC, Cityscapes, ADE20K, CamVid) y tres son médicos (UNet++, TransUNet, VM-UNet).

## Diferencias con `FICHAS_EXTRACCION_7REDES.md`

Las fichas del 18/11 tienen cifras que el PDF no respalda. Se corrigen aquí y no en las fichas (que conservan su rectificación):

| Red | Ficha del 18/11 | PDF |
|---|---|---|
| PIDNet-S | Cityscapes **val** 78,6 | 78,6 es **test**; val es 78,8 (T. 6, p. 8) |
| UNet++ | "Mejora sobre U-Net en 4-6 puntos IoU" | Sin supervisión profunda, de 0,62 (EM) a 2,61 (Liver) puntos (T. IV, p. 6) |
| SegNeXt-T | 4,3 M de codificador, ~4,5 M en total | 4,3 M en total (T. 9, p. 9) |
| TransUNet | ~105 M parámetros | La versión citada no reporta parámetros |
| VM-UNet | ISIC 2017 Dice 90,98, IoU 83,78; profundidades [2,2,9,2]; partición 2000/150/600 | v2: DSC 89,03, mIoU 80,23; base {2,2,2,2-2,2,2,1}; [2,2,9,2] es una fila de ablación (44,27 M, 7,56 GFLOPs); partición 7:3, 1.500/650 (p. 5). Las cifras de la ficha no aparecen en v2 |
| VM-UNet | DOI editorial no verificable (403) | `10.1145/3767748` registrado en Crossref: ACM TOMM, 2025 |

**VM-UNet implementado.** `scripts/modelos_e2.py` usa profundidades [2, 2, 9, 2] / [2, 9, 2, 2] y `libro/modelos_e2.md` mide 44.273.833 parámetros. Eso coincide con la fila {2,2,9,2-2,9,2,2} de la ablación de v2 (44,27 M, 7,56 GFLOPs; ISIC17 mIoU 79,67, DSC 88,68; T. V, p. 6), no con la configuración base de v2 (27,43 M). Al citar el resultado original de la variante que se entrena, la cifra comparable es la de esa fila.
