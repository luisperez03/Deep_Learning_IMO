# Puntuación definitiva de las candidatas

**Generado por `puntuar_candidatos.py`.** Rúbrica de `PROTOCOLO_BUSQUEDA.md` con sus seis componentes. Evidencia de cada nivel: `FICHAS_COTEJADAS.md` (fragmento y página del PDF). Citas de OpenAlex consultadas el 2026-11-22 (máximo entre las obras de cada artículo; detalle en `bibliometria_candidatos.json`). Las reglas operativas de cada nivel están en el encabezado del script. Sustituye a la tabla de `PUNTUACION_Y_EXCLUSIONES.md`, que no tenía citación ni cotejo con el PDF.

| Candidata | Familia | Pert. (25) | Repr. (25) | Evid. (20) | Viab. (10) | Rec. (10) | Citas/año | Citación (10) | **Total** | Sin citación | Sin viabilidad | Estado |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SegFormer-B0 | Transformer | 18,75 | 25,00 | 10,00 | 10,00 | 5,00 | 1975,5 | 8,33 | **77,08** | 76,39 | 74,54 | seleccionada |
| UNet++ | CNN | 18,75 | 18,75 | 20,00 | 10,00 | 2,50 | 742,7 | 3,89 | **73,89** | 77,78 | 70,99 | seleccionada |
| DeepLabv3+ | CNN | 18,75 | 25,00 | 10,00 | 10,00 | 0,00 | 2692,2 | 9,44 | **73,19** | 70,83 | 70,22 | seleccionada |
| TransUNet | Híbrido CNN-Transformer | 18,75 | 18,75 | 5,00 | 10,00 | 5,00 | 1718,5 | 7,22 | **64,72** | 63,89 | 60,80 | seleccionada |
| SegNeXt-T | CNN | 12,50 | 25,00 | 10,00 | 10,00 | 5,00 | 344,6 | 1,67 | **64,17** | 69,44 | 60,19 | seleccionada |
| PIDNet-S | CNN | 12,50 | 18,75 | 10,00 | 10,00 | 7,50 | 276,8 | 0,56 | **59,31** | 65,28 | 54,78 | seleccionada |
| Attention U-Net | CNN | 12,50 | 18,75 | 15,00 | 7,50 | 0,00 | 1391,4 | 5,00 | **58,75** | 59,72 | 56,94 | excluida |
| VM-UNet | SSM | 12,50 | 18,75 | 5,00 | 10,00 | 7,50 | 495,3 | 2,78 | **56,53** | 59,72 | 51,70 | seleccionada |
| Swin-Unet | Transformer | 12,50 | 18,75 | 5,00 | 7,50 | 5,00 | 1671,0 | 6,11 | **54,86** | 54,17 | 52,62 | excluida |

"Sin citación" y "sin viabilidad" reescalan a 0-100 los 90 puntos restantes (sensibilidad que pide el protocolo y control del sesgo a favor de las redes ya ejecutadas).

## Selección óptima con la restricción de cuatro familias

Siete de nueve, con al menos una CNN, un Transformer, un híbrido y un SSM; desempate por reproducibilidad.

| Criterio | Conjunto óptimo | ¿Coincide con la selección operativa? | Cambio |
|---|---|---|---|
| Total | DeepLabv3+, PIDNet-S, SegFormer-B0, SegNeXt-T, TransUNet, UNet++, VM-UNet | sí | — |
| Sin citación | DeepLabv3+, PIDNet-S, SegFormer-B0, SegNeXt-T, TransUNet, UNet++, VM-UNet | sí | — |
| Sin viabilidad | Attention U-Net, DeepLabv3+, SegFormer-B0, SegNeXt-T, TransUNet, UNet++, VM-UNet | no | entra Attention U-Net; sale PIDNet-S |

## Justificación por componente

**SegFormer-B0** (NeurIPS, CORE A* (ICORE2026); licencia del código: NOASSERTION; 11853 citas en OpenAlex).

- Pertinencia 3/4: Robustez a 16 corrupciones (Cityscapes-C: ruido, desenfoque, clima) frente a DeepLabv3+ y otros.
- Reproducibilidad 4/4: Código y pesos entrenados; además, la réplica de evaluación de este trabajo reprodujo 37,43 mIoU con el checkpoint oficial.
- Evidencia 2/4: Tablas de validación como reporte final; test de Cityscapes con val fusionado a train. Sin repeticiones.
- Viabilidad 4/4: Entrenada en el benchmark, 5 semillas.
- Recencia 2/4: NeurIPS 2021.

**UNet++** (IEEE TMI (revista): Q1 en SJR 2020 en sus cuatro categorías (Q1 también en 2025, SJR 2,501); licencia del código: NOASSERTION; 5199 citas en OpenAlex).

- Pertinencia 3/4: Análisis estratificado por tamaño de tumor en siete grupos con prueba t (Fig. 7).
- Reproducibilidad 3/4: Código de autores; el README no publica pesos entrenados.
- Evidencia 4/4: 20 ensayos independientes, media ± DE, prueba t contra U-Net e IC95 (Fig. 4); parada temprana en validación.
- Viabilidad 4/4: Entrenada en el benchmark, 5 semillas.
- Recencia 1/4: IEEE TMI 2020.

**DeepLabv3+** (ECCV, CORE A* (ICORE2026); licencia del código: NOASSERTION; 24230 citas en OpenAlex).

- Pertinencia 3/4: Experimento trimap: mIoU en bandas alrededor del borde; la ganancia del decoder crece al estrechar la banda (ficha: dificultad compartida).
- Reproducibilidad 4/4: Código oficial en tensorflow/models con model_zoo.md de checkpoints entrenados.
- Evidencia 2/4: Ablaciones en val de VOC; la mejor variante se reentrena para test. Sin repeticiones.
- Viabilidad 4/4: Entrenada en el benchmark (SMP, ResNet-18), 5 semillas dentro del presupuesto.
- Recencia 0/4: 2018; excepción fundacional documentada en el protocolo.

**TransUNet** (Preprint arXiv; no aplica; licencia del código: Apache-2.0; 10311 citas en OpenAlex).

- Pertinencia 3/4: Ablación de conexiones con ganancia desagregada en órganos pequeños (aorta, vesícula).
- Reproducibilidad 3/4: Código de autores (Apache-2.0); solo pesos de ViT de ImageNet, no del modelo entrenado.
- Evidencia 1/4: Synapse sin conjunto de selección aparte: las configuraciones se eligen sobre los 12 casos que se reportan.
- Viabilidad 4/4: Entrenada en el benchmark, 5 semillas.
- Recencia 2/4: Preprint 2021.

**SegNeXt-T** (NeurIPS, CORE A* (ICORE2026); licencia del código: Apache-2.0; 1723 citas en OpenAlex).

- Pertinencia 2/4: Escala como motivación del diseño multiescala; solo figura cualitativa de detalles.
- Reproducibilidad 4/4: Código de autores (Apache-2.0) con tabla de modelos entrenados y pesos de ImageNet.
- Evidencia 2/4: Comparaciones sobre validación de ADE20K y Cityscapes como reporte final. Sin repeticiones.
- Viabilidad 4/4: Entrenada en el benchmark, 5 semillas.
- Recencia 2/4: NeurIPS 2022.

**PIDNet-S** (CVPR, CORE A* (ICORE2026); licencia del código: MIT; 1107 citas en OpenAlex).

- Pertinencia 2/4: Bordes y objetos pequeños como motivación y en figuras cualitativas; la ablación de pérdida de frontera se mide en mIoU global, sin métrica de borde.
- Reproducibilidad 3/4: Código de autores (MIT); el autor declara rotos los enlaces de pesos y la alternativa no se comprobó.
- Evidencia 2/4: Ablaciones en val de Cityscapes; test tras reentrenar con train+val. Sin repeticiones.
- Viabilidad 4/4: Entrenada en el benchmark, 5 semillas.
- Recencia 3/4: CVPR 2023.

**Attention U-Net** (MIDL 2018; sin rango CORE; licencia del código: MIT; 12523 citas en OpenAlex).

- Pertinencia 2/4: Páncreas (objeto pequeño y variable) como motivación; resultados por órgano frente a U-Net sin análisis dedicado.
- Reproducibilidad 3/4: Código de autores (MIT); sin pesos entrenados en el README.
- Evidencia 3/4: Validación cruzada de 5 particiones en CT-82, valores p frente a U-Net y evaluación externa CT-150 → CT-82; sin conjunto de validación.
- Viabilidad 3/4: Configuración pública plausible; sin piloto.
- Recencia 0/4: 2018; sin excepción fundacional.

**VM-UNet** (ACM TOMM (revista, 2025): Q1 en SJR 2025 en Computer Networks and Communications y en Hardware and Architecture (SJR 0,925); licencia del código: Apache-2.0; 1486 citas en OpenAlex).

- Pertinencia 2/4: Objetos pequeños y bordes complejos solo en figuras cualitativas.
- Reproducibilidad 3/4: Código de autores (Apache-2.0); el README no publica pesos del modelo entrenado.
- Evidencia 1/4: Partición train/test sin validación; el dropout se elige comparando sobre el mismo test.
- Viabilidad 4/4: Entrenada en el benchmark, 5 semillas.
- Recencia 3/4: Preprint 2024 (versión editorial ACM TOMM 2025).

**Swin-Unet** (Taller de ECCV 2022; el rango de ECCV no se extiende a sus talleres; licencia del código: sin licencia declarada; 8355 citas en OpenAlex).

- Pertinencia 2/4: Bordes afirmados con HD y figura cualitativa; sin análisis dedicado.
- Reproducibilidad 3/4: Código de autores y pesos de Swin-T de ImageNet; el repositorio no declara licencia.
- Evidencia 1/4: Ablaciones sobre el test de Synapse y configuración elegida con esos resultados.
- Viabilidad 3/4: Configuración pública plausible (entrada 224); sin piloto.
- Recencia 2/4: ECCV Workshops 2022.

## Cuartiles

El protocolo pide fuente, categoría y año del cuartil y prohíbe asignarlo a conferencias. Cinco candidatas son de conferencia (rango CORE en la tabla de justificación, consultado en portal.core.edu.au, ICORE2026), una es preprint y una de taller. Las dos de revista son Q1. Fuente: SCImago Journal Rank, datos de Scopus de marzo de 2026, leído a mano el 22/11/2026 (la página exige una verificación de navegador que no se automatiza). Año del cuartil: el de publicación de la versión citada.

| Revista | Candidata | Año | Cuartil y categoría | Año más reciente (2025) |
|---|---|---|---|---|
| IEEE Transactions on Medical Imaging (ISSN 0278-0062) | UNet++ | 2020 | Q1 en Computer Science Applications, Electrical and Electronic Engineering, Radiological and Ultrasound Technology y Software | Q1, SJR 2,501; puesto 38 de 1.030 en Computer Science Applications |
| ACM Transactions on Multimedia Computing, Communications and Applications (ISSN 1551-6857) | VM-UNet | 2025 | Q1 en Computer Networks and Communications y en Hardware and Architecture | Q1, SJR 0,925; puesto 91 de 564 en Computer Networks and Communications |

El cuartil no entra en la rúbrica: el total no depende de él.
