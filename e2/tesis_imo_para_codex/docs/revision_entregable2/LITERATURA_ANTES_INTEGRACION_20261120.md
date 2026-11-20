# Estado del arte y elección de modelos

La revisión responde a dos objetivos fundamentales en el marco del proyecto de investigación:
1. Analizar sistemáticamente la literatura científica previa sobre localización y segmentación de rótulos de mercancías peligrosas (HAZMAT/IMDG).
2. Justificar y seleccionar, bajo un protocolo formal PRISMA 2020 y criterios cuantitativos de viabilidad, un benchmark representativo de **diez modelos de segmentación semántica binaria** (las 3 líneas base del Entregable 1 más 7 candidatas adicionales), abarcando **cuatro familias arquitectónicas distintas**.

---

## 1. Fuentes de aplicación y antecedentes directos en HAZMAT (RQ-L1)

La revisión del dominio de aplicación se fundamenta en las fuentes indexadas y la matriz de antecedentes del proyecto. Se distingue rigurosamente qué tarea resuelve cada estudio (detección mediante cajas delimitadoras, segmentación mediante máscaras o clasificación) y sobre qué unidad se evalúa:

| Estudio | Tarea evaluada | Método y datos | Aporte y límite metodológico |
|---|---|---|---|
| **Sharifi, Zibaei y Rezaei (2021)**, *Machine Learning with Applications* [DOI: 10.1016/j.mlwa.2021.100104](https://doi.org/10.1016/j.mlwa.2021.100104) | Detección y aislamiento de señales HAZMAT | Pipeline **DeepHAZMAT**: YOLOv3-tiny + GrabCut adaptativo + refinamiento morfológico (convex hull). 1.685 fotografías y 13 clases. | **Antecedente directo principal.** Reporta 87,65 % de IoU promedio multiclase. No es un segmentador neuronal end-to-end; combina detección neuronal con segmentación algorítmica clásica. |
| **Brylka, Bierwirth y Schwanecke (2021)**, *HafenCity Universität Hamburg* [DOI: 10.15480/882.3959](https://doi.org/10.15480/882.3959) | Detección de etiquetas y lectura de códigos | YOLOv4-tiny para rótulos; dos U-Net intervienen en el módulo secundario de códigos de barras. Precisión 97,6 %, recall 38,5 % por imagen. | Contexto portuario/logístico real. Ilustra la brecha entre detección por secuencia (96,2 %) y recall por imagen fija. U-Net no segmenta el pictograma de peligro en este estudio. |
| **Emami (2026)**, *Tesis de Maestría, Univ. of Saskatchewan* [Handle: 10388/19451](https://hdl.handle.net/10388/19451) | Localización y clasificación | Sistema multietapa con mAP@50 de 96,2 % en la fase de localización. | Fundamenta la dificultad del contexto industrial exterior. Evalúa detección de cajas rectangulares, no máscaras densas pixel a pixel. |
| **Gaweesh, Khan y Ahmed (2021)**, *Transportation Research Record* [DOI: 10.1177/03611981211026653](https://doi.org/10.1177/03611981211026653) | Reconocimiento en transporte | AlexNet para clasificación de rótulos en video de carretera (exactitud 95 %). | Antecedente en transporte terrestre; no aporta segmentación espacial de contornos. |

---

## 2. Búsqueda sistemática y flujo PRISMA 2020

Para la selección objetiva de las nuevas arquitecturas de segmentación (RQ-L2) y mecanismos transferibles (RQ-L3), se ejecutó una búsqueda sistemática en **Scopus** y **Web of Science Core Collection** cubriendo la ventana 2019–2026:

### 2.1 Conteos y conciliación
- **Registros brutos importados:** 9.646 registros (6.399 Scopus, 3.247 WoS) distribuidos en 13 exportaciones completas (consultas Q0 a Q4). Las consultas transversales Q5 y Q6 fueron omitidas por fallo de sesión institucional y decisión del investigador.
- **Duplicados técnicos consolidados:** 3.321 registros.
- **Registros únicos analizados:** 6.325 registros.
- **Conflictos DOI resueltos documentalmente:** 3 ambigüedades aclaradas con registros oficiales de Crossref (*MCV-UNet*, *Multi-level spatial attention* y *Chinese Journal of Lasers OCT inpainting*).

### 2.2 Diagrama de flujo del cribado

```text
[ IDENTIFICACIÓN ]
  Registros indexados recuperados: Scopus (n = 6.399) + WoS (n = 3.247)
  Total importado: n = 9.646
       │
       ▼
  Duplicados conciliados: n = 3.321  ──►  Registros únicos: n = 6.325
       │
       ▼
[ CRIBADO POR TÍTULO Y RESUMEN ]
  Total registros cribados: n = 6.325
       ├─ Excluidos: n = 2.052 (Dominio no transferible: 1.732; Solo detección: 248; Tipo no experimental: 72)
       ├─ Retenidos directos: n = 2.186
       │    ├─ Dominio HAZMAT / rótulos (RQ-L1): n = 20
       │    ├─ Arquitecturas candidatas (RQ-L2): n = 507
       │    └─ Mecanismos frontera (RQ-L3):      n = 1.659
       └─ Zona de duda (inspección dirigida): n = 2.087
       │
       ▼
[ ELEGIBILIDAD Y EXTRACCIÓN ]
  Auditoría técnica documental a texto completo: n = 7 arquitecturas clave
       │
       ▼
[ INCLUSIÓN EN BENCHMARK ]
  10 modelos consolidados (3 baselines E1 + 7 candidatos seleccionados)
  Paper guía formal: SegFormer-B0 (NeurIPS 2021)
```

---

## 3. Justificación de las 4 familias arquitectónicas (10 modelos)

El diseño experimental ampliado evita comparar modelos pertenecientes exclusivamente a una sola familia. Se cubren de manera controlada cuatro paradigmas matemáticos de extracción y refinamiento espacial:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        COBERTURA DE 4 FAMILIAS                         │
├─────────────────────┬──────────────────────────────────────────────────┤
│ Familia             │ Modelos en el Benchmark                          │
├─────────────────────┼──────────────────────────────────────────────────┤
│ 1. CNN Clásica y    │ • FCN-8s (Línea base E1)                         │
│    Multirrama       │ • SegNet (Línea base E1)                         │
│                     │ • U-Net (Línea base E1)                          │
│                     │ • DeepLabv3+ (ResNet-18 + ASPP)                  │
│                     │ • UNet++ (ResNet-18 + Skip connections densos)   │
│                     │ • PIDNet-S (Tres ramas: P, I, D de borde)        │
│                     │ • SegNeXt-T (Encoder MSCAN + LightHamHead)       │
├─────────────────────┼──────────────────────────────────────────────────┤
│ 2. Transformer Puro │ • SegFormer-B0 (MiT-B0 jerárquico + MLP decoder) │
├─────────────────────┼──────────────────────────────────────────────────┤
│ 3. Híbrido CNN-     │ • TransUNet 2D (ResNet-50 + ViT-B/16 + U-Net dec)│
│    Transformer      │                                                  │
├─────────────────────┼──────────────────────────────────────────────────┤
│ 4. State Space      │ • VM-UNet (VSSM bidireccional SSM)               │
│    Models (SSM)     │                                                  │
└─────────────────────┴──────────────────────────────────────────────────┘
```

---

## 4. Auditoría técnica de las siete redes adicionales

Para cada uno de los siete nuevos modelos se realizó una extracción metodológica exhaustiva sobre sus fuentes primarias y repositorios oficiales:

### 1. DeepLabv3+ (Chen et al., ECCV 2018)
- **Familia:** CNN con pirámide espacial de convoluciones dilatadas (ASPP) y módulo de decodificación para recuperación de contornos.
- **Implementación propuesta:** `DeepLabV3Plus` de la librería *Segmentation Models PyTorch* (SMP), backbone ResNet-18, salida binaria de 1 canal de logits. Licencia MIT.
- **Consideraciones técnicas:** Desactivar pesos externos (`encoder_weights=None`) en la rama principal. Para resolución $256\times 256$, vigilar que las tasas de dilatación altas del ASPP ($\text{rate}=18$) no generen artefactos de muestreo.

### 2. UNet++ (Zhou et al., TMI 2020)
- **Familia:** CNN encoder-decoder con conexiones densas anidadas.
- **Implementación propuesta:** `UnetPlusPlus` de SMP con backbone ResNet-18. Comparador directo contra la U-Net del E1.
- **Advertencia metodológica:** La adaptación SMP no implementa la supervisión profunda de segmentación del artículo original (`aux_params` en SMP añade cabezas de clasificación, no pérdidas auxiliares espaciales). Se registra como variante adaptada sin supervisión profunda.

### 3. PIDNet-S (Xu et al., CVPR 2023)
- **Familia:** CNN de tres ramas inspirada en controladores PID.
- **Implementación propuesta:** Repositorio oficial (licencia MIT), variante S ($m=2, n=3, \text{planes}=32$).
- **Mecanismo distintivo:** Rama P (preserva detalle espacial de alta resolución), Rama I (agrega contexto global mediante PPM) y Rama D (extrae bordes finos mediante operadores diferenciales).
- **Adaptación a HAZMAT:** La rama D requiere derivar en línea mapas ground truth de bordes (Laplaciano/Sobel) calculados exclusivamente a partir de las máscaras de entrenamiento. Salida principal en el índice 1 de la lista de tensores.

### 4. SegNeXt-T (Guo et al., NeurIPS 2022)
- **Familia:** Convolucional con atención multiescala (MSCAN). No es un Transformer.
- **Implementación propuesta:** Repositorio oficial (Apache-2.0), configuración Tiny (`mscan_t` + `LightHamHead`).
- **Consideraciones técnicas:** Requiere adaptar la sincronización de batch norm (SyncBN nativa) a BatchNorm o GroupNorm monodispositivo. Desacoplar del stack MMSegmentation legacy.

### 5. SegFormer-B0 (Xie et al., NeurIPS 2021) — Paper guía
- **Familia:** Transformer jerárquico puro (MiT-B0) con decoder ligero MLP multiescala.
- **Licencia:** NVIDIA Source Code License (uso no comercial de investigación).
- **Propuesta de réplica:** Selección justificada como *paper guía* por su reproducibilidad pública y tamaño compacto (~3,8 M parámetros). Se establece la meta de evaluación de **37,4 % mIoU** en ADE20K (inferencia a escala única con checkpoint oficial). La adaptación binaria a HAZMAT constituye un experimento independiente de transferencia.
- **Adaptación a HAZMAT:** Adaptar la cabeza decodificadora para predecir 1 logit binario y optimizar con criterio combinado BCE + Dice.

### 6. TransUNet 2D (Chen et al., preprint 2021)
- **Familia:** Híbrido CNN-Transformer.
- **Implementación propuesta:** Repositorio oficial (Apache-2.0), variante R50-ViT-B/16 (12 capas Transformer, 12 cabezas, cuello de botella de 768 dimensiones).
- **Riesgo:** Alto consumo de memoria (~105 M parámetros). Retenido bajo condición de presupuesto medido con soporte de aceleración por GPU.

### 7. VM-UNet (Ruan y Xiang, preprint 2024)
- **Familia:** Modelos de Espacio de Estados Visuales (SSM / Mamba).
- **Implementación propuesta:** Repositorio oficial (Apache-2.0), arquitectura VSSM simétrica.
- **Condición operativa:** Sus extensiones de hardware (`causal_conv1d`, `mamba_ssm`, Triton) requieren compilación en Linux/CUDA. Permanece en el benchmark bajo la condición explícita de disponer de dicho entorno, preservando la representación de la familia SSM frente a una sustitución apresurada por otra CNN.

---

## 5. Puntuación cuantitativa y descarte de alternativas

La aplicación de la rúbrica metodológica (Pertinencia 25, Reproducibilidad 25, Evidencia 20, Viabilidad 10, Recencia 10) arrojó las siguientes calificaciones normalizadas (0–100):
- **PIDNet-S:** 88,89 pts (máxima pertinencia por modelado explícito de bordes).
- **SegFormer-B0:** 79,17 pts (alta reproducibilidad, paper guía).
- **UNet++:** 76,39 pts (excelente comparabilidad controlada con U-Net).
- **DeepLabv3+:** 73,61 pts (estándar de ASPP; penalizado únicamente en recencia).
- **SegNeXt-T:** 69,44 pts (sólido balance atención convolucional).
- **VM-UNet:** 63,89 pts (alta pertinencia conceptual; viabilidad condicionada a Linux/CUDA).
- **TransUNet 2D:** 61,11 pts (alta pertinencia híbrida; costo computacional elevado).

**Alternativas excluidas justificadas:**
- *Attention U-Net:* Descartada para evitar redundancia dentro de la familia CNN sin aportar un paradigma matemático nuevo.
- *SAM / SAM-2:* Descartados por requerir prompts externos (puntos/cajas), lo cual introduce disparidad frente a modelos totalmente autónomos o riesgo de fuga de datos si se usa la caja real como prompt.
- *YOLOv8-seg:* Descartado al formularse como segmentación de instancias vinculada a detección de cajas, cuya optimización distorsiona la comparación directa contra pérdidas densas pixel a pixel.

---

## 6. Síntesis y directrices para el diseño experimental

La revisión sistemática fundamenta que:
1. Las líneas base del Entregable 1 (**FCN-8s**, **SegNet**, **U-Net**) representan la base convolucional clásica contra la cual contrastar los nuevos paradigmas.
2. La comparación principal del Entregable 2 debe ejecutarse con **inicialización aleatoria en las 10 redes**, bajo exactamente las mismas particiones agrupadas por similitud, resolución $256\times 256$, semilla de datos y exclusión estricta de padding.
3. El preentrenamiento en ImageNet se evaluará como una rama experimental secundaria e independiente, reportando con rigor los datos externos utilizados para no confundir la capacidad intrínseca de una arquitectura con el volumen de datos visto previamente.
