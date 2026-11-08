# Estado del arte y elección de modelos

La revisión responde a dos preguntas. ¿Qué han medido otros estudios sobre rótulos de mercancías peligrosas? ¿Cómo justificar las tres redes del experimento de segmentación binaria? Para la primera usamos las cuatro fuentes aprobadas en la matriz bibliográfica del proyecto. Para la segunda acudimos a los artículos originales de FCN, SegNet y U-Net.

## Fuentes de aplicación y criterios de lectura

La {download}`matriz bibliográfica del proyecto <../docs/Estado_del_arte_IMO_matriz_4_fuentes.xlsx>`, fechada el 7 de noviembre de 2026, identifica cada estudio por autor, tarea, método, datos, métricas, alcance de la lectura y aporte a la tesis. Para interpretar un resultado comprobamos **qué se predice** (caja, máscara, clase o texto) y **sobre qué unidad se evalúa** (imagen, rótulo o secuencia).

| Fuente y lectura registrada | Qué se evaluó | Qué aporta a este entregable |
|---|---|---|
| Emami (2026), tesis de maestría. Resumen y registro institucional. [Repositorio](https://hdl.handle.net/10388/19451) | Localización, clasificación visual y lectura condicional de texto. La matriz registra mAP@50 de 96,2 % para la primera etapa. | Fundamenta la separación de tareas. Ese resultado corresponde a detección; el resumen no acredita máscaras de rótulos. |
| Sharifi, Zibaei y Rezaei (2021). Texto completo. [Artículo](https://doi.org/10.1016/j.mlwa.2021.100104) | DeepHAZMAT detecta señales con YOLOv3-tiny y usa GrabCut y refinamiento geométrico para aislarlas. Los autores describen 1.685 fotografías originales y 13 categorías HAZMAT. | Es el antecedente directo para localizar y aislar el rótulo. La prepublicación y el artículo publicado se citan como versiones de un solo estudio. [Prepublicación](https://arxiv.org/abs/2007.06392) |
| Gaweesh, Khan y Ahmed (2021). Resumen editorial. [Artículo](https://doi.org/10.1177/03611981211026653) | Reconocimiento de información de rótulos en grabaciones de carretera mediante AlexNet. La matriz recoge una exactitud del 95 %. | Aporta contexto sobre reconocimiento en transporte. La exactitud se refiere a identificación y no cuantifica el solapamiento de máscaras. |
| Brylka, Bierwirth y Schwanecke (2021). Texto completo. [Artículo](https://doi.org/10.15480/882.3959) | YOLOv4-tiny detecta etiquetas de mercancías peligrosas; dos U-Net intervienen en el módulo de códigos de barras. Al umbral de 75 %, la precisión fue 97,6 %, el recall por imagen 38,5 % y la detección por secuencia 96,2 %. | Una secuencia puede contener una detección aunque numerosos fotogramas individuales fallen. La presencia de U-Net en este artículo no demuestra su uso para segmentar el rótulo peligroso. |

**Cómo rastrear las cifras.** La hoja *Matriz bibliografica*, filas 5–8, contiene los cuatro registros. *Evaluacion critica* explica por qué se usa cada fuente y *Referencias APA* proporciona los identificadores. En *Umbrales Brylka*, la fila 10 conserva las tres cifras del umbral de 75 %. El F1 de esa hoja se calcula a partir de precisión y recall; se distingue de las cifras publicadas por los autores.

**Recorrido de la revisión**

```text
Cuatro antecedentes de aplicación aprobados
  ├─ Texto completo → Sharifi et al. y Brylka et al.
  └─ Resumen → Emami y Gaweesh et al.
             ↓
Lectura por tarea, métrica y unidad de evaluación
             ↓
Antecedente de máscaras → DeepHAZMAT (GrabCut)
Pregunta del experimento → comparar tres redes de segmentación binaria
             ↑
Fundamentos de arquitectura → Long et al., Badrinarayanan et al., Ronneberger et al.
```

### Búsqueda exploratoria complementaria

El 6 de noviembre se hicieron nueve consultas abiertas sobre el corpus y las arquitecturas. Este registro complementa la matriz aprobada del día siguiente. Las cadenas se transcriben del {download}`registro de procedencia y literatura <../docs/PROCEDENCIA_Y_LITERATURA.md>`.

```text
hazardous material placard segmentation dataset 1685 images 13 classes
hazmat placard segmentation dataset poison oxygen inhalation hazard VOC2012
SeaFront dataset hazardous goods placards segmentation baseline
"hazmat" "segmentation" "U-Net" "FCN"
"hazardous" "placard" "SegNet"
"HAZMAT13" "segmentation" "U-Net"
"hazmat" "U-Net" segmentation signs
"hazmat" "SegNet" segmentation
"hazmat" "FCN" segmentation
```

## Artículos que fundamentan las tres arquitecturas

| Artículo | Datos y comparación | Resultado o aporte | Alcance para este experimento |
|---|---|---|---|
| Long, Shelhamer y Darrell (2015), *Fully Convolutional Networks for Semantic Segmentation*. [Artículo](https://doi.org/10.48550/arXiv.1411.4038) | PASCAL VOC, NYUDv2 y SIFT Flow. Transfiere redes de clasificación a predicción densa | 62,2 % de IoU media en PASCAL VOC 2012 | Fundamenta la salida espacial. PASCAL VOC del artículo y nuestro corpus de rótulos son distintos |
| Badrinarayanan, Kendall y Cipolla, *SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation* (prepublicación 2015, revisión 2016). [Artículo](https://doi.org/10.48550/arXiv.1511.00561) | Escenas viales e interiores SUN RGB-D. Incluye FCN entre los comparadores | Recupera posiciones con índices de máximos y estudia memoria frente a precisión | Uso comparativo de FCN en escenas generales. SegNet es el método propuesto en ese artículo |
| Ronneberger, Fischer y Brox (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*. [Artículo](https://doi.org/10.48550/arXiv.1505.04597) | Segmentación biomédica, incluidas imágenes de microscopía | Combina representación profunda y rasgos de alta resolución mediante concatenación | Fundamenta la hipótesis sobre contornos. Su dominio y protocolo difieren del test HAZMAT |

**Cómo leer las cifras publicadas.** En CamVid, la variante de SegNet entrenada sobre 3,5 mil imágenes y evaluada tras 140 mil iteraciones obtuvo 60,10 % de mIoU. Su codificador sigue las 13 capas convolucionales de VGG16. En microscopía, U-Net obtuvo IoU 0,9203 en PhC-U373 y 0,7756 en DIC-HeLa. Las tablas citadas presentan estimaciones puntuales; no ofrecen una dispersión entre semillas comparable a la de estas corridas. Tanto la resolución y la definición de clase como la política de aumentos difieren entre protocolos. [SegNet, tabla de CamVid](https://arxiv.org/pdf/1511.00561) · [U-Net, tabla 2](https://arxiv.org/pdf/1505.04597)

**Qué se puede comparar.** La matriz destaca un riesgo concreto en Sharifi et al.: al aumentar el número de imágenes, importa en qué momento se separan las fotografías originales y sus variantes. Para nuestro experimento, todas las imágenes conectadas por los enlaces de similitud detectados se asignan al mismo grupo antes de la partición. En Brylka et al., la diferencia entre recall por imagen y detección por secuencia ilustra por qué también declaramos la unidad de evaluación.

En el antecedente directo, Sharifi y colaboradores informan 87,65 % de IoU y 97,58 % de F1 como promedios sobre las 13 categorías. La tabla del artículo no ofrece IC para esos promedios. Aquí se evalúa una máscara binaria por imagen y se pondera por grupo de similitud. Restar directamente ambos resultados mezclaría definiciones de clase, unidad de agregación y particiones. La verificación del artículo publicado complementa la lectura histórica de su prepublicación. [Sharifi et al., tabla 3](https://eprints.whiterose.ac.uk/id/eprint/176122/7/1-s2.0-S2666827021000529-main.pdf)

**Decisión experimental.** De las cuatro fuentes aprobadas, Sharifi et al. es la que documenta explícitamente el aislamiento del rótulo con una máscara. Brylka et al. emplean U-Net para códigos de barras. La elección de FCN, SegNet y U-Net para este entregable se apoya en sus artículos de arquitectura: permiten comparar la reconstrucción desde un mapa grueso, la recuperación de posiciones de máximos y la incorporación de características superficiales. La revisión aporta esta justificación metodológica; su alcance no establece un ranking de frecuencia en HAZMAT.
