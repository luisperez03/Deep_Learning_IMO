# Lectura inicial de los 22 resultados únicos de Q1

17 de noviembre de 2026. Cribado sugerido por el asistente a partir de títulos y resúmenes exportados; no es revisión de texto completo ni validación humana. Las decisiones se refieren a antecedentes del dominio (RQ-L1). No constituyen selección de redes ni exclusión definitiva de un mecanismo transferible en RQ-L2/L3.

## Prioridad de lectura de texto completo

| ID local | Motivo para conservar y cuestión por comprobar |
|---|---|
| E2-0001 | Fusión visual/textual para rótulos, clases abiertas y desbalance. Verificar anotaciones y evaluación; el resumen describe detección, no máscaras. DOI: 10.1109/tai.2023.3298588. |
| E2-0003 | Rótulos HAZMAT y YOLOv8. El resumen mezcla clasificación con mAP50: aclarar tarea, protocolo y anotación antes de citar resultados. |
| E2-0004 | Etiquetas de mercancías peligrosas, imágenes reales/sintéticas y YOLO. Antecedente de dominio; verificar fuente original y correspondencia con Brylka del E1. |
| E2-0006 | Umbral HSV, morfología y coincidencia de características. Posible referencia para baseline clásico, no red profunda. Comprobar publicación y evaluación de segmentación. |
| E2-0007 | Rectificación geométrica y detección HAZMAT RGB-D. Evaluar qué componente sería aplicable a nuestras entradas RGB sin profundidad. |
| E2-0013 | Detección de señales HAZMAT integrada en robótica/OpenCV. Verificar si aporta evaluación suficiente o sólo contexto de aplicación. |
| E2-0014 | Detección de vehículos ADR y rótulos en túneles. Verificar si evalúa localización de la etiqueta o sólo el vehículo. |
| E2-0016 | YOLOv5-Small entrenado con imágenes sintéticas y evaluado en imágenes reales exteriores. Interesa por cambio de dominio; mAP de cajas no equivale a IoU de máscaras. DOI: 10.1109/iscas56072.2025.11043711. |
| E2-0018 | DeepHAZMAT: YOLO + GrabCut adaptativo + morfología/convex hull según el resumen. Antecedente directo y posible comparación de pipeline; no presentarlo como segmentador neuronal de extremo a extremo. DOI: 10.1016/j.mlwa.2021.100104. |
| E2-0021 | Reconocimiento/localización HAZMAT con métodos clásicos y profundos; restricciones de hardware, iluminación y fondos. Puede fundamentar condiciones de evaluación. |

## Duda a resolver

E2-0011: visión estéreo, tamaño/profundidad y SLAM. El resumen menciona señales genéricas, pero no demuestra segmentación de rótulos HAZMAT. Revisar texto antes de decidir si aporta un antecedente pertinente.

## Exclusiones sugeridas de RQ-L1

| ID local | Motivo explícito |
|---|---|
| E2-0002 | Peligros viales genéricos y cajas; no rótulos de mercancías peligrosas. |
| E2-0005 | Señales acústicas para estado de vías ferroviarias; no imágenes de rótulos. |
| E2-0008 | Clasificación de escenas de incendio/accidente; no rótulos. |
| E2-0009 | Detección/clasificación de camiones completos; el resumen no indica evaluación de rótulos. |
| E2-0010 | Clasificación de chatarra/materiales; no rótulos. |
| E2-0012 | OCR de ingredientes cosméticos; no pictogramas HAZMAT/IMDG. |
| E2-0015 | Objetos peligrosos en rayos X, no rótulos RGB. Se puede reconsiderar como mecanismo de otra modalidad, sin inferir transferencia demostrada. |
| E2-0017 | Clasificación de residuos; no rótulos. |
| E2-0019 | Objetos ocultos en imágenes THz; no rótulos. Una eventual utilidad transversal requiere evaluación separada. |
| E2-0020 | Erratum de declaraciones de conflicto; no estudio experimental independiente. Vincular a DeepHAZMAT como documento asociado. |
| E2-0022 | Clasificación hiperespectral de sustancias; no segmentación de rótulos RGB. |

Balance sugerido: **10 para texto completo, 1 dudoso, 11 exclusiones**. Hasta validación, el inventario generado conserva estado pendiente. No reportar estos números como exclusiones definitivas en PRISMA.

## Implicación para las siete redes

Q1 aporta el contexto aplicado y los problemas reales: fondos, perspectiva, iluminación, desbalance, coste y cambio sintético-real. No basta para construir siete candidatos de segmentación. Esa elección debe venir de Q2/Q3 y las consultas transversales, seguida por verificación del código y de la viabilidad. Los seis nombres de `CANDIDATOS_EXPLORATORIOS.md` son semillas para contrastar, no ganadores anticipados.

Fuentes de esta lectura: los registros y resúmenes de las dos exportaciones Q1 preservadas en `exportaciones/`; correspondencias completas en `registros_conciliados.json`. No se verificaron aún las cifras de rendimiento contra los textos completos.
