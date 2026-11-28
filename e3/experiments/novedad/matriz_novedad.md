# Matriz de novedad (R1)

Fecha de las consultas: 2026-11-23 (hora de Bogotá). Los conteos de resultados son los que declara cada fuente;
los revisados son los títulos (y resúmenes, si los hay) efectivamente leídos, hasta 50 por consulta (12 en Hugging Face Papers);
los relevantes siguen los criterios de `cadenas.md`. `amenaza` marca si algún relevante podría anticipar C1 (encogimiento atenuado) o C2 (expansión consistente):
`parcial` significa que anticipa una parte (expansión del núcleo con offset regresado) sin llegar a la regla.

| fecha | base | cadena | resultados | revisados | relevantes | amenaza_C1_C2 | observacion |
|---|---|---|---|---|---|---|---|
| 2026-11-23 | OpenAlex | C1 | 28195 | 50 | 7 trabajos (10 registros) | no | Los relevantes detectan, clasifican o leen rótulos; ninguno segmenta con red. Nuevos frente a E2: Vision on the Move (2025), HMD-Net (2022), DOT-HAZMAT (2022), informe de lectores de placas en carretera. El resto es ruido de riesgo, logística y derecho. |
| 2026-11-23 | OpenAlex | C2 | 1914 | 50 | 0 | no | Solo ruido de transporte y química; ningún estudio de visión en español entre los 50. |
| 2026-11-23 | OpenAlex | C3 | 1515 | 50 | 4 | no | Mejoras de DBNet por backbone, atención, factor k adaptativo (ASM-DBNet 2026), SPP o convergencia (IDBNet, HTBNet). Ninguna cambia la regla de encogimiento por grosor según el resumen. Muchas aplicaciones industriales de texto (medidores, facturas, placas de equipos). |
| 2026-11-23 | OpenAlex | C4 | 368 | 50 | 4 trabajos (6 registros) | parcial | Text kernel expansion (2024) y Deformable Kernel Expansion (3 registros) expanden el núcleo regresando offsets; PSENet; offsetting de polígonos con aumento de borde (2020, sin leer a fondo). Ninguno resuelve d como punto fijo de la regla de encogimiento. |
| 2026-11-23 | OpenAlex | C5 | 156 | 50 | 2 | no | Ningún trabajo aplica DBNet o núcleos encogidos a rótulos hazmat. Aparece una tesis de 2026 sobre placas hazmat con YOLOv9-t (detección, clasificación y texto) y Similar Mask (2024, texto en tráfico). |
| 2026-11-23 | OpenAlex | C6 | 168 | 50 | 2 | no | Transferencia de DB fuera del texto: DB M-Net (2020, DB sobre una variante de U-Net, TC de esófago) y binarización adaptativa sobre Mask R-CNN (2021). Usan la binarización, no el encogimiento del núcleo. Posiciones 6-50: ruido de otras disciplinas. |
| 2026-11-23 | OpenAlex | C7 | 147 | 50 | 0 | no | Ruido arqueológico (reducción de núcleos líticos). Un único registro de detección de texto sin resumen. |
| 2026-11-23 | OpenAlex | C8 | 638 | 50 | 3 | no | Representaciones por transformada de distancia o esqueleto (TextMountain 2020, T-Skeleton 2024, skeleton-aware distance transform 2023) como mapas a regresar; ninguna atenúa el encogimiento del núcleo por radio inscrito según el resumen. |
| 2026-11-23 | arXiv | C1 | 3 | 3 | 2 | no | DeepHAZMAT (YOLO y GrabCut) y detección planar RGB-D (E2-0007). Solo 3 resultados. |
| 2026-11-23 | arXiv | C2 | 0 | 0 | 0 | no | 0 resultados. El primer intento falló por error de red (HTTP 0) y el reintento respondió. |
| 2026-11-23 | arXiv | C3 | 8 | 8 | 4 | no | DBNet, DBNet++, GWNet (k adaptativo, no encogimiento) e IIM (DB en localización de multitudes). Ningún encogimiento atenuado. |
| 2026-11-23 | arXiv | C4 | 1 | 1 | 1 | parcial | Deformable Kernel Expansion (2023): encogimiento fijo r = 0,4 y offsets de vértices regresados (texto completo leído). |
| 2026-11-23 | arXiv | C5 | 0 | 0 | 0 | no | 0 resultados: el cruce mecanismo-problema está vacío en arXiv. |
| 2026-11-23 | arXiv | C6 | 4 | 4 | 4 | no | DBNet, DBNet++, OCR manuscrito árabe e IIM (transferencia de DB a localización de multitudes, 2020). |
| 2026-11-23 | arXiv | C7 | 0 | 0 | 0 | no | 0 resultados. |
| 2026-11-23 | arXiv | C8 | 0 | 0 | 0 | no | 0 resultados. |
| 2026-11-23 | Semantic Scholar (masiva) | C1 | 64 | 50 | 14 | no | Hazmat: DeepHAZMAT, Zhao 2013 (dos trabajos), Mohamed 2018, Zhang 2024, y varios de detección o clasificación (2008-2026, YOLO, color, SURF). Todos detectan, clasifican o reconocen; ninguno segmenta con red. Orden por citas, no por relevancia. |
| 2026-11-23 | Semantic Scholar (masiva) | C2 | 3 | 3 | 0 | no | Normativa de transporte; sin visión. |
| 2026-11-23 | Semantic Scholar (masiva) | C3 | 75 | 50 | 4 | no | DBNet, DBNet++, GWNet y detección de medidores con DB. Homónimos de DBNet (retinopatía, carriles) y ruido. |
| 2026-11-23 | Semantic Scholar (masiva) | C4 | 2 | 2 | 2 | parcial | Text kernel expansion y Deformable Kernel Expansion; ambos con regresión. |
| 2026-11-23 | Semantic Scholar (masiva) | C5 | 2 | 2 | 0 | no | DBNet en señales turísticas (texto sobre la señal), no el rótulo. |
| 2026-11-23 | Semantic Scholar (masiva) | C6 | 19 | 19 | 0 | no | Todas las aplicaciones siguen siendo de texto (médico, facturas, libros antiguos). |
| 2026-11-23 | Semantic Scholar (masiva) | C7 | 1 | 1 | 0 | no | Un registro general de texto en escenas. |
| 2026-11-23 | Semantic Scholar (masiva) | C8 | 0 | 0 | 0 | no | 0 resultados. |
| 2026-11-23 | OpenReview | C1 | 10000 (tope de la API) | 50 | 2 | no | Placas hazmat con YOLOv5-Small y datos sintéticos (E2-0016) y Vision on the Move (2025). El resto, ruido; muchos registros son comentarios de revisión sin título. |
| 2026-11-23 | OpenReview | C2 | 7 | 7 | 0 | no | Sin relación. |
| 2026-11-23 | OpenReview | C3 | 10000 (tope de la API) | 50 | 1 | no | Texto con DB y segmentación semántica; el resto es binarización de documentos, de redes o de otra disciplina. El tope de 10.000 no es un conteo. |
| 2026-11-23 | OpenReview | C4 | 10000 (tope de la API) | 50 | 4 | parcial | Text kernel expansion, Kernel Adaptive Convolution (mapa de distancia), PSENet y EK-Net (ICASSP 2024, distancia de expansión regresada). |
| 2026-11-23 | OpenReview | C5 | 10000 (tope de la API) | 50 | 1 | no | Solo el registro de DB con segmentación semántica. |
| 2026-11-23 | OpenReview | C6 | 10000 (tope de la API) | 50 | 1 | no | Solo el registro de DB con segmentación semántica. |
| 2026-11-23 | OpenReview | C7 | 2 | 2 | 0 | no | Sin relación. |
| 2026-11-23 | OpenReview | C8 | 10000 (tope de la API) | 50 | 5 | parcial | Reinforcement Shrink-Mask (TMM 2023), Kernel Adaptive Convolution, EK-Net, Text kernel expansion y Zoom Text Detector: kernel con offsets o mapas de distancia regresados, todos con encogimiento de la etiqueta fijo. |
| 2026-11-23 | GitHub (repositorios) | C1 | 0 | 0 | 0 | no | 0 repositorios con la cadena completa; la búsqueda de GitHub exige todas las palabras. Ver consultas cortas. |
| 2026-11-23 | GitHub (repositorios) | C2 | 0 | 0 | 0 | no | 0 repositorios. |
| 2026-11-23 | GitHub (repositorios) | C3 | 0 | 0 | 0 | no | 0 repositorios. |
| 2026-11-23 | GitHub (repositorios) | C4 | 0 | 0 | 0 | no | 0 repositorios. |
| 2026-11-23 | GitHub (repositorios) | C5 | 3 | 3 | 0 | no | Tres reimplementaciones genéricas de DBNet; ninguna con rótulos. |
| 2026-11-23 | GitHub (repositorios) | C6 | 0 | 0 | 0 | no | 0 repositorios. |
| 2026-11-23 | GitHub (repositorios) | C7 | 0 | 0 | 0 | no | 0 repositorios. |
| 2026-11-23 | GitHub (repositorios) | C8 | 0 | 0 | 0 | no | 0 repositorios. |
| 2026-11-23 | Hugging Face Hub | C1 | 17 | 17 | 2 | no | Dos conjuntos de datos de placas hazmat (hmgill/Vehicle-Hazmat-Placards, barissonmezee/un-hazmat); el resto son modelos de texto, tablas regulatorias y espacios sin relación. |
| 2026-11-23 | Hugging Face Hub | C2 | 0 | 0 | 0 | no | 0 resultados. |
| 2026-11-23 | Hugging Face Hub | C3 | 15 | 15 | 1 | no | Modelos DBNet y DBNet++ de texto; uno para códigos de contenedor (mattkro/container-code-dbnetpp, texto, no rótulos). |
| 2026-11-23 | Hugging Face Hub | C4 | 82 | 82 | 0 | no | Los 82 resultados son modelos de difusión 'unclip', homónimos sin relación. |
| 2026-11-23 | Hugging Face Hub | C5 | 0 | 0 | 0 | no | 0 resultados. |
| 2026-11-23 | Hugging Face Hub | C6 | 0 | 0 | 0 | no | 0 resultados. |
| 2026-11-23 | Hugging Face Hub | C7 | 0 | 0 | 0 | no | 0 resultados. |
| 2026-11-23 | Hugging Face Hub | C8 | 0 | 0 | 0 | no | 0 resultados. |
| 2026-11-23 | GitHub (consulta corta) | cortas | 1 | 1 | 1 | no | Consulta: hazmat segmentation |
| 2026-11-23 | GitHub (consulta corta) | cortas | 16 | 16 | 3 (de 16) | no | Consulta: hazmat placard |
| 2026-11-23 | GitHub (consulta corta) | cortas | 25 | 25 | 6 (de 25) | no | Consulta: hazmat detection |
| 2026-11-23 | GitHub (consulta corta) | cortas | 1 | 1 | 1 (conjunto de datos) | no | Consulta: dangerous goods label detection |
| 2026-11-23 | GitHub (consulta corta) | cortas | 3 | 3 | 1 | no | Consulta: hazmat-13 |
| 2026-11-23 | GitHub (consulta corta) | cortas | 1 | 1 | 1 | no | Consulta: DeepHAZMAT |
| 2026-11-23 | GitHub (consulta corta) | cortas | 0 | 0 | 0 | no | Consulta: adaptive shrink ratio text detection |
| 2026-11-23 | GitHub (consulta corta) | cortas | 0 | 0 | 0 | no | Consulta: unclip_ratio adaptive |
| 2026-11-23 | GitHub (consulta corta) | cortas | 1 | 1 | 0 | no | Consulta: differentiable binarization segmentation |
| 2026-11-23 | GitHub (consulta corta) | cortas | 4 | 4 | 0 | no | Consulta: text kernel expansion |
| 2026-11-23 | GitHub (consulta corta) | cortas | 0 | 0 | 0 | no | Consulta: shrink ratio DBNet |
| 2026-11-23 | GitHub (consulta corta) | cortas | 0 | 0 | 0 | no | Consulta: IMDG label detection |
| 2026-11-23 | Hugging Face Papers | ad hoc | 0 | 0 | 0 | no | Consulta: hazmat. Revisados los 12 primeros. Sin artículos. |
| 2026-11-23 | Hugging Face Papers | ad hoc | 120 | 12 | 2 | no | Consulta: differentiable binarization. Revisados los 12 primeros. DBNet y DBNet++ entre los primeros; el resto, binarización de redes, de documentos o de rasterizadores. |
| 2026-11-23 | Hugging Face Papers | ad hoc | 120 | 12 | 2 | no | Consulta: text kernel expansion shrink. Revisados los 12 primeros. PSENet y similares; ruido de modelos de lenguaje. |
| 2026-11-23 | Hugging Face Papers | ad hoc | 120 | 12 | 0 | no | Consulta: hazardous material placard segmentation. Revisados los 12 primeros. Segmentación de residuos y humo; ningún rótulo. |
| 2026-11-23 | Hugging Face Papers | ad hoc | 120 | 12 | 2 | no | Consulta: adaptive shrink ratio text detection. Revisados los 12 primeros. PSENet y FAST; sin regla de encogimiento adaptativa. |
| 2026-11-23 | Google Scholar | C1-C8 | no accesible | 0 | 0 | n/a | HTTP 403 desde el cliente automático (2026-11-23). Queda para quien tenga navegador. |
| 2026-11-23 | Papers with Code | C1-C8 | no accesible | 0 | 0 | n/a | El sitio redirige a huggingface.co/papers/trending (2026-11-23). Se sustituyó por Hugging Face Papers (filas siguientes). |
| 2026-11-23 | Scopus | C1-C8 | pendiente |  |  |  | Requiere el proxy de la biblioteca; cadenas listas en pendiente_bases_suscripcion.md. |
| 2026-11-23 | Web of Science | C1-C8 | pendiente |  |  |  | Ídem. |
| 2026-11-23 | IEEE Xplore | C1-C8 | pendiente |  |  |  | Ídem. Aquí están los textos completos que no se pudieron leer (OKR, ADNet con fórmula, Similar Mask). |

## Snowballing (OpenAlex y Semantic Scholar, 2026-11-23)

Hacia atrás: referencias del trabajo; hacia adelante: citas, filtradas en el servidor por palabras del mecanismo
(shrink, shrunk, kernel, unclip, expansion, offset, inscribed, thickness) en título o resumen para DBNet y DBNet++,
y sin filtro para los demás. Crudos en `resultados/snowball__*.json` y `resultados/snowball_hazmat.json`.

| semilla | referencias (resueltas) | citas totales | citas tras filtro | revisadas | relevantes | observación |
|---|---|---|---|---|---|---|
| Real-Time Scene Text Detection with Differentiable Binarization (https://doi.org/10.1609/aaai.v34i07.6812) | 43 (43) | 1067 | 71 | 71 | 11 | Citas relevantes por mecanismo: ADNet, Reinforcement Shrink-Mask, DKE, Text kernel expansion, EK-Net y EK-Net++, Optimal Kernel (OKR), Text kernel calculation, Pull Pole Points (MTD), CentripetalText y FAST. Ver comparativa.md. Las referencias hacia atrás son PSENet, TextSnake, PixelLink y ATRR (Wang et al., 2019), sin regla adaptativa. El registro de OpenAlex es el de AAAI (1067 citas); otros registros de DBNet no se suman. |
| Real-Time Scene Text Detection With Differentiable Binarization and Ad (https://doi.org/10.1109/tpami.2022.3155612) | 68 (68) | 523 | 41 | 41 | 8 | Mismo subconjunto de DBNet (sin CentripetalText ni OKR en el filtro). DBNet++ cambia la fusión de escalas, no la etiqueta. |
| Arbitrary-Shaped Text Detection With Adaptive Text Region Representati (https://doi.org/10.1109/access.2020.2999069) | 49 (49) | 16 | 16 | 16 | 1 | Solo 16 citas. CM-Net (concentric mask, 2022) es la única de mecanismo; no se leyó el texto. Nadie cita la expansión en inglete con distancia predicha como método. |
| Adaptive Shrink-Mask for Text Detection (https://doi.org/10.48550/arxiv.2111.09560) | 38 (38) | 0 | 0 | 0 | 0 | OpenAlex no registra citas del preprint; su sucesor, Reinforcement Shrink-Mask (TMM 2023), aparece en las citas de DBNet. Las referencias incluyen TextSnake, FCENet, ContourNet, TextRay y ABCNet. |
| DeepHAZMAT: Hazardous Materials Sign Detection and Segmentation with R (https://doi.org/10.2139/ssrn.3649600) | 0 (0) | 5 | 5 | 5 | 1 | Registro SSRN/OSF de 5 citas; la única de dominio cercano es Chemical Safety Sign Detection: A Survey and Benchmark (IJCNN 2022, detección). |
| SegFormer: Simple and Efficient Design for Semantic Segmentation with  (https://doi.org/10.48550/arxiv.2105.15203) | 75 (75) | 11671 | 0 | 0 | 0 | 11671 citas; 0 con palabras del problema (hazmat, placard, dangerous goods, warning sign) en título o resumen. Las 75 referencias son de segmentación semántica general. |
| DeepHAZMAT (MLWA 2021) (Semantic Scholar, DOI 10.1016/j.mlwa.2021.100104) | n/d | 17 (hasta 100) | sin filtro | 17 | 2 | HMD-Net y un sistema de identificación de sustancias peligrosas desde aeronaves no tripuladas (2022, no leído). Ningún citante segmenta rótulos con red. |
| Zhang et al. 2024, fusion visual y textual (TAI) (Semantic Scholar, DOI 10.1109/TAI.2023.3298588) | n/d | 3 (hasta 100) | sin filtro | 3 | 1 | Detection of Multi-Class Hazard Signs using YOLO (2025): detección. |
| Mohamed et al. 2018 (SSRR) (Semantic Scholar, DOI 10.1109/SSRR.2018.8468639) | n/d | 3 (hasta 100) | sin filtro | 3 | 1 | Ídem; más el trabajo RGB-D de geometría previa (E2-0007). |
| Zhao et al. 2013 (GlobalSIP) (Semantic Scholar, DOI 10.1109/GlobalSIP.2013.6736996) | n/d | 3 (hasta 100) | sin filtro | 3 | 2 | Zhang 2024 y el trabajo RGB-D de geometría previa; clásicos o de detección. |
| HMD-Net 2022 (Entropy) (Semantic Scholar, DOI 10.3390/e24040466) | n/d | 3 (hasta 100) | sin filtro | 3 | 2 | Dangerous Goods Detection and Warning (2025) y reidentificación de vehículos hazmat en túneles (2023); detección. |

Las cifras entre fuentes no se suman como estudios únicos: un mismo trabajo aparece en varias.
Los resultados crudos, con título, año, DOI o URL y resumen, están en `resultados/<fuente>__<cadena>.json`.

