# Procedencia y revisión bibliográfica inicial verificable

**Alcance de este registro.** Esta nota conserva la exploración del 6 de noviembre. El capítulo «Estado del arte y elección de modelos» integra, además, las cuatro fuentes aprobadas en la matriz bibliográfica del 7 de noviembre.

Consulta web: 2026-11-06. Se leyeron fuentes de autores y documentos originales.
No se descargaron datasets, no se enviaron mensajes a autores y no se redistribuyó
ninguna imagen. Esta búsqueda abierta no equivale a una revisión exhaustiva de
Scopus, WoS e IEEE Xplore; no se inventan conteos PRISMA ni acceso a esas bases.

## Procedencia probable del corpus local

El repositorio de autores [HAZMAT13](https://github.com/mrl-amrl/HAZMAT13) describe
1.685 imágenes originales, 13 clases, cajas y máscaras en formato Pascal VOC.
El número y los nombres de clases coinciden estrechamente con nuestro inventario.
Esto apoya la identificación como **HAZMAT-13 probable**; no demuestra igualdad
binaria con el archivo descargable, porque no se descargó ni comparó ese archivo.

El repositorio declara [licencia MIT](https://raw.githubusercontent.com/mrl-amrl/HAZMAT13/main/LICENSE),
copyright 2021 MRL Advanced Mobile Robotics Laboratory. Esa evidencia aplica al
repositorio; antes de redistribuir el archivo de imágenes alojado externamente,
confirmar el alcance de la licencia y conservar su aviso. No atribuir CC BY 4.0
a VOC por la licencia de Hazmat Placards, que es otra fuente.

Artículo asociado por los autores: Sharifi, Zibaei y Rezaei (2021),
*A deep learning based hazardous materials (HAZMAT) sign detection robot with
restricted computational resources*, DOI 10.1016/j.mlwa.2021.100104.
La versión previa [DeepHAZMAT](https://arxiv.org/abs/2007.06392) describe un detector
YOLOv3-tiny seguido de GrabCut y posprocesamiento. No equivale a un benchmark de
tres redes de segmentación semántica extremo a extremo.

## Fuentes primarias para las familias candidatas

| Fuente | Aporte comprobado | Papel en el proyecto | Límite |
|---|---|---|---|
| [Long et al., FCN](https://arxiv.org/abs/1411.4038), DOI 10.48550/arXiv.1411.4038 | Predicción densa mediante redes totalmente convolucionales | Fundamento candidato | No demuestra recurrencia como baseline HAZMAT |
| [Badrinarayanan et al., SegNet](https://arxiv.org/abs/1511.00561), DOI 10.48550/arXiv.1511.00561 | Decodificador con índices de pooling; compara con FCN y otras redes | Fundamento candidato, diferente mecanismo de reconstrucción | Escenas generales, no etiquetas HAZMAT |
| [Ronneberger et al., U-Net](https://arxiv.org/abs/1505.04597), DOI 10.48550/arXiv.1505.04597 | Contracción y expansión con información de localización | Fundamento candidato | Aplicación biomédica original |
| [DeepHAZMAT](https://arxiv.org/abs/2007.06392), DOI 10.48550/arXiv.2007.06392 | Detección CNN y segmentación mediante GrabCut | Contexto directo del problema | GrabCut no cuenta como uno de tres modelos DL de segmentación |

Las adaptaciones pequeñas de esas familias serían configuraciones nuevas: no
llamarlas reproducciones exactas de los artículos ni elegir profundidades a
conveniencia para conseguir mejores resultados. Falta justificar recurrencia y
cerrar arquitectura/espacio de búsqueda según la guía del curso.

## Registro de búsqueda abierta

Consultas ejecutadas con el buscador web:

1. `hazardous material placard segmentation dataset 1685 images 13 classes`
2. `hazmat placard segmentation dataset poison oxygen inhalation hazard VOC2012`
3. `SeaFront dataset hazardous goods placards segmentation baseline`
4. `"hazmat" "segmentation" "U-Net" "FCN"`
5. `"hazardous" "placard" "SegNet"`
6. `"HAZMAT13" "segmentation" "U-Net"`
7. `"hazmat" "U-Net" segmentation signs`
8. `"hazmat" "SegNet" segmentation`
9. `"hazmat" "FCN" segmentation`

Cribado: se retuvieron repositorios de los autores y artículos originales
pertinentes; resultados sobre empresas de transporte, abreviaturas FCN sin relación
con redes y segmentación de otros tipos de riesgo no se usaron como evidencia de
frecuencia de baselines de rótulos. No se obtuvo una síntesis suficiente para
afirmar que FCN/SegNet/U-Net son los tres más recurrentes en HAZMAT.

**Estado honesto:** procedencia probable documentada y candidatos fundamentados;
selección bibliográfica final pendiente. La configuración mantiene
`literature_selection_finalized=false`. El siguiente paso académico es extender
la búsqueda institucional o acordar con el docente el uso de baselines de
segmentación general para esta aplicación. La preparación técnica no debe ocultar
esta diferencia respecto a la rúbrica.
