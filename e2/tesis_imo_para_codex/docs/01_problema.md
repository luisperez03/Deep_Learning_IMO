# Segmentación binaria de etiquetas de mercancías peligrosas

Estado: definición inicial del entregable 1; no es un protocolo experimental cerrado.

## Motivación y alcance

Localizar la superficie de etiquetas de peligro es un primer componente del proyecto
de tesis, que posteriormente estudiará su clasificación. En este primer experimento
se propone distinguir etiqueta y fondo en fotografías del corpus local VOC2012.
La validez externa para contenedores y condiciones portuarias debe evaluarse: no se
presupone que este corpus represente ese dominio.

## Formulación

Para una imagen RGB, X pertenece a [0,1]^(H×W×3); la máscara Y pertenece a
{0,1}^(H×W), donde 1 representa una etiqueta de peligro anotada y 0 el fondo.
Una red f_theta(X) produce probabilidades por píxel. La tarea es segmentación
semántica binaria: no distingue la clase IMO ni separa instancias adyacentes.

Propuesta de pérdida, por confirmar con literatura y EDA: entropía cruzada binaria
combinada con Dice suave. Para probabilidades p y máscara y:

L_BCE = -mean[y log(p) + (1-y) log(1-p)]

L_Dice = 1 - (2 sum(p*y) + epsilon)/(sum(p) + sum(y) + epsilon)

L = lambda L_BCE + (1-lambda) L_Dice.

No se han fijado lambda, epsilon, resolución, umbral ni arquitectura. Se definirán
antes de las ejecuciones definitivas y se registrarán en configuración.

## Pregunta e hipótesis operativa propuesta

¿Qué desempeño, variabilidad y costo ofrecen tres arquitecturas básicas respaldadas
por literatura para segmentar etiquetas de peligro bajo una misma partición y
presupuesto experimental?

La comparación estimará diferencias pareadas en Dice entre modelos. Una mejora
prácticamente relevante deberá superar un margen delta definido antes del piloto
y del análisis de resultados; aún no se asigna un valor arbitrario a ese margen.

## Datos y unidad experimental

Base inicial: pares JPEGImages/SegmentationClass de VOC2012. La elegibilidad técnica
se registra en inventario.csv; la elegibilidad visual requiere revisión adicional.
Los colores definidos en labelmap.txt, salvo background, son candidatos a primer
plano. Colores desconocidos se excluyen de una conversión automática. Las máscaras
originales se preservan. No se mezclan Hazmat ni SeaFront en este primer bloque.

Unidad de observación: imagen. Unidad de independencia: escena o grupo de imágenes
relacionadas cuando exista dependencia; identificar estos grupos está pendiente.
Los píxeles no constituyen réplicas independientes para la inferencia estadística.

## Evaluación propuesta y decisiones pendientes

- Primaria propuesta: Dice de etiqueta calculado por imagen y agregado con criterio
  explícito de ponderación por escena cuando proceda.
- Secundarias: IoU de etiqueta, mIoU de etiqueta/fondo, métricas por clase, exactitud
  de píxel, tiempos y número de parámetros.
- Definir tratamiento de máscaras y predicciones vacías, antes de evaluar.
- Separar entrenamiento, validación y prueba por grupos; hashes y prueba congelada.
- Piloto de al menos cinco semillas por cada uno de tres modelos seleccionados de
  literatura; dimensionar ejecuciones definitivas según variabilidad y potencia.
- Comparación pareada, intervalos de confianza y corrección por multiplicidad;
  plan final pendiente. No confundir semillas con nuevas escenas independientes.

Este documento no demuestra cumplimiento completo del entregable: quedan revisión
bibliográfica, procedencia/licencia, EDA ampliado, protocolo y entrenamiento.
