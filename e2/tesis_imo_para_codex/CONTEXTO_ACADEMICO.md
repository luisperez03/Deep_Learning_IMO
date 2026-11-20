# Contexto académico de la investigación

## Programa y propósito
- Programa: Maestría en Estadística Aplicada, Universidad del Norte (Colombia).
- Formación previa del investigador: Ingeniería Industrial.
- Campo de aplicación: visión por computador aplicada a etiquetas de mercancías peligrosas (IMO/IMDG), particularmente en contenedores y logística portuaria.
- Alcance: trabajo de grado **académico y experimental**. La intención es aprender e implementar conceptos matemáticos, estadísticos y computacionales, no construir un sistema de inspección industrial completo.

## Problema de interés
En una fotografía puede haber cero, una o varias etiquetas. Hay clases diferentes que pueden parecer similares, y variaciones de diseño, perspectiva, iluminación y apariencia aun dentro de una misma clase. Una solución potencial incluye dos etapas: (1) localizar/detectar o segmentar las etiquetas presentes, independientemente de su clase; (2) clasificar cada etiqueta identificada.

El director/profesor sugirió explorar **segmentación** como formulación del problema. Esto no implica que ya exista una decisión cerrada sobre arquitectura, máscaras disponibles ni protocolo de evaluación. Una etapa de detección puede ser una base viable antes de segmentación, dependiendo de las anotaciones reales.

## Los tres directorios actuales
- `Hazmat Pacards.v5i.coco`: nombre indicado por el usuario, posiblemente COCO, por comprobar.
- `VOC2012`: nombre indicado por el usuario, no se da por hecho que corresponda íntegramente a una descarga estándar.
- `SeaFront_v1_0_0`: nombre indicado por el usuario, contenido y anotaciones por verificar.

La captura de pantalla solo permite confirmar la existencia de las carpetas, NO su estructura interna. La auditoría local determinará si son técnicamente compatibles y semánticamente útiles.

## Objetivo tentativo (por afinar con el profesor)
Investigar, implementar y evaluar experimentalmente una estrategia de visión por computador para localizar y clasificar etiquetas IMO/IMDG en imágenes, examinando el aporte y las limitaciones de detección y segmentación, con énfasis en el desarrollo y comprensión de componentes fundamentales de redes neuronales.

## Condiciones didácticas
- Investigar primero y programar con justificación; no resolver todo mediante paquetes de alto nivel.
- Implementar elementos matemáticos desde cero cuando aporte al aprendizaje, y usar bibliotecas como contraste cuando corresponda.
- Formular hipótesis y criterios de evaluación antes de cada experimento.
- Separar capacidad para detectar una etiqueta de capacidad para identificar su clase IMO.
- Tratar cambios de dominio entre fotografías reales e imágenes sintéticas como posible amenaza de validez.
