# Estado del primer bloque — 2026-11-06

## Completado

- Definición inicial del problema, entrada/salida, pérdida y evaluación propuestas.
- Auditoría de todos los 1.583 pares JPEGImages/SegmentationClass y SHA256 de los
  1.685 JPG. Se comprobaron apertura completa, dimensiones, colores y presencia de etiqueta.
- Inventario por archivo, hashes, distribución de píxeles y presencia de clases.
- Muestra aleatoria reproducible de 30 imágenes en tres paneles, con inspección
  exploratoria del asistente documentada; revisión humana pendiente.
- Cuaderno inicial: seis celdas de código ejecutadas sin errores, tablas, dos
  histogramas y paneles, con decisiones y riesgos por análisis.
- Cuatro pruebas automatizadas del flujo focalizado superadas. Incluyen repetición
  sobre un corpus sintético y comprobación de que sus originales no cambian.

## Resultados

| Hallazgo | Resultado | Implicación |
|---|---:|---|
| Pares técnicamente válidos | 1.583 | Candidatos a revisión, no aprobación de anotación |
| JPG sin máscara | 102 | No tratarlos como negativos |
| Grupos duplicados exactos entre JPG | 0 | No descarta casi duplicados ni escenas relacionadas |
| Discrepancias de conjuntos de clases XML/máscara | 69 | Revisar antes de entrenar segmentación multiclase |
| Máscaras sin XML | 1: image_1018 | No invalida automáticamente su uso binario |
| Píxeles de fondo | 95,6187 % | Exactitud global engañosa como métrica principal |
| Mediana de superficie de etiqueta por imagen | 3,135 % | Preservar objetos pequeños al redimensionar |
| Imágenes con menos del 5 % de etiqueta | 1.148 | Justificar pérdida, resolución y muestreo |

Fuentes regenerables: `resultados/inicio_voc/resumen.json`, `inventario.csv` y el
cuaderno. No son resultados de entrenamiento.

## Próximo paso

Completar procedencia/licencia y grupos de escenas/casi duplicados, revisar máscaras
difíciles a resolución original y ampliar el EDA. Después fijar splits y protocolo,
con tres modelos seleccionados por evidencia bibliográfica. No hay aún prueba
congelada, piloto, modelos entrenados ni Jupyter Book compilado.

El auditor general heredado y sus cuatro pruebas no se certifican en este bloque;
las cuatro pruebas superadas son las de `tests/test_inicio_voc.py`.
