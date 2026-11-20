# Ruta metodológica — versión inicial

## Fase 0 — Auditoría de datos (ACTUAL)
1. Inventario recursivo de imágenes por dataset y resolución/formato.
2. Reconocimiento de COCO (JSON), YOLO (TXT y YAML) y VOC (XML y máscaras cuando existan).
3. Tabla exacta de clases, ID, frecuencia y tipo de anotación.
4. Verificación de rutas imagen-anotación, faltantes, errores, múltiples etiquetas por imagen, divisiones train/val/test.
5. Evaluación de compatibilidad técnica, semántica, estadística y de licencias; no combinar datos todavía.
6. Entrega de informe y propuesta de transformación (sujeta a aprobación).

## Fase 1 — Definición operativa del problema (PENDIENTE)
- Decidir si es detección, segmentación semántica, segmentación de instancias, clasificación, o pipeline mixto.
- Establecer unidad experimental: imagen, objeto etiquetado, contenedor, escena.
- Definir ontología: qué corresponde a una etiqueta IMO, clase/división y reglas para no IMO, desconocido, múltiples rótulos o etiquetas visualmente ambiguas.
- Elegir datasets válidos, documentar origen/licencia, disponibilidad de máscaras, tamaño efectivo y riesgos de sesgo.

## Fase 2 — Estandarización y diseño de muestreo (PENDIENTE)
- Conservar datasets intactos; derivar copias con transformaciones verificables.
- Diseñar mapeo explícito origen -> clase canónica y marcar clases no equivalentes como `NO_MAPEADA`.
- Mantener geometría original de cajas o máscaras; no presentar cajas como segmentación de píxel.
- Split sin solapamiento por imagen/escena/fuente y, si aplica, barco/contenedor/captura.
- EDA: tamaños, distribuciones por clase, dominio, resoluciones, proporciones y calidad.

## Fase 3 — Experimentos didácticos (PENDIENTE)
- Comprender e implementar neurona, funciones de activación, pérdida, gradiente y retropropagación.
- Construir arquitectura elemental (por ejemplo CNN pequeña) para comprender el aprendizaje.
- Comparar líneas base y variantes de complejidad creciente, explicando cada diferencia.
- Separar experimentos de clasificación sobre recortes de etiquetas de experimentos de localización en imágenes completas.
- Diseñar posteriormente segmentación real si las máscaras o un plan de anotación las hacen viables.

## Fase 4 — Evaluación y redacción (PENDIENTE)
- Clasificación: matriz de confusión, precisión/recobrado, macro F1 y desempeño por clase.
- Detección: precisión/recobrado y mAP con umbrales IoU definidos.
- Segmentación: IoU, Dice y desempeño por clase (solo si hay máscaras verdaderas).
- Incertidumbre, intervalos y variabilidad por semillas/repeticiones, cuando proceda.
- Análisis de errores, amenazas a la validez y reproducibilidad.

## Criterio de avance
La aprobación de cada fase requiere registro de hallazgos y decisión del investigador; Codex propone, no decide unilateralmente.
