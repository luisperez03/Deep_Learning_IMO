# Inspección exploratoria de paneles

Fecha: 2026-11-06. Revisor: asistente Codex. Alcance: las 30 miniaturas de los tres
paneles generados con semilla 20261106. No constituye revisión humana ni aprobación
de todas las máscaras a resolución original.

- Se observan etiquetas en tablas, suelo, maletas, recipientes y otros fondos de
  interior. La muestra no demuestra representatividad de condiciones portuarias.
- Muchas superposiciones corresponden visualmente con superficies de etiquetas.
  Hay oclusiones, etiquetas parcialmente fuera de cuadro y objetos pequeños.
- Revisar a resolución completa `image_1461`, `image_0677` y `image_1251`, por
  superposición de objetos, oclusiones y recortes. No se declaran errores sin esa revisión.
- `image_0025` muestra un símbolo de riesgo biológico, mientras que la máscara usa
  el color asignado a inhalation-hazard. La auditoría automática también registra
  discrepancia entre clases XML y máscara. Mantener la alerta de calidad semántica.
- Los tres paneles son legibles para revisión general, pero no suficientes para
  medir calidad de bordes ni estimar una tasa de error de anotación.

Decisión de trabajo: mantener segmentación binaria como alcance propuesto. Antes
de convertir datos de entrenamiento, definir elegibilidad, reglas de oclusión y
revisión independiente. Los casos señalados no se eliminan automáticamente.
