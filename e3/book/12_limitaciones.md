(sec-limitaciones)=
# Limitaciones

El documento recuperado `experiments/estandar.md` se incluye como antecedente. Su declaración de congelamiento con `e3-protocolo` no está verificada en el checkout local. Las carencias identificadas siguen abiertas; no se interpreta el antecedente como certificación de cumplimiento.

```{include} ../experiments/estandar.md
:start-line: 6
```

## Otras limitaciones

- **Test observado.** Los rivales se evaluaron en E1 y E2; ahora el comparador visual permite inspeccionar las 218 imágenes. Esta exposición debe declararse en el cierre, sin usarla para ajustar recetas o seleccionar semillas. No se presenta como una evaluación confirmatoria independiente.
- **Un solo conjunto de datos**, con licencia de imágenes sin confirmar (`data/DATA_STATEMENT.md`).
- **Sin metadatos de adquisición**: la agrupación por similitud perceptual no sustituye la independencia por escena.
- **Supuestos del teorema** violados por los datos reales (píxeles, no convexidad, errores de la red;
  {ref}`sec-teorema`).
- **Asimetría de esfuerzo de diseño** del original frente a los rivales (`design_log.md`).
- **Energía solo estimada**, calibración del original solo sobre el núcleo.
