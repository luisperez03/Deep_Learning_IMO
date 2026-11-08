# Segmentación de etiquetas de mercancías peligrosas

**Deep Learning · Entregable 1 · Noviembre de 2026**

**Autores:** Luis Perez y Daniel Carbono

{ref}`Ver cuántas imágenes hay de cada clase <conteo-por-categoria>`.

¿Con qué precisión se puede delimitar un rótulo de peligro en una fotografía cuando las imágenes visualmente relacionadas se mantienen juntas? Este libro compara tres redes pequeñas entrenadas desde cero. La referencia para la siguiente entrega es FCN, seleccionada por su Dice medio en validación. El test interno cuantifica su desempeño sobre grupos reservados.

![Flujo desde las imágenes anotadas hasta la comparación de las quince corridas](figuras/flujo_experimento.png)

Las tres ramas del diagrama tienen funciones distintas. Entrenamiento ajusta los pesos; validación elige qué pesos conservar; test mide el desempeño del checkpoint elegido. Cada grupo de fotos relacionadas sigue una sola rama.

```{tableofcontents}
```

## Resumen ejecutivo

```{include} ../docs/RESUMEN_EJECUTIVO.md
:start-line: 1
```

## Documentos de cierre

{download}`Resumen ejecutivo <../docs/RESUMEN_EJECUTIVO.md>` · {download}`Alineación con el profesor <../docs/ALINEACION_PROFESOR.md>` · {download}`Registro de auditoría y correcciones <../docs/HALLAZGOS_ENTREGABLE_1.md>` · {download}`Tareas pendientes de la segunda etapa <../docs/PENDIENTES_ETAPA_2.md>`.
