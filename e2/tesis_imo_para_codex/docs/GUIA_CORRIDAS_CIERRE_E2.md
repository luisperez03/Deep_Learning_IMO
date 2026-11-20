# Cierre del Entregable 2: qué correr antes del domingo, en qué orden

> **Trazabilidad.** Escrito el 2026-11-20 a partir de `docs/COMPLEMENTO_GPU_LEEME.md`,
> `docs/COMPLEMENTO_ENTRENAMIENTO_LEEME.md`, `docs/REPLICA_SEGFORMER_LEEME.md` y del checklist maestro. Ningún
> cuaderno de esta guía se ha ejecutado en Colab: se probaron en CPU las partes que no necesitan GPU, Drive ni
> descargas. Todos los tiempos son **estimados**; cada cuaderno trae una celda piloto que los mide antes de lanzar
> el resto. El test ya fue observado: todo lo que sale de aquí es exploratorio y se reporta así.

## Prioridad por lo que cierra en el checklist

| Orden | Cuaderno | GPU | Estimado | Filas que puede cerrar |
|---|---|---|---|---|
| 1 | `colab/Replica_SegFormer_ADE20K.ipynb` | T4 | 15–30 min | 32 (réplica del paper guía), B19 |
| 2 | `colab/Complemento_GPU_E2.ipynb` | L4 | lo mide su piloto | 57, 58, 59, 60, 62, 64; B16, B17 |
| 3 | `colab/Complemento_Entrenamiento_E2.ipynb`, ablación | L4 | ≈ 10 h (hasta 12,8 h) | 61 |
| 4 | Mismo cuaderno, sobreajuste y permutación | L4 | 0,25 h + 6,5–33 h (FCN, B = 200) | 41, 56; B03 |
| 5 | Mismo cuaderno, búsqueda igual, T = 4 | L4 | 2,3–5,8 h | 37; B06 |
| 6 | Extracción del Dice por imagen de validación y limitación por área | CPU de Colab | lectura de 50 checkpoints | 33 |

La réplica es la más barata y cierra un requisito central: va primero. Los bloques 3 a 5 compiten por las mismas
horas de L4; si no caben todos, el orden de la tabla es el de prioridad. Las sesiones de Colab se cortan: todos
los cuadernos son reanudables por tarea o por corrida, y tras un corte se repiten las celdas de entorno y se
relanza la misma etapa.

## 1. Réplica de SegFormer-B0 en ADE20K

1. Abrir `colab/Replica_SegFormer_ADE20K.ipynb` con T4.
2. Leer los términos de ADE20K y, si se aceptan, poner `ACEPTO_TERMINOS_ADE20K = True` en la celda 1.
3. Correr en orden: entorno, descarga verificada por hash (unos 967 MB), checkpoint oficial con revisión fijada,
   evaluación a escala única, IC bootstrap por imagen, veredicto contra el criterio de
   `config/replica_segformer.json` (fijado antes de cualquier resultado).
4. Revisar: el mIoU y su IC frente a 37,4; el veredicto se reporta sea cual sea.

## 2. Diagnósticos con los 50 checkpoints congelados

Necesita en Drive el ZIP de test cerrado y el paquete v2 (el cuaderno verifica sus SHA-256) y la carpeta de las
50 corridas.

| Celda | Qué hace |
|---|---|
| 1–3 | Drive, paquete verificado, entorno aislado (Python 3.11, torch 2.5.1+cu121, Mamba para VM-UNet) |
| 4 | **Piloto**: las tres etapas con la semilla 11 y pocas imágenes; escribe `PILOTO_TIEMPOS.json` |
| 5 | `clean`: Dice de train, validación y test de las 50 corridas; calibración y bordes; probabilidades del top 3 |
| 6 | `robustness`: 450 tareas; se puede repartir por modelos entre sesiones |
| 7 | `interpret`: Integrated Gradients, fidelidad y estabilidad en el top 3 por validación |
| 8–9 | Resumen JSON (marca `partial` si falta una etapa) y descarga de JSON y logs |

Revisar: que `clean` reproduzca los Dice de test cerrados (con otra GPU no tienen por qué ser idénticos bit a
bit), la cobertura conforme por clase y la fidelidad con IC.

## 3 a 5. Entrenamientos complementarios

Abrir `colab/Complemento_Entrenamiento_E2.ipynb` con L4 y el paquete v2. Las celdas 1–6 preparan el entorno igual
al del benchmark; la celda piloto mide una época de corridas reales y con eso se fijan `B` (permutaciones) y `T`
(intentos de la búsqueda) **antes** de ver ningún resultado. Después, los bloques en el orden de la tabla.
Revisar en `RESUMEN.md`: la diferencia de cada ablación contra TransUNet completo con IC por grupos, s_obs frente
a la distribución permutada (no s_b frente a cero), y la posición de la receta del benchmark dentro de la búsqueda.

## 6. Limitación por área (fila 33)

```
python scripts/extraer_validacion_por_imagen.py --corridas <carpeta de las 50 corridas> --salida resultados/validacion_por_imagen_e2
python scripts/limitacion_area_segformer.py --por-imagen resultados/validacion_por_imagen_e2 --salida resultados/limitacion_segformer_area
```

El extractor verifica el SHA-256 de cada checkpoint y que la media por grupos reproduzca el Dice de validación
registrado.

## Lo que no es de GPU y solo pueden hacer personas

- **Segunda anotación de 50 imágenes** (filas 12 y 63, B13): única vía a un techo humano medido. Kit en
  `resultados/techo_e2/`; quien reanote no debe haber visto las máscaras originales; los rótulos cortados por el
  borde se marcan aparte. Luego `python scripts/techo_reanotacion_e2.py medir`.
- **Confirmar 12 casos** de la revisión de ruido (`resultados/ruido_etiqueta_e2/confirmar.csv`).
- **Ensayar la defensa** con `docs/PRESENTACION_E2.md`.

## Después de las corridas

1. Copiar los JSON de resumen a `resultados/` (sin NPZ ni pesos).
2. Volver a correr `scripts/actualizar_checklist_cierre.py` no sirve para esto: cada fila se pasa a C a mano, con
   la evidencia ejecutada y su ruta.
3. Agregar los resultados al capítulo de complementos del libro y compilar con
   `jupyter-book build libro -W --keep-going`.
