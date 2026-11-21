# Entrenamientos complementarios

**Exploratorio, posterior a la exposición del test.** Lo que exige volver a entrenar y el benchmark no cubría. Corrido
en una GPU L4 con `colab/Complemento_Entrenamiento_E2.ipynb` y `scripts/complemento_entreno_e2.py`, con el entorno y
el protocolo del benchmark (`config/complemento_entreno_e2.json`). Aquí no se carga el test: todo se mide en train o
en validación.

## Prueba de sanidad: sobreajustar un lote

Una implementación correcta debe poder memorizar un lote pequeño. Cada red se entrenó 500 pasos sobre el mismo lote
fijo de 4 imágenes de train, con el optimizador y la pérdida de su receta, y se midió el Dice de esas 4 imágenes en
modo evaluación. La regla, fijada antes de correr: pasa si el Dice medio del lote llega a 0,90; si no, se reporta tal
cual, sin cambiar la receta.

**Nueve de diez redes pasan**, con Dice del lote entre 0,967 (SegNeXt-T) y 1,000 (U-Net++). **VM-UNet no pasa**
(0,750). La tabla por imagen muestra dónde está el fallo:

```{include} ../resultados/complemento_entreno_e2/sobreajuste/TABLA.md
:start-line: 4
```

VM-UNet memoriza tres de las cuatro imágenes con Dice de 1,000 y en la cuarta, image_1231, no predice nada: Dice 0.
Su promedio, 0,750, es exactamente (1 + 1 + 0 + 1) / 4. image_1231 es la del rótulo más pequeño del lote, alrededor
del 1 % de una foto de 1000 × 1000, unos 25 × 25 píxeles tras reducirla a 256. Las otras nueve redes sí la aprenden
(de 0,924 en SegNeXt-T a 1,000).

**No es falta de pasos.** La pérdida de VM-UNet baja hasta el paso 150 y desde el 200 queda plana (0,141 → 0,137
en los 300 pasos siguientes). Las demás redes siguen bajando hasta el final. La red quedó en un mínimo local en el que
el rótulo pequeño se predice vacío.

**Lectura.** Una red mal conectada no memoriza tres imágenes con Dice de 1,000, así que el resultado no apunta a un
error de implementación. Apunta a una dificultad de VM-UNet con objetos pequeños o a una optimización inestable, que
encaja con su dispersión entre semillas en el test del benchmark (± 0,030, la segunda mayor de las diez después
de SegNet). Una explicación posible, no
probada, es que la reducción inicial de resolución deje un rótulo de ese tamaño en muy pocas celdas del mapa de
rasgos. No se repitió con más pasos ni con otra semilla: la regla se fijó antes y la curva ya muestra que más pasos
no cambian el resultado.
