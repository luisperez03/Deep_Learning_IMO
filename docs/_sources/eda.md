# EDA y preparación

El {download}`cuaderno de EDA original <eda_original.ipynb>` conserva el análisis previo a la partición definitiva. Sus menciones a trabajos pendientes reflejan esa fecha. Aquí se integran sus diagnósticos con los manifiestos y máscaras usados en las corridas.

(conteo-por-categoria)=
## Cuántas imágenes contienen cada categoría

La tabla cuenta fotografías entre los **1.583 pares con máscara utilizable**. Una foto con dos rótulos de la misma categoría cuenta una vez en esa fila. Si contiene dos categorías diferentes, cuenta en ambas filas; por eso la suma de las filas puede superar 1.583. Las 102 imágenes sin máscara quedan fuera de este recuento.

```{include} conteo_clases.inc
```

La columna **Total con máscara** reúne entrenamiento (1.015 fotos), validación (218), test (218) y cuarentena (132). Para reconstruir los conteos se cruzó el campo `classes` de `resultados/inicio_voc/inventario.csv` con el manifiesto por ID de imagen y se cotejó cada total con `distribucion_clases.csv`. El fondo es la categoría negativa por píxel y queda fuera de las 13 categorías de rótulos.

## Cobertura, geometría y objetivo

### Las 13 categorías que aparecen en las máscaras originales

¿Qué cuenta como «etiqueta» para la red? En la galería, cada fotografía es un recorte real de entrenamiento. El título traduce el nombre de la categoría; debajo aparece el identificador exacto de la paleta VOC y de la imagen. Para el experimento binario, los píxeles anotados de cualquiera de estas 13 categorías se convierten en **1 (etiqueta)** y el fondo en **0**. Los nombres proceden del corpus y su equivalencia con divisiones IMO requeriría una revisión específica.

```{include} galeria_clases.inc
```

**Cómo se eligieron los ejemplos.** Para cada categoría se tomó la fotografía elegible de entrenamiento con mayor fracción anotada entre las que tienen solo esa categoría. El recorte abarca sus píxeles y añade un margen de contexto. Así se aprecia el pictograma; la galería sobrerrepresenta deliberadamente los rótulos grandes. Los gráficos siguientes cuantifican el tamaño y el desbalance en las imágenes completas.

### Cuánto espacio ocupa una etiqueta

![Fotografía de entrenamiento, regiones de dos categorías y unión binaria de las máscaras](figuras/objetivo_binario.png)

En `image_1165`, la máscara original distingue sólido inflamable y radiactivo. La columna central usa dos colores de visualización para hacer legible esa diferencia. A la derecha se muestra la máscara binaria guardada que recibió el modelo. Conserva las dos regiones, incluso cuando ya no se puede deducir la clase a partir de sus valores.

La función que construyó la máscara comprueba la paleta antes de asignar 1 a los píxeles anotados. El valor RGB empaquetado cero corresponde al fondo negro.

```{literalinclude} ../scripts/preparar_base.py
:language: python
:pyobject: binary_mask
```

![Proporciones de fondo y etiqueta en los pares originales](figuras/08_balance_pixeles.png)

El fondo domina el recuento. Un clasificador que declarase fondo en todos los píxeles alcanzaría una exactitud alta y perdería el rótulo entero. Se eligió Dice de etiqueta como métrica primaria y se combinó Dice suave con BCE en la pérdida.

**De los conteos al porcentaje.** `eda_entregable.py` lee los píxeles por categoría de los 1.583 pares válidos en `resultados/inicio_voc/resumen.json`. La variable `fg` es la fracción de píxeles de etiqueta; multiplicarla por 100 produce el 4,38 % reportado.

```{literalinclude} ../scripts/eda_entregable.py
:language: python
:lines: 31-33
:dedent: 4
```

![Tamaño relativo de etiquetas en las imágenes](figuras/07_objetos_pequenos.png)

Las imágenes con área positiva pequeña son especialmente sensibles a la reducción espacial. La medida se refiere a la suma de área etiquetada por imagen. Más abajo se cuantifican también las regiones individuales que llegan al modelo.

![Relaciones de aspecto del corpus](figuras/05_aspecto.png)

Los encuadres tienen distintas proporciones. Se preservó la relación de aspecto con redimensionado y relleno centrado. Imagen y máscara comparten la geometría; la imagen usa interpolación bilineal y la máscara, vecino más cercano. Los píxeles de relleno se excluyen del entrenamiento y la evaluación.

## Calidad de las anotaciones

![Fotografía, máscara y revisión de cobertura del caso image_1461](figuras/revision_1461.png)

En la cesta se distingue un rótulo aparentemente omitido mientras dos rótulos inferiores sí tienen máscara. Esta clase de error persiste al binarizar. Se puso el grupo en cuarentena. En `image_0025` la discrepancia parece semántica y en `image_1251` la mano oculta parte de las etiquetas. Los paneles se eligieron para diagnóstico, por lo que su frecuencia de problemas no estima una tasa de error poblacional.

De los 1.685 JPG, 102 carecen de máscara utilizable. Los 102 figuran sin asignación en las particiones heredadas y con origen XML desconocido. Entre los 1.583 pares con máscara solo uno tiene origen desconocido. La ausencia se concentra en un subconjunto identificable; estos metadatos no bastan para explicar la causa. Se excluyeron de la supervisión, sin convertirlos en ejemplos de fondo. En 69 pares no coinciden los conjuntos de clases XML y máscara. También falta el XML de `image_1018`. La cuarentena incluye alertas de cobertura inferior al 5 % en cajas y se propaga dentro del grupo. No existe una segunda anotación independiente para estimar acuerdo o un techo de Dice humano.

## Similitud, procedencia y fugas

![Pares de fotografías con firmas perceptuales próximas](figuras/11_similitud.png)

De 1.418.770 pares examinados, 748 cumplieron simultáneamente distancias dHash y aHash de hasta 6 bits sobre 64. De ellos, 146 cruzaban entrenamiento y validación heredados, aunque no se encontraron JPG ni matrices RGB exactamente iguales. Doce pares revisados mostraron composiciones casi coincidentes. La decisión fue mantener juntas las componentes conexas de todos los enlaces candidatos. En la nueva asignación cruzan particiones **cero enlaces conocidos**. Rotaciones y recortes podrían escapar a estos hashes.

En la validación heredada, 33 imágenes provenían de la carpeta XML `00_javad` y 151 de `00_misaq`. Una carpeta puede compartir fondos con otra. Esta concentración y la falta de identificadores certificados de cámara o escena limitan la interpretación del origen. Un grupo visual aproxima dependencia sin garantizarla.

```{include} eda_calculos.inc
```
