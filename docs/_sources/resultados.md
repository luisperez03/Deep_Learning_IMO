# Resultados y baseline

**FCN es la referencia de la entrega.** Obtuvo el mayor Dice medio en validación (0,8330), seguida de U-Net (0,8272) y SegNet (0,8193). En test, su ventaja media sobre U-Net es 0,0079 Dice. El intervalo pareado por grupos va de 0,0006 a 0,0153 con las cinco semillas fijas. Su extremo superior sigue por debajo del margen práctico propuesto de 0,02.

## Qué se estima

La {ref}`tabla de imágenes por clase y partición <conteo-por-categoria>` da el contexto para interpretar estas métricas binarias.

Dice mide $2TP/(2TP+FP+FN)$ e IoU mide $TP/(TP+FP+FN)$ para la etiqueta. La mIoU promedia IoU de etiqueta y fondo. La exactitud mide la fracción de píxeles válidos correctamente clasificados. Dos máscaras vacías reciben Dice e IoU de etiqueta iguales a 1; si solo una está vacía, reciben 0.

Las tablas se regeneran a partir de las 3.270 filas de `por_imagen.csv`. Se comprueba la correspondencia de cada identificador con su grupo congelado. En cada arquitectura y semilla se promedian primero las imágenes del mismo grupo y después los 122 grupos con igual peso. Se promedian las cinco semillas y se cotejan las métricas con `por_corrida.csv` y `resumen.csv`.

**Del CSV al Dice de la tabla.** En `analisis_entregable.py`, `blocks[m, s, g]` reúne las métricas de las imágenes del modelo `m`, la semilla `s` y el grupo `g`. El primer promedio representa a cada grupo; el segundo da a cada grupo el mismo peso. El Dice final de FCN es la media de sus cinco valores de corrida.

```{literalinclude} ../scripts/analisis_entregable.py
:language: python
:lines: 77-80
:dedent: 4
```

**Dos fuentes de incertidumbre.** El IC percentil del 95 % remuestrea 122 grupos con reemplazo 10.000 veces (semilla 20261107). Cada selección conserva juntas las imágenes del grupo, todas sus semillas y los tres modelos. Estos intervalos son marginales para cada contraste y condicionales a las cinco semillas entrenadas. La DE muestral entre las cinco medias por corrida describe por separado la variación de entrenamiento en el mismo test. No se cuentan las 3.270 filas como réplicas independientes.

Para los contrastes de semillas se emparejan corridas de igual semilla sobre el mismo test. Se aplica Wilcoxon bilateral exacto y se ajustan los tres valores p con Holm. El tamaño de efecto $d$ es la diferencia media por grupo dividida por la DE muestral de esas diferencias, tras promediar las cinco semillas. Su IC repite el cálculo completo en cada remuestra de grupos. P(superior) es la fracción de grupos en los que gana la primera red, contando medio punto por empate.

**Del par de redes a la barra del gráfico.** `d` contiene las 122 diferencias emparejadas de Dice y `draws` las 10.000 remuestras de grupos. Estos dos renglones calculan los extremos del intervalo dibujado.

```{literalinclude} ../scripts/analisis_entregable.py
:language: python
:lines: 107-108
:dedent: 8
```

```{include} tablas_resultados.inc
```

## Lectura de las predicciones

![Fotografía, referencia binaria y predicciones FCN, SegNet y U-Net para los tres casos preseleccionados](figuras/comparacion.png)

De arriba abajo aparecen los casos de Dice alto, central y bajo de FCN. La primera columna muestra la fotografía original adaptada al lienzo de evaluación de 256 × 256; la segunda, la anotación disponible en esas coordenadas. El blanco representa etiqueta. Las bandas negras de relleno se excluyen de los conteos.

![Aciertos positivos en verde, falsos positivos en naranja y falsos negativos en azul](figuras/errores.png)

El verde indica verdaderos positivos, el naranja falsos positivos y el azul falsos negativos **respecto de la anotación**. En `image_1118`, las tres redes cubren casi toda la etiqueta. En `image_1241`, SegNet localiza ambos rótulos, pero también activa regiones fuera de ellos (316 FP frente a 68 de FCN). En `image_1656`, U-Net marca el letrero «NO EXIT» y omite los 730 píxeles del rombo anotado. FCN recupera solo 17 y SegNet 54. Un modelo puede segmentar una región con bordes nítidos y, aun así, localizar el objeto equivocado.

![Contorno anotado en azul, predicho en naranja y coincidencia en blanco](figuras/contornos.png)

El azul delinea la referencia, el naranja la predicción y el blanco los píxeles de borde compartidos. Los trazos son el borde interior de un píxel obtenido por erosión de 3 × 3 dentro del área válida. En el ejemplo alto predominan discrepancias pequeñas cerca del perímetro. En el difícil, referencia y predicción de U-Net están separadas espacialmente: atribuir ese fallo únicamente a imprecisión del borde sería incorrecto. Estas láminas no constituyen una medición de Boundary IoU.

```{include} conteos_panel.inc
```

## Discusión, limitaciones y conclusiones

La ventaja de FCN sobre U-Net tiene signo positivo en promedio, aunque FCN gana en menos de la mitad de los grupos. Una ganancia mayor en ciertos grupos puede compensar pérdidas menores en otros. Los IC por grupos cuantifican esa heterogeneidad con los modelos ya entrenados. Los contrastes por semilla responden a otra pregunta: con cinco pares, el menor p bilateral exacto posible es 0,0625, incluso antes de Holm. El resultado por semillas no demuestra superioridad reproducible ni equivalencia.

El protocolo previo propuso **δ = 0,02 de Dice absoluto**. Al nivel de FCN, subir de 0,8531 a 0,8731 reduciría el déficit respecto de Dice 1 de 0,1469 a 0,1269, aproximadamente un 13,6 %. Este umbral de trabajo exige una mejora que compense la complejidad de un modelo posterior. No hay calibración en incidentes portuarios, acuerdo entre anotadores o costos de omitir un rótulo que lo convierta en umbral de seguridad. Solo la estimación puntual de la diferencia FCN–SegNet se acerca a dos puntos y su intervalo por grupos cruza ese margen.

En los grupos con semillas fijas, el contraste de rangos entre FCN y U-Net arroja p ajustado 0,4616. Frente a SegNet, FCN y U-Net tienen p ajustados 0,0012 y 0,0002 respectivamente. La tabla por grupos también muestra que el intervalo para la **diferencia media** U-Net frente a SegNet incluye cero. Los rangos dan peso al orden de las diferencias, mientras que la media conserva su magnitud. Ninguno de estos contrastes por grupos establece estabilidad frente a nuevos entrenamientos.

La columna MMD es una **aproximación de planificación** para la variación entre semillas: $(z_{1-0,05/6}+z_{0,80})s_d/\sqrt{5}$, con desviación de las diferencias pareadas y ajuste conservador para tres contrastes bilaterales. Para FCN frente a U-Net da 0,0262 Dice, mayor que δ. Esta aproximación normal con cinco semillas es inestable. No constituye la potencia del Wilcoxon exacto, que con este diseño no puede rechazar al 5 %. Para un futuro modelo, la MMD dependerá de su propia variación pareada frente a FCN.

SegNet presenta la mayor dispersión entre semillas. Su peor corrida coincide con la migración de hardware; este diseño no separa un efecto del equipo de un efecto de la semilla. La agrupación visual puede omitir dependencias y las máscaras tienen cobertura imperfecta. Con estas restricciones, FCN ofrece el mejor punto de referencia observado, con menos parámetros que las otras dos redes. Una mejora futura debe contrastarse frente a esa misma referencia y valorar tanto su incertidumbre como el margen práctico.
