| Red | Dice | IoU | mIoU | Exactitud | DE semillas | IC 95 % por grupos (Dice) |
| --- | --- | --- | --- | --- | --- | --- |
| FCN | 0,8531 | 0,7673 | 0,8783 | 0,9899 | 0,0063 | 0,8281 a 0,8762 |
| U-Net | 0,8451 | 0,7612 | 0,8747 | 0,9889 | 0,0135 | 0,8161 a 0,8716 |
| SegNet | 0,8386 | 0,7504 | 0,8691 | 0,9884 | 0,0353 | 0,8124 a 0,8634 |

## Diferencias pareadas

![Diferencias medias con sus intervalos pareados del 95 % por grupos y el margen práctico positivo](figuras/intervalos_pareados.png)

El punto es la diferencia media; la barra es el IC del 95 % al remuestrear los mismos grupos con las cinco semillas fijas. A la derecha de cero, la media favorece a la primera red. Para FCN frente a U-Net, la barra completa queda antes de la línea de +0,02. Para U-Net frente a SegNet, la barra atraviesa cero.

| Primer modelo menos segundo | Δ Dice | IC 95 % grupos | d por grupo | IC 95 % de d | P(superior) | p Shapiro (semillas) | p exacto | p Holm | MMD aprox. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCN − U-Net | 0,0079 | 0,0006 a 0,0153 | 0,1898 | 0,0149 a 0,3702 | 0,4836 | 0,7569 | 0,4375 | 1,0000 | 0,0262 |
| FCN − SegNet | 0,0144 | 0,0045 a 0,0244 | 0,2559 | 0,0819 a 0,4393 | 0,6230 | 0,4008 | 0,8125 | 1,0000 | 0,0555 |
| U-Net − SegNet | 0,0065 | -0,0027 a 0,0147 | 0,1303 | -0,0438 a 0,3880 | 0,6885 | 0,4276 | 0,4375 | 1,0000 | 0,0401 |

Contraste global de Friedman por semilla: Q = 1,6000, p exacto = 0,5216 al permutar los rangos de las tres redes dentro de cada uno de los cinco bloques (7.776 asignaciones bajo intercambiabilidad).

### Contrastes sobre grupos con semillas fijas

Para la pregunta de generalización a otros grupos, se contrasta el centro de las diferencias de Dice de los 122 grupos, promediadas sobre las mismas cinco semillas. Wilcoxon bilateral supone independencia entre grupos y simetría de sus diferencias bajo la hipótesis nula. Se obtiene la distribución exacta por asignación de signos a los rangos absolutos, omitiendo ceros y promediando empates. Holm se aplica a esta familia de tres contrastes. La MMD para grupos usa la misma aproximación normal de planificación con la DE de las diferencias por grupo y 122 bloques. Un IC para una media y un contraste de rangos pueden dar lecturas diferentes si las diferencias son asimétricas.

| Contraste | p exacto (grupos) | p Holm (grupos) | MMD aprox. (grupos) |
| --- | --- | --- | --- |
| U-Net − SegNet | 0,0001 | 0,0002 | 0,0146 |
| FCN − SegNet | 0,0006 | 0,0012 | 0,0165 |
| FCN − U-Net | 0,4616 | 0,4616 | 0,0122 |

![Cuantiles de las cinco diferencias pareadas frente a cuantiles normales](figuras/qq_semillas.png)

Los extremos de cada gráfico dependen de apenas una semilla. Con cinco puntos, Shapiro–Wilk y el gráfico Q–Q tienen poca capacidad diagnóstica. Se eligió la prueba bilateral exacta de rangos con signo, que supone simetría de las diferencias entre semillas y evita estimar una varianza común entre redes.

## Presupuesto observado por corrida

| Red | Semilla | Parámetros | Épocas | Mejor época | Dice val. | Dice test | CPU (min) | T4 (min) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCN | 11 | 72145 | 30 | 26 | 0,8331 | 0,8609 | 40.4 | 0.0 |
| FCN | 23 | 72145 | 28 | 23 | 0,8489 | 0,8569 | 37.2 | 0.0 |
| FCN | 37 | 72145 | 30 | 25 | 0,8279 | 0,8493 | 0.0 | 14.1 |
| FCN | 53 | 72145 | 18 | 13 | 0,8086 | 0,8448 | 0.0 | 8.3 |
| FCN | 71 | 72145 | 29 | 24 | 0,8462 | 0,8533 | 0.0 | 13.5 |
| U-Net | 11 | 118273 | 26 | 21 | 0,8274 | 0,8477 | 91.4 | 0.0 |
| U-Net | 23 | 118273 | 19 | 14 | 0,8133 | 0,8267 | 0.0 | 8.8 |
| U-Net | 37 | 118273 | 30 | 29 | 0,8243 | 0,8456 | 0.0 | 13.9 |
| U-Net | 53 | 118273 | 30 | 29 | 0,8429 | 0,8643 | 0.0 | 13.9 |
| U-Net | 71 | 118273 | 23 | 18 | 0,8282 | 0,8414 | 0.0 | 11.1 |
| SegNet | 11 | 85761 | 30 | 25 | 0,8238 | 0,8404 | 68.0 | 0.0 |
| SegNet | 23 | 85761 | 11 | 6 | 0,7494 | 0,7813 | 2.3 | 4.7 |
| SegNet | 37 | 85761 | 28 | 23 | 0,8237 | 0,8379 | 0.0 | 13.0 |
| SegNet | 53 | 85761 | 25 | 20 | 0,8468 | 0,8592 | 0.0 | 11.6 |
| SegNet | 71 | 85761 | 30 | 27 | 0,8529 | 0,8744 | 0.0 | 14.9 |

Se sumaron los segundos registrados por época completa. La migración de SegNet con semilla 23 conserva la parte local del historial. Los totales excluyen interrupciones y transferencias. Las diferencias de duración mezclan arquitectura, hardware y parada temprana.

Dice medio de los mejores checkpoints de validación por red: FCN 0,8330, U-Net 0,8272, SegNet 0,8193.

Huella SHA-256 del manifiesto de test: `7340a0cab140a8180b4a669260809eb79e86be7cb43ef87f5cac9bbd22317e81`.

## Aciertos y errores

Se fijó FCN con semilla 11 y se eligieron tres posiciones en su distribución de Dice por imagen. Son ejemplos descriptivos escogidos después del test. En un empate se usa el identificador como desempate.

| Posición | Imagen | Dice de FCN |
| --- | --- | --- |
| Mínimo | image_1656 | 0,0339 |
| Central (posición 110) | image_1241 | 0,9162 |
| Máximo | image_1118 | 0,9814 |

La selección depende de FCN. Las mismas imágenes permiten comparar las otras dos redes, pero no representan necesariamente sus propios cuantiles. Un caso central por rango puede tener Dice alto si la distribución es asimétrica.
