En los 1.583 pares originales, la etiqueta ocupa 4.38 % de los píxeles y el fondo 95.62 %. La razón fondo/etiqueta es 21.8 y la entropía binaria es 0.260 bits por píxel.

## Regiones que llegan al modelo

En las 1.015 máscaras de entrenamiento a 256 × 256, la mediana de componentes por imagen es 1. Hay 1709 regiones de ocho vecinos. Los percentiles 10, 50 y 90 de área son 219.0, 719.0, 2943.8 píxeles. La razón mediana entre el lado largo y el corto de su caja envolvente es 1.07. Una componente puede reunir rótulos en contacto o dividir uno ocluido. El área total de etiqueta nunca baja de 16 píxeles en las máscaras de entrenamiento a esta resolución, pero componentes individuales pueden ser menores. Esto sustenta conservar el encuadre y medir solapamiento de la clase pequeña.

![Histogramas de RGB por clase en entrenamiento](figuras/rgb_train.png)

Las intensidades de etiqueta y fondo se solapan en los tres canales. Un umbral de color único perdería parte de ambas distribuciones. Para normalizar se usaron exclusivamente píxeles válidos de train, con medias RGB 0,604/0,561/0,507 y desviaciones 0,233/0,227/0,240.

## Comparación de las particiones congeladas

| Partición | Imágenes | Grupos | Mediana fracción etiqueta | Ancho mediano | Alto mediano | Brillo medio por grupo |
| --- | --- | --- | --- | --- | --- | --- |
| train | 1015 | 567 | 0,0317 | 520 | 750 | 0,5449 |
| val | 218 | 122 | 0,0285 | 520 | 749 | 0,5453 |
| test | 218 | 122 | 0,0267 | 520 | 750 | 0,5287 |

El test tiene menor fracción mediana de etiqueta que train. Para el brillo de miniaturas se compararon las medias de grupo entre las tres particiones. El estadístico es la suma de desviaciones cuadráticas de cada media de partición respecto de la global, ponderadas por el número de grupos. Una permutación exploratoria de etiquetas de partición entre grupos, conservando sus tamaños en número de grupos (4.999 permutaciones, semilla 20261107), da p = 0,5068. El diagnóstico cubre un atributo y presupone intercambiabilidad de grupos. Ni un brillo similar ni el resultado de esta prueba certifican igualdad de dominio. Se conservaron las particiones originales del experimento.
