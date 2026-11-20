# Análisis complementario E2

**Exploratorio, especificado después de observar test.** Unidad: 122 grupos; cinco semillas fijas. No hay 610 réplicas independientes.
Se conserva el contraste cerrado previo. Esta ampliación no sustituye un preregistro anterior al entrenamiento.

Friedman por grupos: estadístico 321.48733; p = 7.1653448e-64.
El contraste global detecta heterogeneidad del conjunto; no demuestra que TransUNet supere a FCN.

| Modelo − FCN | Diferencia | IC95 | d_z (IC95) | p intercambio | p intercambio Holm(9) | p Wilcoxon Holm(9) |
|---|---:|---:|---:|---:|---:|---:|
| TransUNet | 0.00906 | [-0.00809, 0.02777] | 0.090 [-0.090, 0.255] | 0.33266 | 0.33266 | 0.204552 |
| U-Net compacta | -0.00718 | [-0.01631, 0.00190] | -0.141 [-0.309, 0.040] | 0.1257 | 0.2514 | 0.562787 |
| VM-UNet | -0.01449 | [-0.03004, 0.00080] | -0.167 [-0.340, 0.009] | 0.06578 | 0.19734 | 0.0934246 |
| SegNet compacta | -0.02886 | [-0.03901, -0.01932] | -0.517 [-0.659, -0.374] | 2e-05 | 0.00018 | 7.26313e-08 |
| DeepLabv3+ R18 | -0.03953 | [-0.06000, -0.02036] | -0.354 [-0.492, -0.207] | 6e-05 | 0.0003 | 8.26072e-06 |
| U-Net++ R18 | -0.04594 | [-0.07204, -0.02088] | -0.319 [-0.487, -0.147] | 0.00042 | 0.00168 | 0.0934246 |
| PIDNet-S | -0.05991 | [-0.08622, -0.03486] | -0.410 [-0.552, -0.261] | 2e-05 | 0.00018 | 7.86423e-07 |
| SegNeXt-T | -0.07435 | [-0.09565, -0.05374] | -0.626 [-0.765, -0.484] | 2e-05 | 0.00018 | 3.53343e-12 |
| SegFormer-B0 | -0.14827 | [-0.18053, -0.11755] | -0.828 [-0.980, -0.693] | 2e-05 | 0.00018 | 9.58156e-16 |

## Supuestos y alcance

- Intercambio pareado: 49.999 cambios de signo Monte Carlo, p con corrección +1, bilateral sobre la media. Requiere intercambiabilidad del par bajo la nula; no es una prueba exacta exhaustiva ni una permutación de etiquetas con reentrenamiento.
- Wilcoxon usa aproximación asintótica con corrección de empates y cero excluido; responde a localización/rangos bajo simetría, no exactamente a la misma hipótesis de media. Shapiro por diferencia se conserva en JSON; normalidad no acredita independencia.
- Friedman/Nemenyi usan grupos como bloques, condicionales a modelos ya entrenados. La independencia de grupos es aproximada; la similitud no identifica escenas certificadas. No extrapolar a otros datasets.
- Holm se aplica por separado a los nueve contrastes contra FCN para cada procedimiento. Los IC95 son marginales, no simultáneos. No elegir la prueba que dé el p más favorable.
- d_z estandariza diferencias por grupo; no es d entre semillas. La probabilidad de superioridad por grupo y su dispersión se conservan en JSON.
- δ heredado = 0,02 Dice. El IC del contraste principal incluye cero y se extiende fuera de ±δ; no acredita superioridad relevante ni equivalencia.

## Referencia trivial y recursos

Máscara siempre vacía: Dice=0.0000, IoU=0.0000, mIoU=0.4812, exactitud=0.9625. No sustituye un clásico fuerte ajustado.
El contraste frente a cero no prueba ausencia de fuga ni reemplaza la permutación completa de etiquetas.

| Modelo | Parámetros | Horas registradas (5 semillas) | Rango por grupos | Rango por semillas | Pareto Dice/parámetros |
|---|---:|---:|---:|---:|---|
| TransUNet | 105,322,001 | 3.461 | 3.180 | 2.00 | Sí |
| FCN compacta | 72,145 | 0.795 | 4.148 | 2.20 | Sí |
| U-Net compacta | 118,273 | 0.741 | 4.180 | 3.20 | No |
| VM-UNet | 44,273,833 | 2.905 | 4.926 | 4.20 | No |
| SegNet compacta | 85,761 | 0.534 | 5.516 | 5.00 | No |
| DeepLabv3+ R18 | 12,329,297 | 1.026 | 5.713 | 5.60 | No |
| U-Net++ R18 | 15,970,449 | 1.001 | 4.770 | 6.40 | No |
| PIDNet-S | 7,716,291 | 2.291 | 6.508 | 7.60 | No |
| SegNeXt-T | 4,225,121 | 2.043 | 7.533 | 8.80 | No |
| SegFormer-B0 | 3,714,401 | 1.599 | 8.525 | 10.00 | No |

El tiempo suma épocas de entrenamiento y validación; excluye parte de E/S, interrupciones e instalación. No es tiempo puro de entrenamiento ni latencia. Pareto se refiere a estimaciones puntuales, sin afirmar dominancia estadística.

## Potencia y sobreajuste

El JSON guarda DE pareada entre semillas, MDD con R=5 y R aproximado para δ=0,02, α/9 y potencia 0,80. Es planificación futura normal basada en un piloto pequeño, no potencia prospectiva ni potencia observada certificada.
Los historiales contienen pérdida de entrenamiento y Dice de validación: son escalas distintas. No restarlas para calcular una brecha ni afirmar ausencia de sobreajuste. Falta Dice de train en modo evaluación en el checkpoint elegido y un margen δg fijado previamente.

Discrepancia máxima al reconstruir la media principal desde los registros agrupados: 0. Esta es reproducción aritmética; no nueva inferencia en entorno limpio.

## Segmentos y errores

Cuartiles de fracción de área positiva definidos en train: [0.018086360733893254, 0.0317840576171875, 0.05170468634959915]. Resultados por estrato e IC remuestreando grupos dentro del estrato en analisis.json. Un grupo puede aparecer en varios estratos; no tratarlos como independientes.
La correlación de 1−Dice coincide con la de Dice y describe dificultad compartida. No es una medida de acuerdo humano ni demuestra ruido irreducible. Los subgrupos son descriptivos, no contrastes confirmatorios.
