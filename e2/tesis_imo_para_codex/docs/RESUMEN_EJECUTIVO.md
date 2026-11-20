# Resumen ejecutivo del Entregable 1

**Pregunta.** ¿Qué precisión de segmentación binaria alcanzan tres redes pequeñas para delimitar rótulos de mercancías peligrosas cuando se reservan grupos de fotografías visualmente relacionadas?

**Diseño.** FCN, SegNet y U-Net se entrenaron desde cero con cinco semillas comunes, tasa 0,001, máximo 30 épocas y paciencia 5. Cada corrida aportó el checkpoint de mayor Dice por grupo en validación. El test interno contiene 218 fotografías de 122 grupos. El análisis pondera por igual los grupos y las semillas; las 3.270 evaluaciones no son réplicas independientes.

| Red | Dice de validación | Dice de test | IoU de etiqueta | DE del Dice entre semillas | IC 95 % por grupos, semillas fijas |
|---|---:|---:|---:|---:|---|
| FCN | 0,8330 | 0,8531 | 0,7673 | 0,0063 | 0,8281 a 0,8762 |
| U-Net | 0,8272 | 0,8451 | 0,7612 | 0,0135 | 0,8161 a 0,8716 |
| SegNet | 0,8193 | 0,8386 | 0,7504 | 0,0353 | 0,8124 a 0,8634 |

**Conclusión.** FCN encabeza validación, tiene el menor número de parámetros (72.145) y ofrece la referencia para el siguiente benchmark. Su ventaja media frente a U-Net es 0,0079 Dice, con IC pareado de 0,0006 a 0,0153 por grupos y semillas fijas. El extremo superior está por debajo del margen de relevancia práctica propuesto de 0,02. Por semillas, Wilcoxon bilateral con Holm da p ajustado 1,0000. La MMD aproximada de planificación por semillas para este par es 0,0262. Estos resultados no establecen equivalencia formal ni superioridad estable en nuevos entrenamientos.

**Qué revelan las figuras.** En el ejemplo de alto Dice, las tres redes recuperan el rótulo y difieren principalmente en los bordes. En el caso central, SegNet produce 316 píxeles falsos positivos. En el difícil, U-Net marca el letrero «NO EXIT» y no coincide con ningún píxel del rombo anotado. Los casos describen modos de error; no estiman su frecuencia.

**Límites.** Las máscaras no tienen certificación individual ni segunda anotación sistemática. Los grupos aproximan dependencia visual. No se evaluaron instancias, clases IMO, rechazo o desempeño en otro puerto. Las conclusiones se restringen a este corpus y a las configuraciones cerradas.
