# Inferencia complementaria: equivalencia, varianza, supuestos y sesgo del ganador

> **Trazabilidad.** Generado el 22/11/2026. Exploratorio y posterior al test: nada de esto se especificó antes de
> observarlo. Scripts: `scripts/inferencia_cierre_e2.py` (Python, numpy/scipy) y `scripts/varianza_mixta_val_e2.R`
> (R 4.5.0, lme4 2.0.1, lmerTest). Datos: los Dice por imagen de validación de las 50 corridas
> (`resultados/validacion_por_imagen_e2/`) y, para test, solo los resúmenes de `resultados/integracion_e2/analisis.json`
> (los Dice por grupo del test no están en el repositorio). Salidas: `inferencia.json`, `TABLAS.md`,
> `varianza_mixta.json`, `varianza_mixta_diagnostico.png`, `varianza_mixta.log`. Semillas de los generadores fijas
> (20261122; el intercambio del libro, 20261120).

Comandos, desde la raíz del proyecto:

```
python scripts/inferencia_cierre_e2.py
"C:\Program Files\R\R-4.5.0\bin\Rscript.exe" scripts/varianza_mixta_val_e2.R 200
```

## Qué es validación y qué es test

- **Test**: un TOST con t pareado exacto. Del libro se tienen, por contraste red − FCN, la media de la diferencia por
  grupo, d_z = media/DE y n = 122 grupos; de ahí DE = media/d_z y el TOST sale sin aproximación adicional. No hay
  bootstrap, ni simetría, ni max-T en test, porque necesitan los 122 valores por grupo.
- **Validación**: es el conjunto con el que se eligió el checkpoint de cada corrida. Los niveles están sesgados al alza
  y las diferencias pueden estarlo. Se usa para lo que el test no permite sin sus grupos, y cada resultado se lee como
  validación.

Estimando, el mismo del libro: Dice medio de cada grupo (promedio de sus imágenes), promediado sobre las cinco
semillas; el contraste es la media sobre grupos de la diferencia pareada. δ = 0,02 Dice. TOST: dos pruebas
unilaterales con α = 0,05 cada una (equivale a que el IC90 quede dentro de ±δ); hay equivalencia si las dos rechazan.

## Equivalencia (fila 54)

**Test, red − FCN.**

| Contraste | Diferencia | IC90 t | p TOST | p TOST Holm(9) | Lectura |
|---|---:|---:|---:|---:|---|
| TransUNet − FCN | 0,0091 | [−0,0061; 0,0242] | 0,117 | 0,94 | ni equivalentes ni superior |
| U-Net − FCN | −0,0072 | [−0,0148; 0,0005] | 0,0032 | 0,028 | equivalentes con δ = 0,02 |
| VM-UNet − FCN | −0,0145 | [−0,0275; −0,0015] | 0,24 | 1 | no equivalentes |

Las otras seis redes quedan por debajo de −δ o lo cruzan (tabla completa en `TABLAS.md`).

La pregunta del libro (RQ3: ¿una red menor conserva el desempeño de TransUNet dentro de ±0,02?) tiene respuesta
negativa para FCN en test: el extremo superior del IC90 (0,0242) pasa de δ, así que no se puede descartar una ventaja
relevante de TransUNet; y como el IC95 bilateral incluye cero, tampoco se demuestra la ventaja. El dato es
indeterminado, no "empate".

**Validación**, con t y con bootstrap de grupos (10.000 réplicas), mismas decisiones en los dos:

- U-Net − FCN: 0,0011, IC90 [−0,0079; 0,0100]: equivalentes (p TOST Holm(9) 0,003 con t, 0,005 con bootstrap).
- Ninguna red es equivalente a TransUNet en validación: la más cercana, U-Net, está a −0,021 (IC90 [−0,033; −0,010]).
  En validación TransUNet supera a FCN por 0,022 (IC90 [0,009; 0,035]).

Supuesto del TOST t: normalidad de la media de 122 diferencias. Las diferencias son asimétricas (Shapiro p < 1e-6 en
las nueve); en validación el bootstrap, que no la necesita, da las mismas decisiones que t, lo que respalda usar t
en test.

## Descomposición de varianza (fila 53)

Modelo lineal mixto (REML) sobre los 10.900 Dice imagen × corrida de validación:

```
dice ~ red + (1|semilla) + (1|red:semilla) + (1|grupo) + (1|grupo:imagen) + (1|red:grupo)
```

Semilla es el efecto común de la semilla a todas las redes; red:semilla, lo propio de cada entrenamiento (la
corrida); red:grupo, que unas redes aciertan en escenas donde otras fallan; el residuo es semilla × imagen dentro de
la red. El término red:grupo:imagen se ajustó y quedó en el borde (varianza 0): con dos imágenes por grupo como máximo
y casi duplicadas, red:grupo lo absorbe.

| Componente | Proporción del total | IC95 bootstrap de grupos | INT (rangos) | Sin Dice = 0 |
|---|---:|---:|---:|---:|
| Grupo | 64,0 % | 53,0–72,2 % | 60,8 % | 65,9 % |
| Red × grupo | 15,1 % | 11,0–19,8 % | 10,5 % | 14,1 % |
| Residuo (semilla × imagen) | 15,2 % | 11,7–21,1 % | 15,4 % | 14,3 % |
| Red (fijo) | 3,8 % | 2,5–5,7 % | 4,5 % | 3,8 % |
| Imagen dentro del grupo | 1,2 % | 0,4–2,3 % | 7,9 % | 1,2 % |
| Corrida (red × semilla) | 0,64 % | 0,49–1,11 % | 0,82 % | 0,66 % |
| Semilla común | 0,07 % | 0,00–0,21 % | 0,03 % | 0,07 % |

Bootstrap: 200 réplicas de grupos completos, las 200 válidas, IC percentil. La única diferencia de peso entre
escalas es la imagen dentro del grupo (1,2 % en Dice, 7,9 % en rangos): sobre rangos, las dos imágenes de un grupo se
separan más porque la transformación estira las diferencias entre Dice altos.

- El grupo (la escena) explica casi dos tercios de la variación; la red, menos de un 4 %; la semilla y la corrida,
  menos de un 1 % entre las dos. La semilla común no es distinguible de cero (LRT p = 0,15); la corrida sí
  (p ≈ 1e-54) pero es pequeña. La interacción red × grupo (15 %) es tan grande como el residuo: es la razón de
  emparejar por grupo.
- Efecto fijo de red: F(9; 123,6) = 15,9 (Satterthwaite), p ≈ 5e-17. En validación, no confirma nada en test.
- **Diagnóstico.** Residuos con asimetría −1,17 y exceso de curtosis 12,3; el 2,3 % pasa de 3 DE; la dispersión
  cae con el ajustado (Spearman |res| vs ajustado −0,55): el Dice está acotado y tiene 96 ceros. Gráficos en
  `varianza_mixta_diagnostico.png`. Por eso las proporciones se contrastan con dos alternativas (sobre la
  transformación normal inversa de rangos y sin los ceros) y con el bootstrap de grupos, que no supone normalidad.
  El reparto se mantiene: los IC y las proporciones son de la escala del Dice, no de un modelo correcto.

## Supuestos y sensibilidad (fila 51)

En validación, con las nueve diferencias red − FCN:

- **Simetría** (lo que necesita Wilcoxon para leerse como desplazamiento): bootstrap simetrizado alrededor de la
  mediana con el estadístico de Miao, Gel y Gastwirth. Se rechaza en 4 de 9 (TransUNet p = 0,008, SegNet 0,005,
  SegNeXt 0,0004, SegFormer 0,0002). En esas cuatro la prueba del signo, que no la necesita, da la misma decisión que
  Wilcoxon.
- **Cuatro procedimientos con Holm(9)**: t, intercambio (media), Wilcoxon y signo (localización). Coinciden en 6 de 9.
  TransUNet: la media rechaza (p Holm 0,02) y la localización no (Wilcoxon 0,066, signo 0,26): media 0,022, mediana
  0,004; la ventaja se concentra en pocos grupos. VM-UNet y DeepLabv3+: al revés; pérdidas pequeñas y frecuentes que
  el signo detecta y la media no.
- **Influencia**: dejar fuera un grupo o una semilla no cambia ninguna decisión t sin ajuste, salvo DeepLabv3+ (en el
  borde: 1 grupo y 2 de 5 semillas). La ventaja de TransUNet va de 0,020 a 0,027 al quitar cada semilla.
- **Dependencia entre grupos**: autocorrelación de lag 1 de las diferencias ordenadas por grupo, entre −0,03 y 0,07;
  p por permutación Holm = 1 en las nueve. Solo mira grupos vecinos en la numeración del corpus: no certifica
  independencia.
- **En test** (desde el libro): t pareado (p calculado con d_z), intercambio y Wilcoxon, con Holm(9), coinciden en 8
  de 9; difiere U-Net++ (Wilcoxon no rechaza), el mismo patrón de asimetría.

## Multiplicidad (fila 50)

Además de Holm por procedimiento, en validación: max-T por cambio de signo (Westfall y Young, paso único, 49.999
réplicas), que respeta la correlación entre los nueve contrastes, e IC95 simultáneos por bootstrap del máximo |t|
(cuantil 2,75). Rechazan las mismas redes que t-Holm: TransUNet (p 0,027, IC simultáneo [0,0009; 0,0438]), SegNet,
PIDNet, SegNeXt y SegFormer. La familia se fijó después del test: esa desviación está en B12 y no se repara aquí.

## Sesgo del ganador (B15)

- **Elegir la red en validación.** TransUNet gana en el 99,9 % de los bootstrap de grupos; el sesgo del máximo como
  estimador de su nivel es 0,0001. Eligiendo en una mitad de los grupos y midiendo en la otra (2.000 divisiones) el
  optimismo es −0,0004 en el nivel y 0,0003 en la ventaja sobre FCN. Con un ganador tan claro, la selección entre
  redes no infla nada visible.
- **Elegir el checkpoint en validación.** Contra el test (independiente del checkpoint, aunque ya observado): test −
  validación es +0,010 de media y 8 de 10 redes suben en test. No aparece el optimismo que se esperaría; si existe,
  queda tapado por la diferencia de dificultad entre los dos conjuntos. Separar las dos cosas pediría el Dice por
  imagen en cada época, que no está en el repositorio.
- **Semillas.** No hay selección: se promedian las cinco.
- **El contraste.** TransUNet − FCN pasa de 0,022 en validación a 0,009 en test (retiene el 41 %). Es lo esperable
  de una diferencia estimada en el conjunto que eligió los checkpoints; por eso la lectura principal sigue siendo la
  del test.

## Lo que no se pudo hacer aquí

- Bootstrap, simetría y max-T en test: necesitan los Dice por grupo del test (`resultados_test_e2_final`), fuera del
  repositorio.
- Separar el optimismo del checkpoint de la diferencia entre validación y test: necesita el Dice por imagen por época.
- Certificar la independencia entre grupos: los grupos son de similitud aproximada.
