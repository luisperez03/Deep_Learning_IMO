# Protocolo de la búsqueda de arquitectura de las redes compactas

> **Trazabilidad.** Fijado el 2026-11-22, antes de correr ninguna arquitectura del espacio. Parámetros en
> `config/busqueda_arquitectura_e2.json`; código en `scripts/busqueda_arquitectura_e2.py` (entrenamiento y
> decisiones entre fases) y `scripts/resumir_busqueda_arquitectura_e2.py` (análisis); cuaderno
> `colab/Busqueda_Arquitectura_E2.ipynb`, generado por `scripts/crear_colab_busqueda_arquitectura_e2.py`.
> Prueba funcional en CPU: `scripts/prueba_cpu_busqueda_arquitectura_e2.py` →
> `resultados/busqueda_arquitectura_e2/prueba_cpu.json`. Costo: `scripts/medir_costo_busqueda_arquitectura_e2.py` →
> `resultados/busqueda_arquitectura_e2/estimacion_costo_local.json`. Queda por verificar: los tiempos de L4, que
> mide el piloto del cuaderno.

## 1. Por qué hay búsqueda y para qué redes

La sección 14 de la guía obliga a buscar la arquitectura de todo modelo cuyo artículo no la define: espacio fijo,
declarado a priori, recorrido en dos fases. Las siete redes nuevas del benchmark (DeepLabV3+, UNet++, PIDNet-S,
SegNeXt-T, SegFormer-B0, TransUNet y VM-UNet) usan la configuración publicada y quedan fuera. Las tres compactas
heredadas (FCN, U-Net y SegNet, de 72.145, 118.273 y 85.761 parámetros, `scripts/modelos_base.py`) son
adaptaciones propias de tres niveles con anchos (16, 32, 64): su arquitectura no sale de ningún artículo, así
que entran en la búsqueda.

La compacta heredada está dentro del espacio: es la arquitectura (16, 32, 64) sin dropout. La prueba en CPU
comprueba que, con la misma semilla, la versión parametrizada tiene los mismos parámetros, la misma
inicialización y la misma salida que la de `modelos_base.py`.

## 2. Espacio (14.1 y 14.2)

- **Profundidad 3**: tres bloques de codificador (dos convoluciones 3×3 con ReLU), con max pooling entre ellos.
- **Anchos** W = {16, 32, 64, 128, 256}: `w_i` son los canales del bloque del nivel i. Las 125 combinaciones,
  sin restricción de forma.
- **Dropout** p = 0,2, `torch.nn.Dropout`, en las 8 combinaciones de posiciones `b ∈ {0,1}³`.
- **Cardinalidad**: 125 × 8 = 1000 por familia; 125 sin dropout y 875 con al menos uno. Identificadores
  a0000-a0999 en el orden de `itertools.product(W, W, W, {0,1}, {0,1}, {0,1})`. Listado con parámetros y MACs por
  familia en `config/arquitecturas_compactas_e2.csv` (FLOPs ≈ 2 × MACs).

Traducción por familia (fila de codificador-decodificador de la tabla 14.2):

| familia | estructura | dónde va el dropout `b_i` |
|---|---|---|
| FCN | codificador e1-e2-e3; cabezal 1×1 sobre e3 e interpolación bilineal | tras e_i (el de e3 queda antes del cabezal) |
| U-Net | decodificador espejo: d2 = bloque(w3 + w2 → w2), d1 = bloque(w2 + w1 → w1), concatenando el mapa del codificador | b1 tras e1 y su espejo d1; b2 tras e2 y su espejo d2; b3 tras e3, el cuello, sin espejo |
| SegNet | índices de pooling; reducción 1×1 w3 → w2; d2 = bloque(w2 → w1), d1 = bloque(w1 → w1) con max unpooling | igual que U-Net; b3 antes de la reducción |

La capa de salida nunca lleva dropout.

## 3. Lo que queda fijo (14.3)

Receta de las compactas en el benchmark (`config/experimento_e2.json`): Adam, weight decay 0, lote 2, 256 px,
pérdida 0,5 BCE + 0,5 Dice suave por imagen en la zona válida, umbral 0,5, float32 con TF32 de cuDNN y algoritmos
deterministas, inicialización por defecto tras `setup_seed`. Misma partición (1.015 train, 218 validación en 122
grupos), mismo preprocesamiento, mismo orden de lotes por semilla. El criterio es el Dice medio por grupo en
validación. **El test no se carga en ningún paso.**

**Ajuste previo de la tasa.** Con la referencia de la guía (64, 64, 64) y dropout (0, 1, 0), por familia, tasas
0,0003, 0,001 y 0,003, con el presupuesto y la fidelidad de la fase 1. Se queda la de mayor Dice de validación,
salvo que gane a la de la receta (0,001) por menos de 0,005; en ese caso queda 0,001. La tasa elegida es la de
las 1000 arquitecturas y la de la fase 2 de esa familia.

## 4. Fase 1: cribado exhaustivo (14.4, A1-A2)

- Las 1000 arquitecturas de cada familia, **todas**, sin poda previa.
- Semilla 97 para todas; 6 de 30 épocas (20 %), con la misma paciencia 5; métrica: mejor Dice de validación.
- Orden: una permutación fija de las 1000 (semilla 20261122), y para cada arquitectura las tres familias
  seguidas. Si hay que parar, lo hecho es una muestra aleatoria del espacio, igual para las tres.
- Registro por corrida: pérdida de entrenamiento, pérdida y Dice de validación por época, parámetros, MACs y
  segundos. No se guardan pesos.

**Fidelidad.** Si el cómputo no alcanza, la guía pide no recortar en silencio. Antes de ver el piloto se fija un
presupuesto de horas de L4 para ajuste + fase 1 + repetición; con el piloto se estima cada opción y se usa la
primera que cabe:

| opción | entrada de la red | train | cómputo de la red |
|---|---|---|---|
| A | 256 px (la del protocolo) | completo | 1 |
| B | 128 px por promedio de área; la salida vuelve a 256 y la pérdida y el Dice se miden a 256 | completo | 1/4 |
| C | como B | la mitad, elegida por grupos de similitud (semilla 20261123) | 1/8 |

Si ninguna cabe, se usa C y queda registrado. En todas las opciones las 1000 arquitecturas, la semilla y las 6
épocas son las mismas. La elección queda en `fidelidad.json` y el código se niega a cambiarla después de empezar
el ajuste. La fase 2 es siempre a fidelidad completa. B y C son una desviación declarada: el cribado mide la
arquitectura a otra escala, y su efecto se juzga con la correlación de rangos entre fases.

## 5. Ruido por semilla y número de finalistas (14.4, A3)

Las primeras 40 arquitecturas del orden (muestra aleatoria del espacio) se repiten con la semilla 98 en las tres
familias, intercaladas al principio de la fase 1. Con d la diferencia entre las dos semillas de una misma
arquitectura:

```
σ_semilla² = media(d²) / 2
σ_arq²     = varianza de las 1000 de la fase 1 − σ_semilla²   (truncada en 0)
r          = σ_semilla / σ_arq
```

**k = 10 si r ≤ 1 en las tres familias; k = 20 si r > 1 en alguna.** El mismo k para las tres. Con r ≤ 1 la
simulación de la guía da al menos un 50 % de probabilidad de que la mejor verdadera esté entre las 10; por encima,
se duplica k. Finalistas: los k mejores de la fase 1 (empate exacto: menos parámetros), más dos controles si no
están ya: la referencia (64, 64, 64)/(0, 1, 0) y la heredada (16, 32, 64)/(0, 0, 0).

La fase 1 es un cribado, no una afirmación de optimalidad: el ganador es el mejor entre los finalistas, no
necesariamente el mejor de las 1000.

## 6. Fase 2: confirmación (A4)

Cada finalista y los dos controles con las semillas 11, 23, 37, 53 y 71 (las del benchmark, distintas de 97 y 98),
30 épocas con paciencia 5, a 256 px y con todo train. Con la misma semilla el orden de lotes es el del
benchmark, así que las comparaciones se parean por semilla. Por corrida se registra el Dice de validación por
grupo en la mejor época, las predicciones por imagen de validación, el Dice de entrenamiento con los pesos de la
mejor época (brecha train-validación), parámetros, MACs y segundos, y se guarda el mejor checkpoint. Las tareas
van por semilla: un corte deja el mismo número de semillas en todos.

## 7. Selección y sesgo del ganador (14.5, A5)

Con δ = 0,02 de Dice, heredado del preregistro:

1. Mayor media de Dice de validación en las 5 semillas.
2. Empate técnico con la mejor: diferencia de medias menor que δ, o intervalos t al 95 % de las medias
   solapados. Entre las empatadas (incluida la mejor), gana la de menos parámetros; si empatan en parámetros,
   la de mayor media. Los controles compiten como cualquier finalista.

Se reporta además el conjunto de finalistas equivalentes a la mejor por TOST pareado por semilla (±δ, dos t
unilaterales al 5 %, 4 gl) y su tamaño. No hay pruebas entre pares de las 1000.

**Lo que no se hace.** El paso A6 de la guía (confirmar la ganadora y evaluarla una sola vez en test) no entra en
esta búsqueda: el test del benchmark ya fue observado, y las conclusiones sobre test quedan las del protocolo
cerrado. Tampoco hay validación anidada: el costo de repetir la búsqueda por pliegue no cabe. Los dos puntos se
declaran como limitación.

## 8. Comparación con la compacta heredada

Pregunta: ¿la arquitectura ganadora de cada familia supera a la heredada, en validación?

- Ganadora frente a heredada, las dos de la fase 2 (misma receta, mismas 5 semillas): diferencia media pareada por
  semilla con IC95 t; bootstrap por grupo de validación (2.000 remuestreos) del Dice medio en las 5 semillas;
  TOST con ±δ.
- Ganadora frente a las 5 corridas originales del benchmark: solo descriptivo, porque si el ajuste previo cambia
  la tasa la receta ya no es la misma.

Lectura fijada antes: si la heredada cae en el conjunto equivalente a la mejor, la herencia queda justificada por
la búsqueda. Si la ganadora la supera por δ o más, se declara que el benchmark subestimó esa familia y se reporta
la diferencia, sin cambiar las conclusiones del test ya cerrado.

## 9. Análisis del espacio (14.6)

Sobre la fase 1 de cada familia:

- Superficie: mapa de calor del Dice sobre (w1, w2) por w3, uno por patrón de dropout.
- Modelo lineal `y = β0 + Σ_i Σ_w β_iw·1[w_i = w] + Σ_i γ_i·b_i + Σ_{i<j} λ_ij·b_i·b_j + ε`, con IC95; F parcial
  por factor (sumas de cuadrados tipo II) con Holm sobre los 9 factores y proporción de varianza de cada uno.
- Tendencia: regresión sobre log2 w (lineal y cuadrática por capa); un término cuadrático negativo indica
  rendimientos decrecientes.
- Dropout: Wilcoxon pareado entre (0, 0, 0) y la media de los 7 patrones con dropout, con las 125 combinaciones de
  anchos como bloques, y correlación biserial de rangos como tamaño de efecto.
- Frente de Pareto: Dice frente a parámetros y frente a segundos.
- Sobreajuste: pérdida de validación menos pérdida de entrenamiento en la última época, por ancho total y patrón
  de dropout; en la fase 2, la brecha de Dice train-validación por finalista.
- Sensibilidad: desviación estándar entre arquitecturas frente a la desviación por semilla (repetición y fase 2).
  Si son parecidas, la arquitectura es irrelevante para este problema y se concluye así.
- Borde: si la ganadora tiene algún ancho en 256 o en 16. Un óptimo en el borde se declara como limitación del
  espacio, sin ampliarlo.
- Correlación de Spearman entre el Dice de la fase 1 y la media de la fase 2 en los finalistas.

## 10. Costo

`C_total = m·(|A|·t̄₁ + (k + 2)·R·t̄₂) + ajuste previo + repetición`, con m = 3, |A| = 1000, R = 5.

**Medido en CPU local** (22/11, 6 hilos, datos sintéticos de 256 px, sin lectura de imágenes): el segundo por
época sigue a los MACs con R² 0,99 (≈ 87 GMAC/s), así que el costo de cada arquitectura se puede proyectar desde
sus MACs. Las 1000 suman 15,5, 48,5 y 40,2 TMAC por imagen en FCN, U-Net y SegNet (mediana 10, 31 y 14 GMAC;
la heredada U-Net, 1,8). El ancho de la capa 1, que trabaja a resolución completa, domina el costo.

**Estimado para L4** (un proceso, `estimacion_costo_local.json`). La calibración con los tiempos del benchmark
no sirve para convoluciones 3×3 anchas (R² 0,65 con las redes SMP), así que se dan rangos: rendimiento efectivo
de 8 a 25 TFLOPS y sobrecosto fijo de 5 a 12 s por época con la caché llena.

| opción de la fase 1 | ajuste + fase 1 + repetición, h de L4 |
|---|---|
| A (256 px) | 74 a 211 |
| B (128 px) | 38 a 100 |
| C (128 px, mitad de train) | 32 a 82 |

Fase 2 con k = 10 (12 arquitecturas × 5 semillas × 3 familias, 24 épocas medias): de 12 a 24 h si los
finalistas tienen el costo mediano del espacio y de 51 a 145 h si son los más anchos; con k = 20, más o menos el
doble. Varios procesos en la misma GPU reparten el sobrecosto fijo, que en B y C es la mitad del total o más. El
piloto en L4 reemplaza estos rangos por una medición antes de fijar la fidelidad.

## 11. Reproducibilidad (14.7)

| elemento | dónde |
|---|---|
| Espacio declarado | este documento y `config/busqueda_arquitectura_e2.json` |
| Listado de arquitecturas | `config/arquitecturas_compactas_e2.csv` (id, w, b, parámetros y MACs por familia) |
| Fidelidad elegida y su estimación | `fidelidad.json`, `estimacion_computo.json` |
| Resultados de la fase 1 | `fase1/<id>/result.json` y `fase1.csv` (Dice de las 1000, semilla, presupuesto) |
| Finalistas, k y ruido | `finalistas.json` |
| Resultados de la fase 2 | `fase2/<id>/result.json` y `best_val_predictions.json` por finalista y semilla |
| Correlación fase 1-fase 2, regla, ganadora | `RESUMEN.json`, `RESUMEN.md` |
| Lectura | `resultados/busqueda_arquitectura_e2/LECTURA.md` |

Cada corrida guarda su configuración y una huella (tarea, protocolo, identidad de datos y fuentes, entorno): un
resultado con otra huella no se reutiliza. El cuaderno se reanuda solo tras un corte (la fase 1 por corrida
terminada, la fase 2 por época).
