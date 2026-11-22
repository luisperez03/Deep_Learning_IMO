# Protocolo: taxonomía de errores, subgrupos y atajos

> **Trazabilidad.** Escrito el 22/11/2026, antes de calcular cualquier frecuencia de error, tabla de subgrupos o
> prueba de atajos. Los umbrales están en `config/errores_atajos_e2_20261122.json`, que leen los scripts; su SHA-256
> queda en cada salida. Ya se conocían: el test (observado en la evaluación cerrada), las tablas agregadas de
> diagnósticos (`resultados/diagnosticos_gpu_e2/TABLAS.md`) y el Dice por cuartil de área en test. Por eso todo lo que
> sigue es **exploratorio**: no cambia modelos, umbral (0,5) ni checkpoints, y no es un preregistro retroactivo.

## Preguntas

1. ¿De qué tipo son los errores de cada red y en qué condiciones de la imagen aparecen? (taxonomía, guía 12.3)
2. ¿Hay redes que rinden distinto según la clase, el tamaño, el borde, la luz, el contraste o el origen de la foto?
   (evaluación por segmentos y daño potencial, guía 8 y 12.3)
3. ¿Las redes segmentan el rótulo por el rótulo mismo o se apoyan en el contexto o en el color? (atajos, guía 2.1 y
   12.3)

## Datos y unidad

- Partición principal: **validación** (218 imágenes, 122 grupos de similitud), las 10 redes y sus 5 semillas. El test
  se reporta aparte como descriptivo.
- Unidad de inferencia: el **grupo de similitud**. Los IC son bootstrap por grupos (4.000 remuestreos, percentil 95 %,
  semilla 20261122).
- Fuente local: TP, FP, FN, HD95 y Dice por imagen de las 50 corridas (salida `clean` del complemento GPU), máscaras
  binarias de los manifiestos, imágenes y XML de VOC2012. Las predicciones (máscaras) no están en local: lo que las
  necesita va al cuaderno `colab/Atajos_Errores_E2.ipynb`.

## 1. Taxonomía de errores (dos ejes, cada uno mutuamente excluyente)

**Caso de error:** par imagen-semilla de validación con Dice < 0,80. Sensibilidad con 0,70 y 0,90. Si una red no
llega a 100 casos, se declara; no se mezcla el test para completar.

**Eje A, forma del error** (del conteo de píxeles; primera regla que se cumple):

| Categoría | Regla |
|---|---|
| A1 omisión total | TP = 0 y FP = 0 |
| A2 detección desplazada | TP = 0 y FP > 0 |
| A3 subsegmentación | FN ≥ 2·FP |
| A4 exceso adyacente | FP ≥ 2·FN y HD95 ≤ 16 px |
| A5 falso positivo distante | FP ≥ 2·FN y HD95 > 16 px |
| A6 error mixto | resto |

HD95 es un sustituto local de "exceso lejos del rótulo". En Colab se mide exacto (componente predicha sin solapamiento
con la máscara real dilatada 5 px) y se informa el acuerdo entre las dos versiones.

**Eje B, condición de la imagen** (solo imagen y máscara real, nunca la predicción; primera regla que se cumple):

| Categoría | Regla |
|---|---|
| B1 ruido confirmado | error de anotación confirmado a mano en `resultados/ruido_etiqueta_e2/confirmar.csv` |
| B2 rótulo en el borde | alguna región anotada toca el borde de la foto |
| B3 varios rótulos | dos o más regiones anotadas (≥ 0,05 % de la foto) |
| B4 rótulo pequeño | área < cuartil 1 de train |
| B5 luz extrema | luminancia media del rótulo < P10 o > P90 de train |
| B6 bajo contraste | diferencia de luminancia rótulo-anillo < P10 de train |
| B7 sin condición | resto |

Se etiqueta el **censo** de casos de error de cada red (no una muestra) con estas reglas, y se reporta la frecuencia
por categoría con IC. Como el etiquetado es automático, una persona revisa una **muestra estratificada de 30 casos por
red del top 3** (proporcional a la categoría A, mínimo 3 por categoría con casos) con un panel imagen / máscara real /
predicción, y marca si está de acuerdo con A y con B. Se reporta el acuerdo con IC de Wilson. Las reglas no se
cambian después de ver la hoja.

**Errores compartidos:** imagen "difícil para todas" si su Dice medio de 5 semillas es < 0,80 en al menos 8 de las
10 redes; se cruza con el eje B.

## 2. Subgrupos y daño potencial

- Atributos: clase (según SegmentationClass; "mixta" si hay más de una), cuartil de área (cortes de train), borde,
  número de rótulos, tercil de luminancia y de contraste (cortes de train) y origen (carpeta del XML, codificada
  O1–O7 por frecuencia en train; los nombres no se publican).
- Estimando: Dice medio por grupo dentro del subgrupo (Dice de la imagen promediado en 5 semillas). Contraste:
  subgrupo menos el resto, IC bootstrap por grupos y p bilateral con Holm dentro de cada red y atributo. Subgrupos con
  menos de 5 grupos se informan sin contraste.
- Daño potencial: **omisión grave** = recall < 0,5 (se pierde la mayor parte del rótulo), tasa por clase, área y luz.
  Es el error con más costo en la aplicación: un rótulo de mercancía peligrosa que no se ve. **Falsa alarma** en
  negativos: no hay imágenes sin rótulo en el conjunto; se miden en negativos sintéticos (C2).

## 3. Atajos (Colab, validación, 10 redes × 5 semillas)

| Condición | Qué cambia | Qué mide |
|---|---|---|
| C0 limpia | nada | Dice de referencia y componentes predichas (taxonomía exacta) |
| C1 sin contexto | fuera de las cajas de los rótulos (+15 %, mínimo 8 px) se pone la media de train | si el Dice cae, la red usa el contexto |
| C2 sin rótulo | la región del rótulo (+3 px) se rellena con el píxel exterior más cercano y se suaviza | si la red sigue marcando rótulo donde ya no está, lo infiere del contexto; además, falsas alarmas en un negativo |
| C3 gris | se quita el color | si el Dice cae, la red depende del color |

Regla de decisión, fijada aquí: con Δ = Dice(condición) − Dice(C0) por grupo, **dependencia material** si la cota
superior del IC95 < −δ (δ = 0,02, la del estudio) y p Holm < 0,05 sobre las 20 pruebas (10 redes × C1, C3); **sin
dependencia material** si la cota inferior > −δ; si no, indeterminado. En C2, **atajo de contexto** si la cota inferior
del IC95 de la fracción de la zona borrada predicha como rótulo supera 0,10.

**Límite que se declara de antemano:** C1 y C2 generan imágenes fuera de la distribución de entrenamiento. Una caída
puede deberse al artefacto y no a un atajo; un resultado "sin dependencia" es más fuerte que uno "con dependencia".
La dependencia del color no es necesariamente un atajo: el color es parte del significado del rótulo; lo que se mide
es cuánto pesa.

## Qué no se hace

- No se reentrena, no se recalibra y no se elige nada con estos resultados.
- No se usa el test para completar casos ni para decidir; el test aparece solo como descripción secundaria.
- No se interpreta un subgrupo pequeño sin su IC.
