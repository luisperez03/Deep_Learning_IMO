# Bitácora de diseño de DBNet-AT (enunciado de E3, §7.2 y plantilla A.3)

> Cada variante del mecanismo que se evaluó, con qué datos, qué se vio y qué se decidió. Sirve para declarar Nd y
> estimar el sesgo de selección.
>
> - Ninguna variante se evaluó con un modelo entrenado en validación ni en test. Todas las decisiones salieron del
>   oráculo de representación (la etiqueta como predicción perfecta) o de formas sintéticas.
> - Las particiones están en `diseno.md`, §5.6.
> - g0 = 4 px se fijó por la arquitectura (el paso del FPN) antes de cualquier medición y no se varió. Su
>   sensibilidad se mide después del congelamiento, como estudio exploratorio (`diseno.md`, §5.7).

| N.° | Variante evaluada (cambio respecto de la anterior) | Datos | Métrica | Decisión | Motivo y fecha |
|---|---|---|---|---|---|
| 0 | DBNet de E2 (`etiquetas.db_objetivos` y `db_post`) | Validación, 40 imágenes (oráculo de E2) | Dice del oráculo 0,957 | Referencia | Línea de partida, medida en E2 (`oraculo.json`) |
| 1 | C1 (g0 = 4) y C2 con d consistente por bisección y expansión euclídea | **Validación, 6 imágenes** | Dice del oráculo: DBNet de 0,95 a 0,98; la variante de 0,91 a 0,97 | Descartar la expansión euclídea | Encoger y expandir por distancia es una apertura morfológica que redondea las esquinas, y la r' = 1,5 de DBNet las compensa por casualidad. 2026-11-17 |
| 2 | Expansión en inglete (pyclipper), con un corrimiento de −0,5 px deducido de la geometría de los centros de píxel | Train, 12 imágenes; un caso con falsos negativos | Media de 0,96; en un caso, el inglete se quedaba corto (75 FN en el borde de una región de 372 px) | Convertir el corrimiento en un parámetro y calibrarlo en train | El −0,5 deducido no coincidía con el relleno de cv2. 2026-11-17 |
| 3 | El corrimiento calibrado en una rejilla de −0,5 a 1,0 px | Train, 200 primeras (`calibrar_ajuste.py`) | Dice del oráculo 0,969 con −0,25; de 0,86 a 0,97 en la rejilla | Fijar −0,25 provisionalmente | Lo eligió la regla fijada antes (el mayor Dice). La sensibilidad tan alta al ajuste indicaba otro problema. 2026-11-17 |
| 4 | Diagnóstico en formas sintéticas: un cuadrado de 32 px con C1 se reconstruía con un anillo de más | Sintético (`geometria_sintetica.py`), sin datos | La bisección cruzaba en el escalón de al lado de la región correcta | Agregar el refinamiento en píxeles (`_refinar`): elegir la región que, encogida con la regla, reproduce el núcleo | Error de la inversa en un espacio discreto, no de la regla. 2026-11-17 |
| 5 | Refinamiento en píxeles y recalibración del corrimiento | Train, 200 primeras | Dice del oráculo de 0,969 a 0,971 en toda la rejilla; elegido 0,75 (0,9707) | **Versión candidata a congelar** | El refinamiento quitó la sensibilidad al ajuste. 2026-11-17 |

**Nd = 5** variantes evaluadas, sin contar la referencia.

**Vistos después de la variante 5, sin ninguna decisión posterior:**
- El oráculo completo en las 218 imágenes de validación y en todo train (`resultados/oraculo_at.json`).
- La geometría sintética (`resultados/geometria_sintetica.json`).
- La prueba de sobreajuste: 8 imágenes de train, que se usan para entrenar y para evaluar
  (`resultados/sobreajuste_at.json` y `sobreajuste_reevaluado.json`).

**Agregado el 2026-11-18 al cotejar con el enunciado, sin evaluar nada nuevo:** dos variantes de inferencia para
completar el factorial 2³ de la ablación (`inglete_rfijo` y `c1_inglete_rfijo`). No son variantes de diseño: son
brazos de la ablación del modelo ya fijado.

**Congelamiento:** pendiente. Será la etiqueta `e3-protocolo`, antes de la búsqueda de DBNet-AT en GPU. Cualquier
variante posterior se agrega aquí con su fecha y reinicia el protocolo.

**Agregado el 2026-11-24, sin evaluar nada:** variante de inferencia `okr_sobre_at` (núcleos C1 con la dilatación de
Optimal Kernel, ICASSP 2023), para comparar con el trabajo más cercano hallado en la revisión de novedad
(`e3/experiments/novedad/lectura_okr.md`). No es una variante de diseño: no cambia C1, C2 ni el entrenamiento. Nd sigue
en 5. El diseño se porta a la U-Net compacta del E2 (`e3/src/`), con g0 = 4 px como paso del mapa más profundo; el
congelamiento es la etiqueta `e3-protocolo` del repositorio de estudio.
