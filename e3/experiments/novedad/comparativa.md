# Tabla comparativa de métodos afines (R1)

Fecha de la revisión: 2026-11-23. Fuente de cada fila: el texto completo cuando se pudo leer (indicado), y si no, el resumen.
Modelo propuesto: encogimiento del núcleo atenuado por el radio inscrito (C1) y expansión consistente en inferencia como
punto fijo con expansión en inglete (C2), sobre la representación de DBNet. En el diseño el modelo se llama DBNet-AT
(ResNet-50 con FPN); el encargo lo llama UNet-AT sobre una U-Net compacta. Los dos cambios son los mismos; el antecedente de
DB sobre una variante de U-Net es DB M-Net (2020, doi 10.1145/3421515.3421531, TC de esófago).

Tipos de diferencia: **mecanismo** (cambia la regla de la etiqueta o de la reconstrucción), **pérdida o entrenamiento**
(cambia lo que se optimiza o cómo), **solo aplicación** (el mismo método en otro dominio).

## Tabla

| # | Método (referencia) | Qué hace con el encogimiento y la expansión | Qué conserva el modelo propuesto | Qué es nuevo en el modelo propuesto frente a él | Tipo |
|---|---|---|---|---|---|
| 1 | **DBNet**, Liao et al., AAAI 2020, doi 10.1609/aaai.v34i07.6812 (texto completo leído) | Etiqueta: D = A(1 − r²)/L con r = 0,4 fijo. Inferencia: expande con D' = A'·r'/L', r' = 1,5 fijo, uniones redondas (Vatti). Binarización diferenciable con k = 50 y mapa de umbral | Red, binarización diferenciable, umbral aprendido, pérdida 5·Ls + 10·Lt + Lb, k = 50 y todo el protocolo | C1: D atenuado por φ(g) = g²/(g² + g0²) con g el radio inscrito. C2: d como punto fijo de la regla, con inglete, en lugar de r' fijo | mecanismo (C1 y C2) |
| 2 | **DBNet++**, Liao et al., TPAMI 2023, doi 10.1109/TPAMI.2022.3155612 (resumen y referencias) | Misma etiqueta y misma expansión que DBNet; añade Adaptive Scale Fusion para la fusión de escalas | No usa ASF; hereda la etiqueta y la expansión de DBNet | La regla de etiqueta y de expansión; ASF queda fuera del modelo propuesto | mecanismo (ASF es arquitectura; no se adopta) |
| 3 | **PSENet**, Wang et al., CVPR 2019, doi 10.1109/CVPR.2019.00956 (descripción de la fuente original, sin releer en esta revisión) | Varios núcleos con razones de encogimiento fijas y expansión progresiva por búsqueda en anchura desde el núcleo más pequeño | La idea de núcleo encogido que separa instancias | Un solo núcleo con regla adaptativa y expansión calculada, sin núcleos múltiples ni búsqueda en anchura | mecanismo |
| 4 | **TextSnake**, Long et al., ECCV 2018, doi 10.1007/978-3-030-01216-8_2 (descripción de la fuente original) | Línea central y radio de disco por píxel regresados; reconstruye con discos. El radio es el del círculo inscrito | Nada de la representación (cambia de familia, ver diseño §3.4) | C1 usa el radio inscrito como una medida de la región para atenuar el encogimiento, no como cantidad a regresar | mecanismo |
| 5 | **ADNet** (Adaptive Dilation Network), Qu et al., IEEE TMM 2023, doi 10.1109/TMM.2022.3216729 (resumen y código oficial, github.com/qqqyd/ADNet, leído) | Encogimiento fijo r = 0,4. Una cabeza regresa un factor de dilatación por píxel; su etiqueta es k = D·L'/A' (D, perímetro y área del núcleo), el factor que invierte el encogimiento con la fórmula de unclip. En inferencia, k = promedio sobre la región y expansión con uniones redondas | Conserva la idea de que la expansión debe depender de la instancia | Sin cabeza de regresión: d sale del punto fijo de la regla, sin parámetros. Inglete en vez de uniones redondas. Regla de encogimiento distinta (C1). Teorema de inversión exacta en polígonos convexos | mecanismo (C2) y entrenamiento (sin cabeza ni pérdida adicionales) |
| 6 | **Jiang et al.**, Adaptive Text Region Representation, IEEE Access 2020, doi 10.1109/ACCESS.2020.2999069 (arXiv 2104.00297, texto completo leído) | Núcleos con varias razones de encogimiento en entrenamiento; una cabeza regresa la distancia de expansión d por instancia; la reconstrucción mueve cada vértice sobre la bisectriz d/sen(θ/2), es decir, un desplazamiento en inglete | La idea de expandir con distancia por instancia y en inglete | La distancia d no se regresa: se resuelve como punto fijo de la regla de encogimiento. Regla de encogimiento atenuada (C1) | mecanismo (C2) |
| 7 | **ASMTD** (arXiv 2111.09560), **Reinforcement Shrink-Mask** (IEEE TMM 2023), **EK-Net** (ICASSP 2024, arXiv 2401.11704), **Deformable Kernel Expansion** (arXiv 2303.15737) (los tres primeros y DKE con texto completo, salvo Reinforcement Shrink-Mask, solo resumen) | Encogimiento de etiqueta fijo (r = 0,4); regresan offsets o un mapa de distancia de expansión para reconstruir el contorno desde el núcleo | La etiqueta de encogimiento de DBNet como punto de partida | Sin regresión de offsets; regla de encogimiento adaptativa y expansión determinista | mecanismo (C1 y C2) y pérdida o entrenamiento (ellos añaden cabezas y pérdidas; el modelo propuesto no) |
| 8 | **Optimal Kernel Representation (OKR)**, Ma et al., ICASSP 2023, doi 10.1109/ICASSP49357.2023.10094734 (solo resumen; texto en IEEE Xplore) | El resumen dice que "erosiona los textos según sus características geométricas". La regla no se pudo verificar | Por verificar | Por verificar. Es el antecedente más cercano a C1 y el único que no se pudo descartar | por verificar (riesgo para C1) |
| 9 | **DeepHAZMAT**, Sharifi et al., Machine Learning with Applications 2021, doi 10.1016/j.mlwa.2021.100104 (arXiv 2007.06392, leído) | YOLOv3-tiny para detectar el rótulo, GrabCut adaptativo, morfología y envolvente convexa para la máscara; conjunto de rótulos hazmat | El problema (rótulos hazmat) y los datos tipo HAZMAT-13 | Segmentación por píxel con una red entrenada de extremo a extremo y representación de núcleo con expansión | solo aplicación (el problema) y mecanismo (el método) |
| 10 | **HMD-Net**, Vehicle Hazmat Marker Detection Benchmark, Entropy 2022, doi 10.3390/e24040466 (resumen) | Detección de marcadores hazmat en vehículos con una red compacta y poda de canales; 10.000 imágenes | El problema (rótulos hazmat) | Máscara por píxel en lugar de cajas y representación de núcleo con expansión | solo aplicación y mecanismo |

## Lectura de la tabla

- **Qué es nuevo, en una línea.** C1 (atenuar el encogimiento según el radio inscrito frente al paso del mapa, g0 = 4 px) y la
  forma de C2 (resolver d como punto fijo de la regla, sin cabeza y con inglete, con prueba de inversión exacta) no aparecen en
  los trabajos revisados. Lo que sí está en la literatura es la **meta** de C2 (que la expansión dependa de la instancia y
  deshaga el encogimiento): filas 5 a 7.
- **Qué se conserva.** La representación de DBNet completa (binarización diferenciable, umbral aprendido, pérdida), el
  protocolo de E2 y la arquitectura. Los parámetros y los FLOPs no cambian.
- **Clasificación.** El aporte es de mecanismo en la etiqueta (C1) y en la reconstrucción (C2). No hay pérdida nueva, y el
  diseño lo declara así (§3.1). La aplicación a rótulos hazmat es solo aplicación (filas 9 y 10).
- **Lo que debe citarse en el documento del diseño.** El §3.4 descarta "r' aprendido por región" como si fuera una alternativa
  sin precedente; los precedentes son ADNet (factor por píxel regresado, etiqueta k = D·L'/A'), Jiang et al. (distancia
  regresada con expansión en inglete) y EK-Net, DKE y ASMTD (offsets regresados). La nota de R2 sobre `pyclipper` y el inglete
  debe citar también a Jiang et al., Eq. 4 y Algoritmo 1.
- **No incluidos.** PAN y FCENet figuran en las referencias de DBNet++ y ASMTD, pero sus textos no se leyeron en esta
  revisión y no se afirma nada sobre su regla. ContourNet, TextRay y ABCNet son de otra familia (contorno).
- **Límite.** Las filas 3, 4 y 8 descansan en descripciones o resúmenes. La fila 8 es la que cambiaría la evaluación.
