# Declaración de novedad (borrador) y evaluación del riesgo de R1

Estado: borrador provisional, 2026-11-23. Cubre fuentes abiertas (OpenAlex, arXiv, Semantic Scholar, OpenReview, GitHub,
Hugging Face Hub y Papers). Scopus, Web of Science e IEEE Xplore están pendientes (`pendiente_bases_suscripcion.md`) y la
verificación por un tercero también (`guia_verificador.md`). La declaración no es definitiva hasta cerrar esas dos partes.

## Evaluación del riesgo, sin rodeos

**La parte de aplicación está libre.** Ningún estudio encontrado aplica una representación de núcleo encogido con expansión
(DBNet y familia) a rótulos o placas de mercancía peligrosa. Los trabajos de dominio detectan con cajas (YOLO y otros),
clasifican, leen o usan visión clásica; el único que produce máscara, DeepHAZMAT, lo hace con GrabCut sobre la caja. El cruce
mecanismo-problema (cadena C5) devolvió 0 resultados en arXiv y ninguno de dominio en OpenAlex, Semantic Scholar y OpenReview.

**C1 (encogimiento atenuado por el radio inscrito): sin antecedente confirmado, con un cabo suelto.**
- No se encontró ningún método que atenúe la distancia de encogimiento según el radio inscrito o el grosor de la instancia,
  ni respecto del paso del mapa de predicción. Las variantes revisadas mantienen r = 0,4 fijo en la etiqueta (ADNet, ASMTD,
  EK-Net, DKE) o cambian de representación (CentripetalText, FAST, TextSnake).
- **Cabo suelto: Optimal Kernel Representation (OKR), ICASSP 2023.** Su resumen dice que "erosiona los textos según sus
  características geométricas". La regla no se pudo leer (IEEE Xplore). Si la erosión depende del grosor o del radio inscrito,
  C1 tendría antecedente en texto de escena y la novedad quedaría en el umbral g0 ligado al paso del mapa y en su aplicación.
  **Prioridad 1 de la parte pendiente.**
- También sin leer a fondo: Text kernel calculation (Visual Computer 2023) y Pull Pole Points to Text Contour by Magnetism
  (TIP 2025). Este último critica que el encogimiento sea "la misma distancia en todas partes", lo que muestra que la
  limitación de C1 es conocida; su remedio es otro (puntos polares), según el resumen.

**C2 (expansión consistente): la meta tiene antecedentes; la forma, no.** Es el punto con más riesgo, y debe redactarse con
cuidado.
- **ADNet** (IEEE TMM 2023) ya hace que la expansión dependa de la instancia y deshaga el encogimiento: su etiqueta del factor
  es k = D·L'/A', que invierte el encogimiento con la fórmula de unclip. Lo hace con una cabeza que regresa k por píxel y con
  uniones redondas (código oficial leído).
- **Jiang et al.** (IEEE Access 2020, Eq. 4 y Algoritmo 1) ya expanden en inglete (desplazamiento d/sen(θ/2) sobre la
  bisectriz) con una distancia por instancia que una cabeza predice.
- **EK-Net, DKE, ASMTD y Reinforcement Shrink-Mask** regresan offsets o mapas de distancia de expansión.
- **Lo que no apareció:** resolver d como punto fijo de la propia regla de encogimiento (sin cabeza ni parámetros), la prueba
  de inversión exacta en polígonos convexos con unicidad (Bonnesen y fórmulas de Steiner, §3.5 del diseño) y la combinación
  de ambas con una regla de etiqueta atenuada.
- **Consecuencia para la redacción.** No debe afirmarse que "la expansión adaptativa por instancia" o "la expansión en
  inglete" sean nuevas. Lo que se reclama es la **expansión determinista, sin cabeza y con inversión exacta demostrada**, y
  debe compararse en la ablación con la variante regresada (ADNet) cuando el diseño lo permita. El §3.4 del diseño trata
  "r' aprendido por región" como una alternativa descartada; debe citar a ADNet y a Jiang et al. como sus realizaciones
  publicadas.

**Veredicto del riesgo de R1.** Medio para C2 (antecedentes de la meta, no del método), por cerrar para C1 (OKR sin verificar)
y bajo para la aplicación. Ningún trabajo encontrado hace C1 ni C2 tal como están definidos; uno (ADNet) hace una versión
aprendida de la meta de C2, y esa diferencia debe quedar escrita.

## Declaración propuesta

Con la fórmula del enunciado, acotada a lo que la búsqueda permite decir:

> Al mejor de nuestro conocimiento y según la búsqueda descrita, no se encontraron estudios que apliquen el modelo propuesto
> (encogimiento del núcleo atenuado por el radio inscrito y expansión consistente en inferencia, resuelta como punto fijo con
> expansión en inglete, sobre la representación de binarización diferenciable) a la segmentación de rótulos de mercancías
> peligrosas en fotografías de contenedores.

Texto de acompañamiento:

- **Fuente del mecanismo.** La representación de núcleo encogido con binarización diferenciable proviene de la detección de
  texto en escena: Liao et al. (2020), *Real-time Scene Text Detection with Differentiable Binarization*, AAAI,
  doi 10.1609/aaai.v34i07.6812, con la extensión de Liao et al. (2023), doi 10.1109/TPAMI.2022.3155612. El mismo mecanismo
  se transfirió a otras tareas de segmentación sin cambiar el encogimiento: Gao et al. (2020), arXiv 2012.04164
  (localización de multitudes), y DB M-Net (2020, doi 10.1145/3421515.3421531, TC de esófago).
- **Trabajos más cercanos.** En expansión del núcleo: ADNet (doi 10.1109/TMM.2022.3216729), Jiang et al.
  (doi 10.1109/ACCESS.2020.2999069), EK-Net (arXiv 2401.11704), Deformable Kernel Expansion (arXiv 2303.15737) y ASMTD
  (arXiv 2111.09560). En regla de encogimiento por geometría: Optimal Kernel Representation
  (doi 10.1109/ICASSP49357.2023.10094734; por verificar). En el problema de aplicación: DeepHAZMAT
  (doi 10.1016/j.mlwa.2021.100104), HMD-Net (doi 10.3390/e24040466), Zhang et al. (2024, doi 10.1109/TAI.2023.3298588) y los
  trabajos de detección y clasificación del E2.
- **Diferencia.** Frente a esos trabajos, la expansión no se regresa con una cabeza: se calcula como punto fijo de la regla
  de encogimiento y deshace exactamente el encogimiento en polígonos convexos (§3.5 del diseño); y la regla de encogimiento se
  atenúa en las regiones cuyo radio inscrito es del orden del paso del mapa.
- **Alcance.** Fuentes abiertas consultadas el 2026-11-23; bases con suscripción y verificación por un tercero, pendientes al
  redactar este borrador. Los resúmenes sin texto completo se listan en `comparativa.md`.

## Qué cambiaría esta evaluación

- Que OKR (o Text kernel calculation) use el radio inscrito o el grosor en su regla: C1 pasa a tener antecedente y la
  declaración debe mencionarlo como tal.
- Que Scopus, Web of Science o IEEE Xplore devuelvan un trabajo que resuelva d como punto fijo o con una inversa calculada:
  C2 pierde su diferencia principal frente a ADNet.
- Que el verificador encuentre un estudio de segmentación de rótulos hazmat con representación de núcleo encogido.
