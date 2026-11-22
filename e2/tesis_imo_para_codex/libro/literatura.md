# Literatura, selección operativa y paper guía

## Alcance de la revisión

Se investigan antecedentes de localización de rótulos, arquitecturas de segmentación transferibles y mecanismos de otras modalidades. Las ecuaciones exactas, filtros y versiones están en el {download}`protocolo de búsqueda <../docs/revision_entregable2/PROTOCOLO_BUSQUEDA.md>`; fechas y conteos, en la {download}`bitácora <../docs/revision_entregable2/BITACORA_BUSQUEDAS.md>`. Se consultaron Scopus y Web of Science Core Collection, con ventana 2019–2026 y excepciones fundacionales explícitas para FCN, SegNet, U-Net y DeepLabv3+.

La selección de siete candidatas nuevas fue aceptada como **lista de trabajo**. No equivale a una revisión sistemática completa de todas las referencias recuperadas. Según confirmación del usuario, el profesor acepta diez modelos totales, incluyendo las tres redes del E1.

## Identificación, cribado y elegibilidad

| Etapa | Conteo | Evidencia |
|---|---:|---|
| Scopus Q0–Q4 | 6.399 | Registros exportados |
| WoS Q0–Q4 | 3.247 | Registros exportados |
| Total importado | 9.646 | Trece archivos; sha256 fijados en `conciliacion.json` |
| Duplicados | 3.324 | DOI normalizado y título/año; incluye los tres pares de DOI fusionados con sus alias |
| Grupos únicos | **6.322** | `procesar_exportaciones.py`, sin coincidencias ambiguas |
| Precribado automático: retener | 3.687 | 23 dominio, 343 arquitectura, 3.321 mecanismo |
| Precribado automático: excluir | 2.513 | 2.143 dominio no transferible, 185 tarea, 124 tipo documental, 61 detección |
| Precribado automático: duda | 122 | |
| Núcleo de dominio revisado | 25 | Incluidos 10; excluidos 15 con motivo |
| Muestra de validación del precribado | 200 | Aleatoria estratificada, título y resumen |

```text
9.646 registros importados (Scopus 6.399, WoS 3.247)
  └─ 3.324 duplicados
      └─ 6.322 grupos únicos
          ├─ Núcleo de dominio (precribado de dominio, mención de rótulos o consultas Q0/Q1): 25
          │     ├─ excluidos: 15 (13 tarea sin transferencia, 2 sin método original)
          │     └─ INCLUIDOS: 10 estudios sobre rótulos o placas de mercancía peligrosa
          └─ Resto: 6.297 con decisión automática
                ├─ muestra revisada: 200 (60 excluir, 60 duda, 40 mecanismo, 40 arquitectura)
                └─ incluibles estimados a título y resumen: 3.103 (2.256–3.940), no evaluados
                   a texto completo: fuera del alcance de la síntesis

Síntesis: 10 estudios de dominio + 9 arquitecturas puntuadas (fichas cotejadas)
          + fuentes primarias de los mecanismos
```

Los tres pares de DOI son el mismo artículo en cada caso: dos variantes no están registradas y la tercera es un alias que el propio registro redirige al canónico ({download}`resolución con evidencia <../docs/revision_entregable2/RESOLUCION_CONFLICTOS_DOI.md>`). Con los alias en el parser, los tres pares se fusionan sin perder identificadores: cada grupo absorbido queda registrado en el que lo recibe.

**El precribado se reconstruyó.** Del filtro automático original solo quedaron los conteos, sin sus reglas ni sus decisiones por registro. `precribado_reconstruido.py` lo sustituye con las mismas ocho categorías y reglas explícitas. Coincide en lo que importa para el gap (23 registros de dominio frente a 20) y difiere en el resto: manda a "mecanismo" casi todas las antiguas dudas, porque términos como atención o Transformer aparecen en casi todo el corpus.

**Su error se midió, y es grande.** Una muestra aleatoria estratificada de 200 registros (semilla fija) se revisó a título y resumen con criterios escritos antes de empezar ({download}`criterios <../docs/revision_entregable2/CRITERIOS_ELEGIBILIDAD.md>`), sin ver la decisión del filtro. En el estrato de exclusión, 22 de 60 registros eran incluibles (37 %, IC95 de Wilson 26–49 %): el filtro descarta como "dominio no transferible" muchos estudios médicos 2D con mecanismos generales que los criterios sí admiten. En los estratos de retención sobra entre el 20 % (arquitectura) y el 45 % (mecanismo). Por eso ninguna decisión automática se presenta como exclusión humana. El corpus se usa en dos niveles: el núcleo de dominio, revisado registro por registro, y una estimación de cuántos incluibles hay en el resto (unos 3.100), que no se leyeron a texto completo y quedan fuera de la síntesis. Flujo completo con exclusiones por motivo en {download}`PRISMA <../docs/revision_entregable2/PRISMA_FINAL.md>`.

Las decisiones las propone un asistente de IA con esos criterios. Una segunda pasada sobre 20 de las 200 coincidió en las 20. No es un segundo revisor independiente, así que no se informa kappa.

**Búsquedas transversales: el requisito no se cumple.** La guía pide tres búsquedas transversales (mecanismos de otras modalidades) ejecutadas en las bases; se ejecutó una parte. Q5 llegó a mostrar conteos piloto (38 en Scopus y 14 en WoS), pero la exportación falló en las dos bases el 18/11 por la caída de la sesión institucional, y Q6 no se ejecutó; después se decidió no reabrirlas. Un fallo de exportación no es un resultado cero, así que esos 52 registros no se suman al corpus ni se cuentan como cribados.

Lo que hay en su lugar es más débil y se declara como tal: los cuatro mecanismos de la tabla de abajo se documentaron desde su fuente primaria por búsqueda dirigida por nombre, y Q3 y Q4 (mecanismos dentro de visión) sí se exportaron. Eso prueba que cada mecanismo existe y de dónde viene; no prueba que no haya otros mecanismos transferibles que una búsqueda sistemática habría encontrado. El riesgo concreto es de omisión: un trabajo que lleve, por ejemplo, un mecanismo de audio o de series temporales a la segmentación de objetos pequeños podría faltar. Por eso las hipótesis de transferencia se presentan como hipótesis y ninguna conclusión del benchmark depende de que la lista de mecanismos sea exhaustiva.

## Antecedentes y pertinencia

Seis estudios aplicados a rótulos de mercancía peligrosa, con DOI verificado, ubican el problema: Sharifi et al. (2021, *Machine Learning with Applications*, DOI 10.1016/j.mlwa.2021.100104), Zhang et al. (2024, *IEEE Transactions on Artificial Intelligence*, DOI 10.1109/TAI.2023.3298588), Mohamed et al. (2018, IEEE SSRR, DOI 10.1109/SSRR.2018.8468639), Edlinger et al. (2019, *Electronic Imaging*, DOI 10.2352/ISSN.2470-1173.2019.7.IRIACV-463), Zhao et al. (2013, IEEE GlobalSIP, DOI 10.1109/GlobalSIP.2013.6736996) y Thongkor et al. (2016, *Electronic Imaging*, DOI 10.2352/ISSN.2470-1173.2016.11.IMAWM-459). Todos detectan, reconocen o localizan el rótulo; qué hace cada uno está en {download}`aplicación y gap <../docs/revision_entregable2/APLICACION_Y_GAP.md>`, leído de los resúmenes.

Se distingue detección de cajas, clasificación y segmentación de píxeles. DeepHAZMAT combina detección y segmentación algorítmica; su desempeño no equivale al Dice binario local. La identificación de la copia local como HAZMAT13 es probable, sin certificación byte a byte frente a una descarga externa. Fuentes y reservas en el capítulo de datos y la {download}`matriz documental de procedencia <../docs/PROCEDENCIA_Y_LITERATURA.md>`.

FCN fundamenta predicción densa convolucional; SegNet reutiliza índices de pooling; U-Net transporta características de mayor resolución al decoder. Se usan adaptaciones compactas, no reproducciones exactas. [FCN, CVPR](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html), [SegNet](https://arxiv.org/abs/1511.00561), [U-Net](https://arxiv.org/abs/1505.04597).

## Diversidad y comparación crítica

| Familia | Implementaciones | Aporte y límite |
|---|---|---|
| CNN | FCN, SegNet, U-Net compactas; DeepLabv3+ R18; U-Net++ R18; PIDNet-S; SegNeXt-T | Localidad, multiescala, skips y bordes; capacidades y auxiliares diferentes |
| Transformer | SegFormer-B0 | Atención jerárquica y decoder MLP; receta local distinta del artículo |
| Híbrido | TransUNet R50-ViT-B/16 | Contexto global con características CNN; mucho mayor tamaño |
| SSM visual | VM-UNet | Estados selectivos y escaneo; dependencia de kernels CUDA |

Las [fichas de implementación](modelos_e2.md) incluyen diez diagramas, mecanismos matemáticos, referencias y parámetros medidos. Los resultados originales de las siete redes nuevas (dataset, métrica, cifra, varianza y costo, con tabla y página del PDF) están cotejados en {download}`resultados originales <../docs/revision_entregable2/RESULTADOS_ORIGINALES_7REDES.md>`: 21 de 21 cifras confirmadas. Solo UNet++ reporta varianza entre corridas (20 corridas, media ± DE). Los FLOPs de un paper a otra resolución no son FLOPs de nuestra adaptación.

### Extracción y calidad de cada estudio

Cada una de las siete candidatas, y las dos alternativas que se puntúan con ellas (Attention U-Net y Swin-Unet), tiene una ficha de trece campos leída del PDF: tarea, mecanismo, datos y unidad de partición, preprocesamiento, entrenamiento, métricas, repeticiones, ablaciones, dificultad compartida con HAZMAT-13, costo, código, riesgo de selección por test y limitaciones. Cada valor lleva la página y un fragmento literal, y `cotejar_fichas.py` comprueba que el fragmento está en esa página: **229 de 229 confirmados**; siete campos son ausencias declaradas ("no reportado"), todas de repeticiones o semillas ({download}`fichas cotejadas <../docs/revision_entregable2/FICHAS_COTEJADAS.md>`).

Tres hallazgos de calidad pesan en la lectura de la literatura:

- **Repeticiones.** Solo UNet++ repite corridas con contraste formal (20 ensayos, prueba t, IC95). Attention U-Net usa validación cruzada de cinco particiones; las otras siete dan una cifra de una sola corrida.
- **Selección sobre el conjunto reportado.** En TransUNet, VM-UNet y Swin-Unet no hay conjunto de selección aparte: configuraciones como el dropout, la resolución o las conexiones se eligen comparando sobre el mismo conjunto cuyos resultados se publican. Sus cifras de cabecera son optimistas por construcción.
- **Dificultad compartida.** Cuatro artículos traen un análisis cuantitativo dedicado a algo que también complica HAZMAT-13: bordes (experimento trimap de DeepLabv3+), tamaño del objeto (UNet++ por tamaño de tumor; TransUNet en órganos pequeños) y corrupción de la imagen (SegFormer en Cityscapes-C). En PIDNet-S los bordes son la motivación del diseño, pero su ablación de la pérdida de frontera se mide en mIoU global, sin métrica de borde.

### Puntuación definitiva

La rúbrica del protocolo (pertinencia 25, reproducibilidad 25, evidencia 20, viabilidad 10, recencia 10, citación 10) se aplicó a las nueve candidatas con reglas operativas escritas antes de calcular y evidencia de la ficha para cada nivel. La citación es el percentil de citas por año en OpenAlex, misma fecha para todas; la reproducibilidad se comprobó en el repositorio oficial (licencia y pesos publicados).

| Candidata | Total | Sin citación | Sin viabilidad | Estado |
|---|---:|---:|---:|---|
| SegFormer-B0 | 77,08 | 76,39 | 74,54 | seleccionada |
| UNet++ | 73,89 | 77,78 | 70,99 | seleccionada |
| DeepLabv3+ | 73,19 | 70,83 | 70,22 | seleccionada |
| TransUNet | 64,72 | 63,89 | 60,80 | seleccionada |
| SegNeXt-T | 64,17 | 69,44 | 60,19 | seleccionada |
| PIDNet-S | 59,31 | 65,28 | 54,78 | seleccionada |
| Attention U-Net | 58,75 | 59,72 | 56,94 | excluida |
| VM-UNet | 56,53 | 59,72 | 51,70 | seleccionada |
| Swin-Unet | 54,86 | 54,17 | 52,62 | excluida |

Con la restricción de cuatro familias, el mejor conjunto de siete es exactamente la selección operativa, con la puntuación total y también sin citación. VM-UNet queda por debajo de Attention U-Net, pero entra porque es la única SSM. La viabilidad favorece a las redes ya ejecutadas (las siete tienen piloto; las dos alternativas no). Si se quita ese componente, Attention U-Net (56,94) desplaza a PIDNet-S (54,78): la elección de PIDNet-S frente a Attention U-Net depende de la viabilidad demostrada y no de la literatura. Desglose, justificación por componente y rango de la sede en {download}`puntuación definitiva <../docs/revision_entregable2/PUNTUACION_DEFINITIVA.md>`.

SAM y YOLO-seg no se puntúan porque no pasan el filtro previo del protocolo: SAM necesita prompts y YOLO-seg da máscaras de instancia por caja. Son razones operativas, no pruebas de inferioridad. U-Net compacta no comparte backbone ResNet-18 con U-Net++.

## Mecanismos transferibles

| Mecanismo | Fuente primaria | Propiedad transferible | Riesgo |
|---|---|---|---|
| Autoatención | [Vaswani et al.](https://arxiv.org/abs/1706.03762), secuencias | Relaciones distantes | Costo y pérdida de detalle por reducción |
| Estado selectivo | [Mamba](https://arxiv.org/abs/2312.00752), secuencias | Contexto con estado de dimensión fija | Orden de escaneo y portabilidad |
| Dilatación multiescala | [Yu y Koltun](https://arxiv.org/abs/1511.07122), [WaveNet](https://arxiv.org/abs/1609.03499) | Campo receptivo amplio sin pooling adicional | No afirmar genealogía unidireccional audio→visión; escala inadecuada para objetos pequeños |
| Detalle/contexto/borde | [PIDNet](https://arxiv.org/abs/2206.02066), analogía de control | Fusión guiada por fronteras | Es una analogía estructural, no un controlador físico |

Estas propiedades motivan hipótesis; no prueban beneficio en HAZMAT. Las cuatro familias son categorías de diseño, no cuatro réplicas independientes de un efecto de familia.

## Paper guía y réplica de evaluación

Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021. [Artículo oficial](https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf), [DOI del preprint](https://doi.org/10.48550/arXiv.2105.15203), [código de autores](https://github.com/NVlabs/SegFormer).

Combina encoder Transformer jerárquico y decoder MLP ligero. B0 se seleccionó por disponibilidad de configuración y escala manejable. El artículo reporta 37,4 % mIoU en la validación de ADE20K a escala única.

Antes de ejecutar se fijaron en `config/replica_segformer.json` el checkpoint oficial (`nvidia/segformer-b0-finetuned-ade-512-512`, revisión y SHA-256 fijados), los datos (las 2.000 imágenes de validación, con el SHA-256 del archivo), el preprocesamiento de la configuración original (redimensionado alineado a 2048 × 512 e inferencia sobre la imagen completa), la métrica (mIoU acumulado sobre 150 clases), la incertidumbre (bootstrap por imagen, 2.000 réplicas) y el criterio: *replicado* si el IC95 de la diferencia D con 37,4 cae dentro de ±1 punto; *compatible* si |D| ≤ 1 pero el IC sale de ese margen.

| Variante | mIoU (%) | IC95 bootstrap por imagen (%) |
|---|---:|---:|
| Preprocesamiento de la configuración original (primaria) | 37,43 | 35,47–38,30 |
| Preprocesador por defecto de Hugging Face, 512 × 512 sin proporción (D1) | 35,94 | 34,02–36,85 |
| Media de IoU por imagen, que no es la métrica del artículo (D2) | 33,90 | — |

D = +0,03 puntos, con IC95 de −1,93 a +0,90. **Veredicto según el criterio fijado: compatible.** La cifra coincide y 37,4 cae dentro del IC95, pero la incertidumbre por imagen es más ancha que el margen de ±1 punto, de modo que no alcanza la categoría de replicado. El preprocesamiento explica diferencias del orden de 1,5 puntos: quien evalúe con el preprocesador por defecto obtiene un número menor sin que el modelo cambie. La conversión de los pesos frente al archivo original de los autores (D3) no se evaluó.

Es una réplica de **evaluación**: confirma que los pesos publicados, con el preprocesamiento y la métrica del artículo, dan la cifra reportada. No replica el entrenamiento. Corrida en una GPU T4 con `colab/Replica_SegFormer_ADE20K.ipynb`; detalle en `resultados/replica_segformer/LECTURA.md` y salida completa en `resumen_replica.json`.

La adaptación HAZMAT cambia dataset, objetivo, inicialización, resolución y pérdida: **su Dice no valida una réplica del paper**. La réplica valida el checkpoint de Hugging Face y la métrica, no la implementación de SMP (`mit_b0`) entrenada en HAZMAT. El último lugar de SegFormer tampoco acredita una limitación intrínseca del mecanismo.

### Limitación por área del rótulo

El protocolo `config/limitacion_segformer_area.json`, fijado antes de mirar resultados por área, plantea que SegFormer-B0 pierde más Dice que otras redes en los rótulos pequeños. Unidad: los 122 grupos de validación, en terciles de la fracción de área (cortes 0,0195 y 0,0411); Δ es el Dice del tercil grande menos el del pequeño, promediado en las cinco semillas, con IC95 bootstrap por grupos. La limitación *existe* si el límite inferior del IC95 de Δ de SegFormer supera 0,02; es *específica* si además Δ de SegFormer supera al de U-Net, DeepLabv3+ y SegNeXt-T con significación tras Holm.

| Red | Dice tercil pequeño | Dice tercil grande | Δ [IC95] |
|---|---:|---:|---:|
| SegFormer-B0 | 0,661 | 0,743 | 0,082 [−0,030; 0,194] |
| U-Net | 0,782 | 0,864 | 0,081 [0,004; 0,150] |
| DeepLabv3+ | 0,772 | 0,845 | 0,073 [−0,007; 0,148] |
| SegNeXt-T | 0,699 | 0,819 | 0,120 [0,030; 0,207] |
| TransUNet | 0,814 | 0,871 | 0,057 [−0,014; 0,119] |

**Veredicto a priori: no confirmada.** El IC95 de Δ de SegFormer incluye valores menores que 0,02, y su diferencia de Δ con cada red piloto es casi nula (U-Net 0,001 [−0,076; 0,082]; DeepLabv3+ 0,010; SegNeXt-T −0,038; p de Holm ≥ 0,55). Lo que sí aparece es un déficit de nivel: en el tercil pequeño SegFormer queda 0,12 por debajo de U-Net [−0,184; −0,064] y 0,11 por debajo de DeepLabv3+. Todas las redes rinden menos en rótulos pequeños, en parte porque el Dice castiga más un mismo error de borde en un objeto pequeño; SegFormer no pierde más que las demás, parte más abajo. Datos: Dice por imagen de validación extraído de los 50 mejores checkpoints con `colab/Extraer_Validacion_E2.ipynb` (`resultados/validacion_por_imagen_e2/`), análisis en `resultados/limitacion_segformer_area/RESUMEN.md`.

## Gap y límites de la síntesis

**Afirmación.** Ningún estudio de los identificados sobre rótulos o placas de mercancía peligrosa segmenta el rótulo por píxel con una red profunda ni compara arquitecturas de segmentación con varias semillas. El denominador son 13 estudios: los 10 de dominio que el cribado incluyó del corpus y tres anteriores a 2019, fuera de la ventana de búsqueda, hallados por búsqueda dirigida (Mohamed et al., 2018; Zhao et al., 2013; Thongkor et al., 2016). Los 13 detectan, reconocen, clasifican o localizan. El único que produce máscara, DeepHAZMAT, la obtiene con GrabCut sobre la caja del detector. E2-0007 publica un conjunto RGB-D de señales hazmat, pero su segmentación es de planos sobre la nube de puntos y su detector, yolov3-tiny, da cajas.

**Qué la refuta.** Un solo estudio revisado por pares que segmente por píxel rótulos o placas de mercancía peligrosa con una red profunda, o que compare arquitecturas de segmentación sobre ese objeto informando la variabilidad entre corridas.

**Alcance de la lectura.** Dos de los diez se leyeron a texto completo: E2-0007 y DeepHAZMAT, en sus versiones de arXiv. Los otros ocho se decidieron con el resumen, que en todos declara el método: un detector de cajas (YOLO u otro), un clasificador o visión clásica (umbral en HSV, OpenCV). Su texto completo no se pudo leer: IEEE Xplore no cargó a través del ezproxy, Uninorte no tiene suscripción a Springer ni a SPIE, e IS&T no ofrece el PDF. Por eso la afirmación se acota a "en los estudios identificados", y no a "ningún estudio previo". No se estima la proporción de estudios con datos o código: el denominador es de 13 y su propósito es el gap, no esa estadística.

La síntesis de los tres mecanismos transversales (atención, estado selectivo y dilatación), con fuente primaria, hipótesis para HAZMAT y el resultado del benchmark que refutaría cada una, está en {download}`mecanismos <../docs/revision_entregable2/MECANISMOS_TRANSVERSALES.md>`, sección 4.

No se asignan cuartiles sin año, categoría y fuente: las cinco conferencias llevan su rango CORE (ECCV, CVPR y NeurIPS son A* en ICORE2026) y las dos revistas son Q1 en SCImago en el año de la versión citada: IEEE TMI en 2020, en sus cuatro categorías, para UNet++; ACM TOMM en 2025, en Computer Networks and Communications y en Hardware and Architecture, para VM-UNet. Ambas siguen en Q1 en 2025. La consulta, a mano, fue el 22/11/2026, con datos de Scopus de marzo de 2026. El cuartil no entra en la rúbrica. El {download}`BibTeX <referencias_e2.bib>` reúne 21 referencias (diez arquitecturas en su versión editorial cuando existe, siete de aplicación y cuatro de mecanismos), con DOI verificados el 20/11/2026 en Crossref o DataCite.
