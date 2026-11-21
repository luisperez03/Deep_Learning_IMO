# Literatura, selección operativa y paper guía

## Alcance de la revisión

Se investigan antecedentes de localización de rótulos, arquitecturas de segmentación transferibles y mecanismos de otras modalidades. Las ecuaciones exactas, filtros y versiones están en el {download}`protocolo de búsqueda <../docs/revision_entregable2/PROTOCOLO_BUSQUEDA.md>`; fechas y conteos, en la {download}`bitácora <../docs/revision_entregable2/BITACORA_BUSQUEDAS.md>`. Se consultaron Scopus y Web of Science Core Collection, con ventana 2019–2026 y excepciones fundacionales explícitas para FCN, SegNet, U-Net y DeepLabv3+.

La selección de siete candidatas nuevas fue aceptada como **lista de trabajo**. No equivale a una revisión sistemática completa de todas las referencias recuperadas. Según confirmación del usuario, el profesor acepta diez modelos totales, incluyendo las tres redes del E1.

## Identificación y precribado verificables

| Etapa | Conteo | Alcance de la evidencia |
|---|---:|---|
| Scopus Q0–Q4 | 6.399 | Registros exportados |
| WoS Q0–Q4 | 3.247 | Registros exportados |
| Total bruto | 9.646 | Trece archivos |
| Duplicados técnicos | 3.321 | Conciliación con reglas DOI/título |
| Grupos provisionales | 6.325 | Tres pares con DOI contradictorios, adjudicados el 20/11 con Crossref y doi.org; alias en el parser, fusión aún no aplicada: se esperan 6.322 grupos y 3.324 duplicados |
| Sugerencia automática: retener | 2.186 | 20 dominio, 507 arquitectura, 1.659 mecanismo |
| Sugerencia automática: excluir | 2.052 | No son exclusiones humanas definitivas |
| Duda | 2.087 | Requiere inspección |
| Elegibles e incluidos definitivos | No cerrado | No se deducen del número de redes entrenadas |

```text
9.646 registros importados
  └─ 3.321 duplicados técnicos
      └─ 6.325 grupos provisionales
          ├─ 2.186 sugerencias de retención
          ├─ 2.052 sugerencias de exclusión
          └─ 2.087 dudas
Adjudicación humana, recuperación y elegibilidad final: pendientes

Selección operativa paralela: 3 redes E1 + 7 candidatas nuevas
(no es una etapa final demostrada del flujo PRISMA)
```

Los conteos son aritméticamente coherentes, pero no acreditan la elegibilidad. Los tres pares de DOI son el mismo artículo en cada caso: dos variantes no están registradas y la tercera es un alias que el propio registro redirige al canónico ({download}`resolución con evidencia <../docs/revision_entregable2/RESOLUCION_CONFLICTOS_DOI.md>`). Los alias ya están en `procesar_exportaciones.py`, pero las exportaciones no están en el repositorio: 6.322 es el conteo esperado, no uno ejecutado, y no se usa como cifra cerrada hasta correr el parser. No se informa kappa porque no hay dos cribados independientes completos.

**Q5 y Q6 se omitieron por decisión del usuario.** Las fuentes suplementarias no sustituyen tres ecuaciones transversales ejecutadas. Este requisito continúa como alcance reducido; no se reabren las consultas automáticamente.

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

Las [fichas de implementación](modelos_e2.md) incluyen diez diagramas, mecanismos matemáticos, referencias y parámetros medidos. Los resultados originales de las siete redes nuevas (dataset, métrica, cifra, varianza y costo, con tabla y página del PDF) están cotejados en {download}`resultados originales <../docs/revision_entregable2/RESULTADOS_ORIGINALES_7REDES.md>`: 21 de 21 cifras confirmadas. Solo UNet++ reporta varianza entre corridas (20 corridas, media ± DE). Las {download}`fichas bibliográficas preliminares <../docs/revision_entregable2/FICHAS_EXTRACCION_7REDES.md>` conservan campos del artículo y advertencias de verificación. Los FLOPs de un paper a otra resolución no son FLOPs de nuestra adaptación.

Attention U-Net aumentaría representación CNN; SAM exige definir prompts sin ground truth; YOLO-seg requiere armonizar instancia y máscara semántica. Son razones operativas, no pruebas de inferioridad. La puntuación completa por citación normalizada/cuártil sigue pendiente. Los subtotales históricos no son un ranking validado. PIDNet no obtiene evidencia directa de dominio HAZMAT solo por incorporar bordes; U-Net compacta no comparte backbone ResNet-18 con U-Net++.

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

En los seis estudios aplicados identificados no se encontró ninguno que segmente el rótulo por píxel con una red profunda ni que compare arquitecturas con varias semillas; el único que produce máscara usa GrabCut sobre la caja del detector. La afirmación es refutable: basta un estudio revisado por pares que haga cualquiera de las dos cosas. Su denominador son esos seis estudios, no el corpus: los 20 registros de dominio y las 2.087 dudas del precribado no se han revisado a texto completo, así que la conclusión queda condicionada a esa revisión y no se escribe "ningún estudio previo". Tampoco hay denominador final para estimar proporción de estudios con datos o código.

La síntesis de los tres mecanismos transversales (atención, estado selectivo y dilatación), con fuente primaria, hipótesis para HAZMAT y el resultado del benchmark que refutaría cada una, está en {download}`mecanismos <../docs/revision_entregable2/MECANISMOS_TRANSVERSALES.md>`, sección 4.

No se asignan cuartiles sin año, categoría y fuente. El {download}`BibTeX <referencias_e2.bib>` reúne 21 referencias (diez arquitecturas en su versión editorial cuando existe, siete de aplicación y cuatro de mecanismos), con DOI verificados el 20/11/2026 en Crossref o DataCite.
