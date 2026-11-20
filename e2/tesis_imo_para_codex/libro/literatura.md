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
| Grupos provisionales | 6.325 | Tres pares con DOI contradictorios pendientes de adjudicación aplicada |
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

Los conteos son aritméticamente coherentes, pero no acreditan la elegibilidad. El documento antiguo de resolución DOI no equivale a cambios aplicados a los registros; no se usa 6.322 sin una correspondencia validada. No se informa kappa porque no hay dos cribados independientes completos.

**Q5 y Q6 se omitieron por decisión del usuario.** Las fuentes suplementarias no sustituyen tres ecuaciones transversales ejecutadas. Este requisito continúa como alcance reducido; no se reabren las consultas automáticamente.

## Antecedentes y pertinencia

Se distingue detección de cajas, clasificación y segmentación de píxeles. DeepHAZMAT combina detección y segmentación algorítmica; su desempeño no equivale al Dice binario local. La identificación de la copia local como HAZMAT13 es probable, sin certificación byte a byte frente a una descarga externa. Fuentes y reservas en el capítulo de datos y la {download}`matriz documental de procedencia <../docs/PROCEDENCIA_Y_LITERATURA.md>`.

FCN fundamenta predicción densa convolucional; SegNet reutiliza índices de pooling; U-Net transporta características de mayor resolución al decoder. Se usan adaptaciones compactas, no reproducciones exactas. [FCN, CVPR](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html), [SegNet](https://arxiv.org/abs/1511.00561), [U-Net](https://arxiv.org/abs/1505.04597).

## Diversidad y comparación crítica

| Familia | Implementaciones | Aporte y límite |
|---|---|---|
| CNN | FCN, SegNet, U-Net compactas; DeepLabv3+ R18; U-Net++ R18; PIDNet-S; SegNeXt-T | Localidad, multiescala, skips y bordes; capacidades y auxiliares diferentes |
| Transformer | SegFormer-B0 | Atención jerárquica y decoder MLP; receta local distinta del artículo |
| Híbrido | TransUNet R50-ViT-B/16 | Contexto global con características CNN; mucho mayor tamaño |
| SSM visual | VM-UNet | Estados selectivos y escaneo; dependencia de kernels CUDA |

Las [fichas de implementación](modelos_e2.md) incluyen diez diagramas, mecanismos matemáticos, referencias y parámetros medidos. Las {download}`fichas bibliográficas preliminares <../docs/revision_entregable2/FICHAS_EXTRACCION_7REDES.md>` conservan campos del artículo y advertencias de verificación. Los FLOPs de un paper a otra resolución no son FLOPs de nuestra adaptación.

Attention U-Net aumentaría representación CNN; SAM exige definir prompts sin ground truth; YOLO-seg requiere armonizar instancia y máscara semántica. Son razones operativas, no pruebas de inferioridad. La puntuación completa por citación normalizada/cuártil sigue pendiente. Los subtotales históricos no son un ranking validado. PIDNet no obtiene evidencia directa de dominio HAZMAT solo por incorporar bordes; U-Net compacta no comparte backbone ResNet-18 con U-Net++.

## Mecanismos transferibles

| Mecanismo | Fuente primaria | Propiedad transferible | Riesgo |
|---|---|---|---|
| Autoatención | [Vaswani et al.](https://arxiv.org/abs/1706.03762), secuencias | Relaciones distantes | Costo y pérdida de detalle por reducción |
| Estado selectivo | [Mamba](https://arxiv.org/abs/2312.00752), secuencias | Contexto con estado de dimensión fija | Orden de escaneo y portabilidad |
| Dilatación multiescala | [Yu y Koltun](https://arxiv.org/abs/1511.07122), [WaveNet](https://arxiv.org/abs/1609.03499) | Campo receptivo amplio sin pooling adicional | No afirmar genealogía unidireccional audio→visión; escala inadecuada para objetos pequeños |
| Detalle/contexto/borde | [PIDNet](https://arxiv.org/abs/2206.02066), analogía de control | Fusión guiada por fronteras | Es una analogía estructural, no un controlador físico |

Estas propiedades motivan hipótesis; no prueban beneficio en HAZMAT. Las cuatro familias son categorías de diseño, no cuatro réplicas independientes de un efecto de familia.

## Paper guía y réplica pendiente

Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021. [Artículo oficial](https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf), [DOI del preprint](https://doi.org/10.48550/arXiv.2105.15203), [código de autores](https://github.com/NVlabs/SegFormer).

Combina encoder Transformer jerárquico y decoder MLP ligero. B0 se seleccionó por disponibilidad de configuración y escala manejable. El plan documental propone evaluar ADE20K a escala única, con referencia de 37,4 % mIoU. Antes de ejecutar deben fijarse checkpoint, partición, resolución, protocolo original, tolerancia de éxito y cálculo de incertidumbre. No existe aún resultado ejecutado ni discrepancia de réplica.

La adaptación HAZMAT cambia dataset, objetivo, inicialización, resolución y pérdida: **su Dice no valida una réplica del paper**. El último lugar de SegFormer tampoco acredita una limitación intrínseca del mecanismo. Falta un piloto específico en al menos tres modelos, con condición medible, magnitud e IC.

## Gap y límites de la síntesis

La comparación local motiva investigar preservación de detalle frente a complejidad bajo recetas controladas. Es un gap candidato, no la afirmación de que ningún estudio previo lo examinó. El cribado incompleto impide esa conclusión universal. Tampoco hay denominador final para estimar proporción de estudios con datos/código.

No se asignan cuartiles sin año, categoría y fuente. Las referencias de arquitecturas están en {download}`BibTeX <referencias_e2.bib>`; falta completar el mismo control para todas las referencias de aplicación y mecanismos.
