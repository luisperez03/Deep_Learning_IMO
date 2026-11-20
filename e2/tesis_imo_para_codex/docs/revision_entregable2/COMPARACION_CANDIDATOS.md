# Candidatos para contrastar: siete plazas adicionales

La auditoría técnica posterior y las variantes recomendadas están en [VIABILIDAD_MODELOS.md](VIABILIDAD_MODELOS.md). En particular, se propone UNet++ con ResNet-18 sin supervisión profunda para el benchmark; no atribuir a esa adaptación toda la receta del artículo original.

Actualizado el 18 de noviembre de 2026. **Propuesta de lectura y viabilidad, no Top 7 definitivo ni resultado de un cribado exhaustivo.** La búsqueda general tiene miles de referencias todavía sin revisar. No asignar puntuaciones completas, cuartiles ni citas faltantes como si estuvieran verificados.

**Decisión del usuario del 18/11:** continuar con estas siete candidatas como lista de trabajo y omitir Q5/Q6. Se prioriza su verificación documental y técnica. Esto no convierte la lista en ganadora de una revisión sistemática ni autoriza entrenamiento, subida a GitHub o despliegue.

Supuesto pendiente: FCN, SegNet y U-Net del E1 cuentan dentro de diez modelos totales. La siguiente combinación permitiría explorar cuatro familias: CNN, Transformer, híbrido CNN–Transformer y SSM. Los nombres y variantes son una propuesta; no se ha instalado ni entrenado ninguno.

| Candidato | Familia y aporte a contrastar | Adaptación / condición por resolver | Procedencia verificada |
|---|---|---|---|
| DeepLabv3+ | CNN; contexto multiescala y refinamiento de bordes | Fijar backbone y salida binaria; no confundirlo con DeepLabv3. Excepción fundacional de 2018. | [Artículo ECCV/arXiv](https://arxiv.org/abs/1802.02611). |
| UNet++ | CNN; conexiones densas entre encoder y decoder, supervisión profunda | Comparar con U-Net controlando ancho, backbone y supervisión. Distinguir versión 2018 de extensión TMI 2020. | [Artículo extendido](https://arxiv.org/abs/1912.05074), [código oficial](https://github.com/MrGiovanni/UNetPlusPlus). |
| PIDNet-S | CNN; ramas de detalle, contexto y borde | Derivar objetivos de borde sólo de máscaras de entrenamiento. Comprobar pesos y coste en el hardware disponible. | [Artículo CVPR 2023, versión arXiv](https://arxiv.org/html/2206.02066v3), [código oficial](https://github.com/XuJiacong/PIDNet). |
| SegNeXt-T | CNN con atención convolucional multiescala; contraste con atención Transformer | No contarlo como familia Transformer. Revisar compatibilidad de MMSegmentation y configuración de variante Tiny. | Q3 Scopus, E2-4774; [artículo](https://arxiv.org/abs/2209.08575), [código oficial](https://github.com/Visual-Attention-Network/SegNeXt). |
| SegFormer-B0 | Transformer jerárquico y decoder MLP | Resolver preentrenamiento y adaptación a máscara binaria; medir objetos pequeños y bordes en nuestro conjunto. | Q3 Scopus, E2-5366; [artículo](https://arxiv.org/html/2105.15203v3), [código oficial](https://github.com/NVlabs/SegFormer). |
| TransUNet 2D | Híbrido CNN–Transformer; detalle local y contexto global | Congelar variante 2D concreta. No mezclar automáticamente el preprint 2021 con la extensión 2D/3D de 2024. | [Preprint 2021](https://arxiv.org/abs/2102.04306), [código y referencia de extensión](https://github.com/Beckschen/TransUNet). |
| VM-UNet | SSM; bloques de estado visual para contexto espacial | Validar instalación y memoria antes de comprometerlo. Mantener alternativa SSM si no es reproducible en el entorno. | [Preprint v2](https://arxiv.org/html/2402.02491v2), [código oficial](https://github.com/JCruan519/VM-UNet). El repositorio anuncia ACM TOMM; la ficha editorial del DOI 10.1145/3767748 no pudo verificarse por acceso 403. |

Attention U-Net queda como alternativa de lectura, no exclusión científica: su mecanismo de compuertas interesa, pero sigue siendo CNN y por sí solo no amplía las cuatro familias. No se afirma que rinda menos. La búsqueda deberá contrastar también candidatos recientes y los que no figuran en esta lista.

## Qué ya se comprobó en las implementaciones

- SegFormer declara código, configuraciones y pesos, sobre MMSegmentation 0.13.0 y un entorno antiguo. SegNeXt utiliza MMSegmentation 0.24.1. Esto exige revisar compatibilidad; aún no se ejecutaron. Fuentes: repositorios enlazados arriba.
- VM-UNet documenta `causal_conv1d`, `mamba_ssm` y Triton, con ejemplos de ruedas Linux/CUDA. Esto identifica un riesgo de instalación en Windows; no demuestra imposibilidad. Su repositorio muestra Apache-2.0. [Entorno oficial](https://github.com/JCruan519/VM-UNet#0-main-environments).
- TransUNet señala que un enlace antiguo de pesos expiró y ofrece otra ubicación; PIDNet también avisa de enlaces de pesos reemplazados. Se comprobó el aviso, no la descarga de los nuevos pesos. Fuentes: README oficiales enlazados.
- Cada licencia de código, pesos y datos se verificará por separado antes de incorporar o redistribuir material. No se clonaron repositorios ni se descargaron pesos.

## Extracción parcial de tres artículos

Se leyeron secciones de método y configuración experimental, no se declara una auditoría completa de cada artículo.

**SegFormer, §4.1:** encoder preentrenado en ImageNet-1K; AdamW, tasa inicial 0,00006 y planificación polinómica. Los experimentos principales emplean hasta 160.000 iteraciones y ocho V100. **Implicación propia:** usar B0 es una propuesta de viabilidad; entrenarlo en nuestros datos no replica esos experimentos. [Fuente](https://arxiv.org/html/2105.15203v3#S4.SS1).

**PIDNet, §4.2:** el protocolo distingue preentrenamiento, ajuste y evaluación; para resultados de test de Cityscapes/CamVid combina train y val antes de probar. La velocidad se mide en RTX 3090, lote uno. **Implicación propia:** no trasladar esa fusión a nuestro benchmark mientras validación siga eligiendo modelos; sus FPS no son una promesa para nuestro equipo. [Fuente](https://arxiv.org/html/2206.02066v3#S4.SS2).

**VM-UNet, §IV-B y tabla III:** utiliza pesos VMamba-S preentrenados en ImageNet-1K, AdamW, tasa inicial 0,001, lote 32 y 300 épocas. Incluye ablación con y sin preentrenamiento. **Implicación propia:** una comparación contra las redes compactas del E1 debe declarar y controlar esta diferencia. [Fuente](https://arxiv.org/html/2402.02491v2#S4.SS2).

## Recomendación operativa

1. Trabajar sobre Q0–Q4 y priorizar las siete candidatas aceptadas. Q3 recuperó los originales de SegFormer y SegNeXt, pero su ausencia para otros nombres no demuestra que no estén indexados: el filtro de año y los términos del resumen pueden excluirlos. No reanudar Q5/Q6.
2. Registrar una búsqueda dirigida suplementaria por títulos/DOI para candidatos fundacionales y comprobar versiones. No sumarla a Q3 sin registrar su procedencia.
3. Completar fichas de calidad y presupuesto; puntuar con la rúbrica del protocolo y analizar sensibilidad al peso de recencia/citas.
4. Confirmar viabilidad de las siete plazas aceptadas y documentar cualquier sustitución necesaria. SegFormer-B0 y PIDNet-S son prioridades de verificación por su relación con contexto/bordes; esta es una propuesta técnica, no un ranking bibliométrico.
5. Definir por separado benchmark adaptado a HAZMAT y réplica del paper guía. No usar test para elegir arquitectura, pesos iniciales ni hiperparámetros.
