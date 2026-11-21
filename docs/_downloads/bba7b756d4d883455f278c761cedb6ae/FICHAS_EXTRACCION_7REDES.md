# Fichas de extraccion: siete articulos prioritarios

**Rectificación 20/11:** fichas preliminares, no extracción final íntegramente verificada. U-Net E1 es compacta propia 16/32/64, NO ResNet-18; compararla con UNet++ R18 no aísla skips. Cifras y cuartiles no verificados requieren cotejo primario. Implementaciones ejecutadas: `libro/modelos_e2.md`.

Fecha: 18 de noviembre de 2026. Extraccion basada en secciones de metodo, configuracion experimental y repositorios oficiales consultados documentalmente. No se verificaron cifras de rendimiento contra textos completos mas alla de las secciones indicadas. Campos no disponibles marcados como tal; no se inventan cuartiles, conteos de citas ni puntuaciones completas.

---

## 1. DeepLabv3+ (Chen et al., ECCV 2018)

| Campo | Valor y evidencia |
|---|---|
| DOI, titulo, autores, anio | 10.48550/arXiv.1802.02611; "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"; L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, H. Adam; 2018 |
| Versiones relacionadas | ECCV 2018 (Lecture Notes in Computer Science, vol 11211). Preprint arXiv:1802.02611 |
| Base, consulta, archivo de origen | Excepcion fundacional (pre-2019); no recuperado directamente por Q0-Q4. Referencia por busqueda suplementaria |
| Tipo de publicacion y revision por pares | Conferencia revisada por pares (ECCV) |
| Decision de cribado | Retener como arquitectura candidata CNN con contexto multiescala |
| Tarea y modalidad | Segmentacion semantica multiclase; entrada RGB, salida mapa de etiquetas por pixel |
| Arquitectura, familia, mecanismo | CNN encoder-decoder; ASPP (Atrous Spatial Pyramid Pooling) con convoluciones dilatadas; decoder con refinamiento de bordes. DeepLabv3+ agrega decoder; DeepLabv3 no lo tiene |
| Datos principales | PASCAL VOC 2012, Cityscapes, JFT-300M (preentrenamiento) |
| Tamano y anotaciones | VOC 2012 augmented: 10582 train / 1449 val / 1456 test; mascaras semanticas por pixel, 21 clases |
| Licencia de datos | VOC: uso academico; JFT-300M: interno de Google, no disponible |
| Separacion train/val/test | Particiones estandar de cada benchmark |
| Preprocesamiento | Recorte aleatorio, escala [0.5, 2.0], flip horizontal; normalizacion ImageNet |
| Aumentos | Recorte aleatorio 513x513; multiscala en inferencia |
| Preentrenamiento | Backbone Xception-65 / ResNet-101 preentrenado en ImageNet o ImageNet+JFT-300M; codigo SMP permite ResNet-18 |
| Perdida | Entropia cruzada estandar por pixel |
| Optimizador, epocas, seleccion | SGD, tasa polinomial inicial 0.007, momentum 0.9, weight decay 4e-4; 30K iteraciones VOC, 90K Cityscapes |
| Checkpoint | Seleccion por mIoU de validacion |
| Metricas | mIoU (media de IoU por clase) |
| Resultados comparables | VOC 2012 test: 89.0% mIoU (Xception-65 + JFT); 87.8% sin JFT. Cityscapes test: 82.1% |
| Semillas, incertidumbre | No reportadas; sin repeticiones |
| Ablaciones | Tabla de ablacion sobre output stride, decoder, backbone, dilatacion (Tabla 1-4 del articulo) |
| Hardware | No especificado explicitamente; implica GPUs de Google |
| Parametros | Xception-65: ~41M; ResNet-101: ~59M. ResNet-18 (SMP): ~12M |
| Codigo oficial | github.com/tensorflow/models/tree/master/research/deeplab (TensorFlow) |
| Implementacion propuesta | SMP `DeepLabV3Plus`, ResNet-18, MIT license. No atribuir a autores originales |
| Pesos | SMP: `encoder_weights=None` para rama principal; ImageNet para secundaria |
| Dependencias | PyTorch, torchvision, SMP |
| Licencia de codigo | Implementacion original: Apache-2.0; SMP: MIT |
| Riesgos | BatchNorm con lote pequeno (1) puede desestabilizar; probar lote 2 o GroupNorm. ASPP con dilataciones altas puede no aportar a 256px |
| Limitaciones reconocidas | Autores no evaluan objetos muy pequenos ni segmentacion binaria |
| Adaptacion RGB binaria | Un canal de logits, `activation=None`, stride 16; sustituir loss multiclase por BCE+Dice binaria |
| Revista, cuartil | ECCV: conferencia de rango A* (CORE); no aplica cuartil de revista |
| Citas | No verificadas en esta sesion |
| Puntuacion rubrica | Pendiente de completar con componente de citacion |
| Decision de candidatura | Retener: CNN de referencia con ASPP y decoder; contrasta con FCN/SegNet/U-Net del E1 |

---

## 2. UNet++ (Zhou et al., TMI 2020 / DLMIA 2018)

| Campo | Valor y evidencia |
|---|---|
| DOI, titulo, autores, anio | 10.1109/TMI.2019.2959609; "UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"; Z. Zhou, M. Siddiquee, N. Tajbakhsh, J. Liang; 2020 (extension de 2018) |
| Versiones relacionadas | DLMIA 2018 workshop paper; TMI 2020 version extendida con poda y supervision profunda |
| Base, consulta, archivo de origen | Excepcion fundacional; no recuperado directamente por Q0-Q4 |
| Tipo de publicacion | Revista revisada por pares (IEEE TMI) |
| Decision de cribado | Retener como arquitectura candidata CNN con conexiones densas |
| Tarea y modalidad | Segmentacion semantica de imagenes medicas; entrada RGB o escala de grises, salida mascara |
| Arquitectura, familia, mecanismo | CNN encoder-decoder con skip connections anidados y densos; supervision profunda opcional; poda de subredes |
| Datos principales | 6 datasets medicos (nodulos pulmonares, polipos, celulas, higado, etc.) |
| Separacion train/val/test | Varia por dataset; reportada en cada experimento |
| Preentrenamiento | No mencionado explicitamente para los benchmarks medicos originales |
| Perdida | BCE + Dice por nodo de supervision profunda |
| Optimizador | Adam, lr 3e-4; detalles por dataset |
| Metricas | IoU y Dice por dataset |
| Resultados comparables | Mejora sobre U-Net en 4-6 puntos IoU en la mayoria de datasets |
| Ablaciones | Supervision profunda y poda; tablas 2-4 del TMI |
| Codigo oficial | github.com/MrGiovanni/UNetPlusPlus; licencia ASU no comercial |
| Implementacion propuesta | SMP `UnetPlusPlus`, ResNet-18, MIT. Sin supervision profunda. `aux_params` en SMP es clasificacion, no supervision profunda de segmentacion |
| Pesos | `encoder_weights=None` para rama principal |
| Riesgos | Conexiones densas consumen mas memoria que U-Net. No prometer coste equivalente. Supervision profunda requiere implementacion manual si se quiere replicar |
| Adaptacion RGB binaria | Un canal de logits, sin activacion final; BCE+Dice binaria |
| Revista, cuartil | IEEE TMI: JCR Q1 (no verificado en esta sesion; dato conocido de referencia general) |
| Puntuacion rubrica | Pendiente |
| Decision de candidatura | Retener: contraste controlado con U-Net del E1 usando mismo backbone (ResNet-18 SMP) |

---

## 3. PIDNet-S (Xu et al., CVPR 2023)

| Campo | Valor y evidencia |
|---|---|
| DOI, titulo, autores, anio | arXiv:2206.02066v3; "PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers"; J. Xu, Z. Xiong, S. P. Bhatt; CVPR 2023 |
| Base, consulta | No recuperado directamente por Q0-Q4; busqueda suplementaria |
| Tipo de publicacion | Conferencia revisada por pares (CVPR 2023) |
| Decision de cribado | Retener como arquitectura CNN con tres ramas (detalle, contexto, borde) |
| Tarea y modalidad | Segmentacion semantica en tiempo real; entrada RGB, salida mapa de etiquetas |
| Arquitectura, familia, mecanismo | CNN de tres ramas: P (detalle), I (contexto), D (borde). Fusion de ramas con funciones auxiliares. Configuracion S: m=2, n=3, planes=32, ppm_planes=96, head_planes=128 |
| Datos | Cityscapes (5000 finamente anotadas), CamVid (701 imagenes) |
| Separacion | Cityscapes: 2975 train, 500 val, 1525 test. CamVid: 367/101/233 |
| Preentrenamiento | ImageNet para backbone; para test de Cityscapes combinan train+val |
| Perdida | Entropia cruzada + OhemCE para ramas P e I; perdida de borde basada en deteccion de borde de Laplaciano para rama D; pesos de las ramas documentados |
| Optimizador | SGD, lr 0.01 (Cityscapes) / 0.001 (CamVid), polinomico, momentum 0.9, weight decay 5e-4 |
| Metricas | mIoU y FPS |
| Resultados (Tabla 2, seccion 4.2) | PIDNet-S Cityscapes val: 78.6% mIoU, 93.2 FPS (RTX 3090). CamVid test: 80.1% mIoU |
| Hardware reportado | RTX 3090 para FPS; no reportan VRAM ni RAM |
| Parametros | PIDNet-S: ~7.6M |
| Codigo oficial | github.com/XuJiacong/PIDNet; licencia MIT |
| Pesos | Ofrecidos en README; enlaces pueden cambiar. `get_seg_model(..., imgnet_pretrained=False)` aun carga checkpoint de segmentacion; no desactiva toda carga |
| Dependencias | PyTorch, utils del proyecto |
| Riesgos | Perdida de borde requiere derivar objetivos solo de mascaras de entrenamiento. Con lote 1, BatchNorm es problematico. Constructor permite init aleatoria, pero helper no |
| Adaptacion RGB binaria | Entrada RGB, segmentacion binaria, objetivo de borde derivado de mascara binaria. La salida principal es indice 1 de la lista de salidas. Redimensionar logits |
| Puntuacion rubrica | Pendiente |
| Decision de candidatura | Retener: CNN en tiempo real con tres ramas y perdida de borde; contraste unico con las demas |

---

## 4. SegNeXt-T (Guo et al., NeurIPS 2022)

| Campo | Valor y evidencia |
|---|---|
| DOI, titulo, autores, anio | arXiv:2209.08575; "SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation"; M. Guo et al.; NeurIPS 2022 |
| Base, consulta, archivo de origen | Q3 Scopus, E2-4774 |
| Tipo de publicacion | Conferencia revisada por pares (NeurIPS) |
| Decision de cribado | Retener como CNN con atencion convolucional (no Transformer) |
| Tarea y modalidad | Segmentacion semantica; entrada RGB, salida mapa de etiquetas |
| Arquitectura, familia, mecanismo | MSCAN (Multi-Scale Convolutional Attention Network) como encoder; LightHamHead como decoder. Es CNN con atencion convolucional multiescala, NO Transformer |
| Datos | ADE20K (20K train / 2K val / 150 clases), Cityscapes, Pascal Context |
| Preentrenamiento | Encoder MSCAN-T preentrenado en ImageNet-1K; pesos `mscan_t.pth` |
| Perdida | Entropia cruzada |
| Optimizador | AdamW, lr 6e-5, planificacion polinomial, weight decay 0.01; 160K iteraciones |
| Metricas | mIoU |
| Resultados (Tabla 1) | SegNeXt-T ADE20K: 41.1% mIoU (single-scale); ~4.3M parametros encoder |
| Hardware | 8 V100 GPUs |
| Parametros | MSCAN-T: ~4.3M; total con decoder: ~4.5M |
| Codigo oficial | github.com/Visual-Attention-Network/SegNeXt; Apache-2.0 |
| Dependencias | MMSegmentation 0.24.1, MMCV; entorno aislado necesario |
| Configuracion | local_configs/segnext/tiny/segnext.tiny.512x512.ade.160k.py; 150 clases, SyncBN/GN |
| Riesgos | MMSegmentation antiguo; SyncBN requiere adaptacion para ejecucion no distribuida. No instalar encima del entorno E1 |
| Adaptacion RGB binaria | Desactivar carga de pesos para rama aleatoria; cambiar cabeza a un logit; perdida binaria; adaptar SyncBN a BN o GN |
| Puntuacion rubrica | Pendiente |
| Decision de candidatura | Retener: CNN con atencion convolucional; complementa familia CNN pura y familia Transformer |

---

## 5. SegFormer-B0 (Xie et al., NeurIPS 2021)

| Campo | Valor y evidencia |
|---|---|
| DOI, titulo, autores, anio | arXiv:2105.15203v3; "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"; E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, P. Luo; NeurIPS 2021 |
| Base, consulta, archivo de origen | Q3 Scopus, E2-5366 |
| Tipo de publicacion | Conferencia revisada por pares (NeurIPS) |
| Decision de cribado | Retener como Transformer jerarquico; propuesto como paper guia |
| Tarea y modalidad | Segmentacion semantica; entrada RGB, salida mapa de etiquetas |
| Arquitectura, familia, mecanismo | Encoder: MiT (Mix Transformer) jerarquico con atencion eficiente por reduccion espacial. Decoder: MLP ligero que fusiona escalas. No usa convoluciones como modulo principal |
| Datos | ADE20K (20K/2K/150 clases), Cityscapes, COCO-Stuff |
| Preentrenamiento | MiT-B0 preentrenado en ImageNet-1K (`mit_b0.pth`) |
| Perdida | Entropia cruzada estandar multiclase |
| Optimizador (seccion 4.1) | AdamW, lr 6e-5, planificacion polinomial, weight decay 0.01; 160K iteraciones; batch 16 (8 GPUs x 2) |
| Metricas | mIoU (single-scale y multi-scale) |
| Resultados (Tabla 1, Fig. 4) | B0 ADE20K: **37.4% mIoU** single-scale, 38.0% multi-scale. ~3.8M parametros |
| Hardware | 8 V100 GPUs |
| Parametros | MiT-B0: ~3.4M encoder; total ~3.8M |
| Memoria/latencia | No reportada explicitamente para B0; FLOPs: 8.4G a 512x512 |
| Codigo oficial | github.com/NVlabs/SegFormer; licencia NVIDIA no comercial (investigacion/evaluacion) |
| Dependencias | PyTorch 1.7.1, CUDA 10.1, timm 0.3.2, MMCV 1.2.7, MMSegmentation 0.13.0 |
| Configuracion | local_configs/segformer/B0/segformer.b0.512x512.ade.160k.py |
| Pesos | Enlaces en README; no descargados ni verificados |
| Riesgos | Compatibilidad con PyTorch CPU del E1 no probada. Cambiar `num_classes` sin cambiar loss de softmax multiclase a binaria no funciona. MMCV/MMSeg antiguos |
| Adaptacion RGB binaria | Un logit de salida, perdida binaria explicita (BCE+Dice), restaurar tamano espacial antes de medir. Quitar preentrenamiento en rama principal |
| Revista, cuartil | NeurIPS: conferencia A* (CORE) |
| Puntuacion rubrica | Pendiente |
| Reproduccion de evaluacion | Checkpoint B0 ADE20K, clases originales, inferencia single-scale: 37.4% mIoU como referencia. Umbral de auditoria: >0.5 puntos |
| Decision de candidatura | Retener como prioridad y paper guia: variante pequena, configuracion publica, familia Transformer |

---

## 6. TransUNet 2D R50-ViT-B/16 (Chen et al., preprint 2021)

| Campo | Valor y evidencia |
|---|---|
| DOI, titulo, autores, anio | arXiv:2102.04306; "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"; J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, L. Lu, A. L. Yuille, Y. Zhou; 2021 (preprint) |
| Versiones relacionadas | Extension en Medical Image Analysis 2024, DOI 10.1016/j.media.2024.103280; no mezclar con version inicial |
| Base, consulta | No recuperado por Q0-Q4; busqueda suplementaria |
| Tipo de publicacion | Preprint; NO revisado por pares en su version 2021 |
| Decision de cribado | Retener como hibrido CNN-Transformer |
| Tarea y modalidad | Segmentacion de imagenes medicas (Synapse multi-organ, ACDC cardiac); entrada 2D grayscale/RGB, salida mascara |
| Arquitectura, familia, mecanismo | Hibrido: ResNet-50 + ViT-B/16 (Transformer de 12 capas, dim 768, 12 cabezas); decoder U-Net con skips de CNN |
| Variante | R50-ViT-B/16; cuadricula de tokens 16x16 |
| Datos | Synapse multi-organ CT (30 casos), ACDC cardiac MRI |
| Preentrenamiento | ViT-B/16 con checkpoint de ImageNet-21K (no 1K); pesos descargables |
| Perdida | Entropia cruzada + Dice |
| Optimizador | SGD, lr 0.01, momentum 0.9, weight decay 1e-4; 150 epocas (Synapse) |
| Metricas | Dice y HD95 por organo |
| Resultados | Synapse: 77.48% Dice promedio, 31.69mm HD95 (R50-ViT-B/16) |
| Hardware | No especificado explicitamente |
| Parametros | ~105M (R50-ViT-B/16) |
| Codigo oficial | github.com/Beckschen/TransUNet; Apache-2.0 |
| Dependencias | torch 1.4.0, torchvision 0.5.0, scipy, ml-collections; requirements.txt fija versiones antiguas |
| Pesos | Enlace actualizado en README; no descargados |
| Riesgos | Alto consumo de memoria por la variante hibrida. Embeddings posicionales requieren interpolacion si se cambia resolucion/cuadricula. Dependencias medicas a separar del lector RGB |
| Adaptacion RGB binaria | Entrada RGB 256x256, un logit de salida; fijar cuadricula 16x16; interpolar embeddings con logica oficial. Rama aleatoria sin `load_from` |
| Revista, cuartil | Preprint; no aplica cuartil. Extension MIA 2024: JCR Q1 (dato conocido; no verificado aqui) |
| Puntuacion rubrica | Pendiente. Penalizar por ser preprint no revisado |
| Decision de candidatura | Retener con condicion de presupuesto medido; complementa familia hibrida CNN-Transformer |

---

## 7. VM-UNet (Ruan y Xiang, preprint 2024)

| Campo | Valor y evidencia |
|---|---|
| DOI, titulo, autores, anio | arXiv:2402.02491v2; "VM-UNet: Vision Mamba UNet for Medical Image Segmentation"; J. Ruan, S. Xiang; 2024 |
| Versiones relacionadas | Repositorio anuncia aceptacion en ACM TOMM; ficha editorial DOI 10.1145/3767748 no verificable por error 403 |
| Base, consulta | No recuperado por Q0-Q4; busqueda suplementaria |
| Tipo de publicacion | Preprint v2; estado de revision no verificado |
| Decision de cribado | Retener como representante de familia SSM |
| Tarea y modalidad | Segmentacion de imagenes medicas (ISIC 2017/2018, Synapse); entrada 2D, salida mascara |
| Arquitectura, familia, mecanismo | VSSM (Visual State Space Model) como encoder; profundidades [2,2,9,2]; decoder simetrico [2,9,2,2]. Bloques VSS con estados selectivos para contexto espacial amplio |
| Datos | ISIC 2017 (2000 train, 150 val, 600 test), ISIC 2018 (2594 imagenes), Synapse multi-organ |
| Preentrenamiento (seccion IV-B) | VMamba-S preentrenado en ImageNet-1K; carga remapea claves de encoder a decoder |
| Perdida | BCE + Dice; wrapper oficial ya aplica sigmoid, exponer logits para BCEWithLogits |
| Optimizador | AdamW, lr 0.001, batch 32, 300 epocas |
| Metricas | Dice, IoU, accuracy, precision, recall, specificity |
| Resultados (Tabla III) | ISIC 2017: 90.98% Dice, 83.78% IoU (con preentrenamiento). Sin preentrenamiento: menor; ablacion documentada |
| Hardware | No reportado explicitamente; implica GPU CUDA |
| Parametros | No reportados explicitamente; VMamba-S tiene ~50M; configuracion base puede ser menor |
| Codigo oficial | github.com/JCruan519/VM-UNet; Apache-2.0 |
| Dependencias | torch 1.13, CUDA 11.7, Triton, causal_conv1d, mamba_ssm (ruedas Linux). No hay ruta Windows/AMD probada |
| Riesgos | Requiere Linux/NVIDIA. Wrapper aplica sigmoid: doble sigmoid con BCEWithLogits si no se modifica. Carga de pesos remapea encoder a decoder; registrar que claves se cargan |
| Adaptacion RGB binaria | Entrada 3 canales, 1 clase; modificar forward para exponer logits en lugar de sigmoid. Rama aleatoria sin cargar VMamba-S |
| Revista, cuartil | Preprint; estado editorial no verificado. No asignar cuartil |
| Puntuacion rubrica | Pendiente. Penalizar por entorno no compatible con hardware actual y preprint no verificado |
| Decision de candidatura | Retener en espera de prueba Linux/CUDA; unico representante de familia SSM. No sustituir por CNN |

---

## Notas generales

1. Ninguna ficha esta completa segun la plantilla de extraccion del protocolo. Faltan: conteos de citas verificados por base y fecha, cuartiles verificados por fuente oficial y anio, puntuacion completa de rubrica, y validacion de texto completo para todos los campos.

2. Las tres fichas que ya tenian extraccion parcial (SegFormer, PIDNet, VM-UNet) se ampliaron con los datos de VIABILIDAD_MODELOS.md. Las otras cuatro se completaron con datos del protocolo y los repositorios oficiales documentados.

3. Los campos de puntuacion se dejan pendientes intencionalmente: la rubrica requiere el componente de citacion normalizada que no se puede calcular sin verificar citas en la misma base y fecha para todos los candidatos.

4. No se descargaron checkpoints, datasets ni se ejecuto codigo. Las fichas reflejan lo publicado en articulos y repositorios, no resultados propios.
