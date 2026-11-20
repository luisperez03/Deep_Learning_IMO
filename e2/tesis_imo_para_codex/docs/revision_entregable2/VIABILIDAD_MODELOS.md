# Viabilidad de las siete redes y diseño experimental propuesto

Revisión documental: 18 de noviembre de 2026. Lista de trabajo aceptada por el usuario; variantes y diseño recomendados, todavía sin ejecutar. No es un ranking bibliométrico. Se consultaron artículos y código de sus autores, además de documentación de implementadores cuando se propone una adaptación.

## Decisión recomendada

Conservar las siete arquitecturas. Empezar la implementación por SegFormer-B0 y PIDNet-S. Usar SegFormer como paper guía. Reservar VM-UNet hasta comprobar un entorno Linux/NVIDIA; no sustituirlo automáticamente por otra CNN porque se perdería la familia SSM. TransUNet requiere especial atención al presupuesto. Las otras cinco tienen una ruta razonable de adaptación, no una certificación de funcionamiento.

| Modelo y variante propuesta | Familia | Implementación para nuestro benchmark | Dictamen documental |
|---|---|---|---|
| DeepLabv3+ con ResNet-18 | CNN | `DeepLabV3Plus` de SMP, salida binaria | Viable con adaptación; probar BatchNorm y dilataciones |
| UNet++ con ResNet-18, sin supervisión profunda | CNN | `UnetPlusPlus` de SMP | Viable como variante adaptada; no réplica íntegra del original |
| PIDNet-S | CNN | Implementación oficial, ramas P/I/D conservadas | Viable con adaptador de salidas y pérdida auxiliar |
| SegNeXt-T | CNN con atención convolucional | MSCAN-T + LightHamHead oficiales | Condicionado a resolver entorno MMSegmentation antiguo |
| SegFormer-B0 | Transformer | MiT-B0 + SegFormerHead oficiales | Prioridad; resolver compatibilidad y salida binaria |
| TransUNet 2D R50-ViT-B/16 | CNN–Transformer | Variante híbrida oficial | Condicionado al presupuesto medido; priorizar GPU |
| VM-UNet, configuración base del repositorio | SSM | VSSM con profundidades encoder [2,2,9,2] | En espera de prueba Linux/CUDA; no listo para el portátil |

El equipo registrado el 06/11 tiene Ryzen 5 7520U, unos 7,28 GiB de RAM utilizable, Radeon integrada y Windows. No se ha vuelto a medir ni se ha constatado GPU NVIDIA. Evidencia: `resultados/preentrenamiento/hardware.json`. La corrida FCN guardó PyTorch CPU, lote 1 y 256 píxeles (`resultados/piloto_local_v1/fcn_seed11_lr0.001/config.json`). No extrapolar su velocidad a estas redes. No se han medido RAM, VRAM, parámetros adaptados ni segundos por época de las siete.

## Fichas y fuentes primarias

### 1. DeepLabv3+

- **Referencia:** [Chen et al., ECCV 2018](https://arxiv.org/abs/1802.02611). Encoder con convolución dilatada, ASPP y decoder; DeepLabv3 sin decoder no es sustituto equivalente.
- **Código:** el [repositorio original TensorFlow](https://github.com/tensorflow/models/tree/master/research/deeplab) sirve de referencia. Para HAZMAT se propone la implementación mantenida por [SMP](https://github.com/qubvel-org/segmentation_models.pytorch), no atribuirla a los autores del artículo.
- **Adaptación propuesta:** ResNet-18, RGB, un canal de logits, `activation=None`, stride 16. Fijar explícitamente parámetros de decoder y dilataciones antes de correr. Compartir backbone con UNet++ ayuda a interpretar esa pareja, pero no aísla todas las diferencias frente al E1.
- **Pesos:** rama principal `encoder_weights=None`; ImageNet sólo en experimento separado. Dependencias: PyTorch, torchvision y SMP; congelar versiones compatibles en un entorno nuevo.
- **Licencia:** implementación SMP [MIT](https://raw.githubusercontent.com/qubvel-org/segmentation_models.pytorch/main/LICENSE). Esto no acredita automáticamente licencias de checkpoints o datasets.
- **Riesgo:** comprobar entrenamiento con lote pequeño y pooling/BatchNorm del ASPP. [API de DeepLabV3Plus](https://segmentation-models-pytorch.readthedocs.io/en/latest/models.html#deeplabv3).

### 2. UNet++

- **Referencia:** distinguir propuesta de 2018 de [extensión TMI 2020](https://arxiv.org/abs/1912.05074), DOI 10.1109/TMI.2019.2959609.
- **Código/licencia original:** [MrGiovanni/UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus); [licencia ASU](https://github.com/MrGiovanni/UNetPlusPlus/blob/master/LICENSE) de uso no comercial. No describirla como MIT.
- **Ruta recomendada:** SMP, ResNet-18, RGB, un logit, sin activación final, sin pesos externos en la rama principal. SMP tiene licencia MIT propia. Registrar que se usa decoder anidado con encoder ResNet y sin supervisión profunda ni poda del artículo.
- **Advertencia de implementación:** `aux_params` en [SMP](https://segmentation-models-pytorch.readthedocs.io/en/latest/models.html#unet) agrega clasificación, no la supervisión profunda de segmentación de UNet++. Para estudiar esta última habría que implementar cabezas intermedias y declararlo como otro experimento.
- **Viabilidad:** stack compartido con DeepLabv3+; medir memoria de las conexiones densas. No prometer que cuesta lo mismo que U-Net.

### 3. PIDNet-S

- **Referencia:** [CVPR 2023, preprint v3](https://arxiv.org/html/2206.02066v3); [repositorio oficial](https://github.com/XuJiacong/PIDNet), [MIT](https://raw.githubusercontent.com/XuJiacong/PIDNet/main/LICENSE).
- **Variante:** configuración pequeña `m=2, n=3, planes=32, ppm_planes=96, head_planes=128`. [Código auditado](https://raw.githubusercontent.com/XuJiacong/PIDNet/refs/heads/main/models/pidnet.py).
- **Adaptación:** entrada RGB, segmentación binaria y objetivo de borde derivado exclusivamente de la máscara correspondiente. En entrenamiento la lista contiene salida auxiliar P, salida principal y borde D; la principal es índice 1. Redimensionar logits a la geometría de evaluación, excluyendo padding también en pérdidas auxiliares.
- **Pesos:** el constructor permite inicialización aleatoria. En cambio, `get_seg_model(..., imgnet_pretrained=False)` todavía carga un checkpoint de segmentación: ese booleano no desactiva toda carga. La rama principal deberá evitar esa función o modificar explícitamente la carga.
- **Dependencias/riesgos:** PyTorch y utilidades del proyecto; comprobar BatchNorm, pérdida de borde y configuración original completa. El README ofrece pesos y avisa de cambios de enlaces; descarga e integridad no comprobadas. No confundir ImageNet con preentrenamiento en Cityscapes.

### 4. SegNeXt-T

- **Referencia:** [NeurIPS 2022](https://arxiv.org/abs/2209.08575); original recuperado en Q3 Scopus, E2-4774.
- **Código/licencia:** [oficial](https://github.com/Visual-Attention-Network/SegNeXt), [Apache-2.0](https://raw.githubusercontent.com/Visual-Attention-Network/SegNeXt/main/LICENSE).
- **Variante:** MSCAN-T + LightHamHead. Conservar ambos: reemplazar la cabeza por una U-Net no reproduce esta variante. [Configuración Tiny ADE20K](https://raw.githubusercontent.com/Visual-Attention-Network/SegNeXt/main/local_configs/segnext/tiny/segnext.tiny.512x512.ade.160k.py) carga `mscan_t.pth`, tiene 150 clases y usa normalizaciones SyncBN/GN.
- **Adaptación:** desactivar carga para experimento aleatorio, cambiar cabeza a un logit y pérdida binaria; adaptar SyncBN para ejecución no distribuida documentando el cambio. Es CNN con atención, no Transformer.
- **Pesos/dependencias:** pesos de encoder ImageNet anunciados; integridad sin comprobar. MMSegmentation 0.24.1 y MMCV requieren entorno aislado o portado validado; no instalar encima del entorno E1. Riesgo medio/alto de integración, coste real pendiente.

### 5. SegFormer-B0

- **Referencia:** [NeurIPS 2021, versión v3](https://arxiv.org/html/2105.15203v3); Q3 Scopus E2-5366. Encoder Transformer jerárquico MiT y decoder MLP.
- **Código/licencia:** [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer); [NVIDIA Source Code License](https://raw.githubusercontent.com/NVlabs/SegFormer/refs/heads/master/LICENSE), uso no comercial de investigación/evaluación. Conservar avisos aplicables.
- **Variante:** [B0 oficial](https://raw.githubusercontent.com/NVlabs/SegFormer/master/local_configs/segformer/B0/segformer.b0.512x512.ade.160k.py), canales [32,64,160,256], dimensión de decoder 256. Configuración original usa `mit_b0.pth`, 150 clases y entropía cruzada multiclase.
- **Adaptación:** RGB, un logit, pérdida binaria explícita, restaurar tamaño espacial antes de medir; no basta cambiar `num_classes` manteniendo una pérdida softmax de una sola clase. Quitar preentrenamiento en rama principal.
- **Dependencias:** README histórico con PyTorch 1.7.1, CUDA 10.1, timm 0.3.2, MMCV 1.2.7 y MMSegmentation 0.13.0. No se ha probado compatibilidad con el PyTorch CPU del E1. Los pesos están enlazados, no descargados ni verificados.
- **Dictamen:** buen primer candidato por variante pequeña y configuración pública; tamaño publicado no garantiza coste local.

### 6. TransUNet 2D

- **Referencia:** [preprint 2021](https://arxiv.org/abs/2102.04306). La extensión Medical Image Analysis 2024 (DOI 10.1016/j.media.2024.103280) no debe mezclarse con una réplica de la versión inicial.
- **Código/licencia:** [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet), [Apache-2.0](https://raw.githubusercontent.com/Beckschen/TransUNet/main/LICENSE).
- **Variante:** R50-ViT-B/16, no ViT puro. [Configuración](https://raw.githubusercontent.com/Beckschen/TransUNet/main/networks/vit_seg_configs.py): Transformer de 12 capas, dimensión 768, 12 cabezas y decoder con skips de CNN. El checkpoint propuesto por la configuración procede de ImageNet-21K, no es una condición equivalente a ImageNet-1K.
- **Adaptación:** RGB, una salida; fijar tamaño 256 y cuadrícula de tokens 16×16. Revisar/interpolar embeddings posicionales al cargar pesos con otra cuadrícula usando la lógica de [carga oficial](https://raw.githubusercontent.com/Beckschen/TransUNet/main/networks/vit_seg_modeling.py). Rama aleatoria sin `load_from`.
- **Dependencias:** [requirements original](https://raw.githubusercontent.com/Beckschen/TransUNet/main/requirements.txt) fija torch 1.4.0/torchvision 0.5.0 e incluye scipy, ml-collections y librerías médicas. Separar dependencias del modelo de las del lector de datos médicos al portar RGB.
- **Dictamen:** riesgo de memoria/tiempo mayor por la variante híbrida base; no hay medida local. README actualiza enlaces de pesos, aún sin descarga comprobada.

### 7. VM-UNet

- **Referencia controlada:** [preprint v2 de 2024](https://arxiv.org/html/2402.02491v2). La ficha editorial TOMM anunciada por el repositorio no se verificó por acceso 403; no asignar cuartil ni metadatos editoriales sin evidencia.
- **Código/licencia:** [JCruan519/VM-UNet](https://github.com/JCruan519/VM-UNet), [Apache-2.0](https://raw.githubusercontent.com/JCruan519/VM-UNet/main/LICENSE).
- **Variante:** [wrapper oficial](https://raw.githubusercontent.com/JCruan519/VM-UNet/main/models/vmunet/vmunet.py), entrada 3 canales, una clase, encoder [2,2,9,2], decoder [2,9,2,2]. La salida binaria ya aplica sigmoid; para el criterio `binary_cross_entropy_with_logits` del proyecto se deben exponer logits, evitando doble sigmoid.
- **Pesos:** artículo usa VMamba-S ImageNet-1K. La carga oficial también remapea capas del encoder al decoder; registrar qué claves realmente se cargan, no etiquetarlo simplemente «encoder preentrenado».
- **Entorno:** README documenta torch 1.13/CUDA 11.7, Triton, causal_conv1d y mamba_ssm con ruedas Linux. No hay evidencia de una ruta equivalente ya probada en nuestro Windows/AMD.
- **Dictamen:** conservar candidato, ejecutar primero prueba mínima en Linux/NVIDIA disponible. Si falla, documentar error y buscar otro SSM; una CNN con atención no conserva esta cobertura de familias. No iniciar servicios pagos ni descargar datos para resolverlo automáticamente.

## Paper guía y alcance verificable de la réplica

**Recomendación: SegFormer, variante B0.** Tiene correspondencia clara entre artículo, configuración y checkpoint; además aporta la familia Transformer. Selección por trazabilidad y presupuesto previsto, no por una puntuación bibliométrica final.

1. **Reproducción de evaluación:** checkpoint B0 ADE20K, clases y protocolo originales, inferencia a una escala. Referencia publicada: **37,4 % mIoU** (38,0 % corresponde a multiescala). Registrar diferencia en puntos porcentuales, versión, procesamiento, checkpoint y hash. Ver [tabla de variantes del artículo](https://arxiv.org/html/2105.15203v3#S4). Proponer investigar diferencias superiores a 0,5 puntos; es un umbral de auditoría nuestro, no garantía estadística ni criterio de los autores.
2. **Réplica de entrenamiento:** sólo podría declararse si se reproduce entrenamiento y evaluación originales, registrando recursos y desviaciones. La primera fase no la sustituye. Estimar coste con el entorno y configuración originales antes de comprometerla; confirmar con el docente si la reproducción de evaluación cumple su requisito.
3. **Experimento propio:** adaptar B0 a HAZMAT binario y contrastarlo con las otras redes. Eso evalúa transferencia de método, no reproduce el 37,4 % de ADE20K.
4. **Limitación a medir:** pérdida de detalle en etiquetas pequeñas/bordes. Definir estratos por fracción de área usando únicamente train, congelarlos y reportar Dice por estrato y una métrica de borde con tolerancia fijada antes del test. No interpretar mejoras sin revisar la calidad de las máscaras de referencia.

No se descargaron ADE20K ni checkpoints. Las fuentes se consultaron en ramas públicas móviles; falta congelar commit, versiones y hashes de pesos. Disponibilidad anunciada no equivale a descarga comprobada.

## Diseño justo para HAZMAT — propuesta a cerrar antes de ejecutar

**Histórico E1:** mantener intactas las 15 corridas. Se entrenaron redes compactas con Adam, tasa 0,001, máximo 30 épocas, paciencia 5 y semillas 11,23,37,53,71. Es una referencia histórica, no un control equivalente a cualquier entrenamiento nuevo.

**Comparación principal E2:** las diez variantes desde inicialización aleatoria, mismos manifiestos/splits agrupados, resolución 256, máscaras, augmentations y semilla de datos. Conservar logits binarios y criterio en píxeles válidos del E1. PIDNet requiere documentar objetivos y pesos auxiliares; tratarlos como parte del método, no ocultarlos como si todas las pérdidas fueran idénticas. Mantener pesos externos desactivados de forma verificable.

**Selección por validación:** presupuesto propuesto de dos configuraciones de optimización por modelo con semilla 11 y máximo 30 épocas/paciencia 5; fijar esas configuraciones por escrito antes de lanzarlas. Usar recetas adecuadas a cada familia, no imponer 0,001 a todos. Elegir por Dice agrupado de validación y correr la ganadora en las cinco semillas comunes. Si cambian reglas frente al E1, volver a correr las tres bases bajo el mismo presupuesto para la tabla principal; el histórico permanece aparte. No se afirma que este presupuesto produzca la mejor versión posible de cada red.

**Preentrenamiento secundario:** si hay recursos, estudiar pares con/sin pesos por arquitectura. Separar ImageNet-1K, ImageNet-21K y preentrenamiento de segmentación; registrar normalización, capas cargadas y datos adicionales. Una tabla mezclada no identifica un efecto puro de arquitectura. En la rama sin pesos, emplear procesamiento común; en la preentrenada respetar el procesamiento de sus pesos y declarar la diferencia.

**Medidas:** principal Dice foreground con igual peso por grupo, resumen entre semillas; secundarias IoU foreground, mIoU binaria, bordes y coste. Excluir padding en pérdidas y métricas. Fijar umbral 0,5 y convención para máscaras vacías antes de test. Para incertidumbre usar grupos como unidades de remuestreo y reportar variación por semilla aparte, evitando tratar píxeles como observaciones independientes.

**Test ya utilizado:** una inferencia en modo evaluación no entrena pesos, siempre que no ejecute optimización ni actualice estadísticas de entrenamiento. Sin embargo, mirar resultados puede orientar decisiones humanas. Mantener el test histórico congelado y declarar exposición previa; seleccionar por train/val y evaluar cada configuración final una vez. No volver a repartir datos para aparentar un test nuevo. Si se requiere evidencia totalmente independiente, reservar datos nuevos de captura independiente en una etapa posterior.

## Puertas de ejecución y presupuesto

1. Congelar código por commit y dependencias en entorno separado; conservar licencias de código, y revisar condiciones de cada peso/dataset antes de incorporarlo. Este documento no es una autorización para entrenar o descargar datasets.
2. Probar forward/backward de cada adaptador con datos sintéticos: RGB 256, tamaño de salida, logits finitos, pérdida y gradientes, ausencia de carga de pesos en la rama aleatoria. Comprobar que padding y objetivos auxiliares no introducen etiquetas falsas en bordes.
3. Probar lote físico 2 cuando BatchNorm lo necesite; acumulación de gradientes no corrige estadísticas de un lote físico 1. Cambiar normalización modifica el método y debe registrarse. No congelar BatchNorm aleatoria como arreglo silencioso.
4. Medir parámetros, pico RAM/VRAM y tiempo por paso en cada entorno; extrapolar con número de pasos reales y reportar estimación, no resultado de entrenamiento. Comparar velocidad únicamente en el mismo dispositivo y condiciones.
5. Orden sugerido: SegFormer/PIDNet, DeepLabv3+/UNet++, SegNeXt, TransUNet y VM-UNet. Para los dos últimos priorizar GPU; disponibilidad y presupuesto todavía pendientes. La prueba de compatibilidad VM-UNet puede hacerse antes para detectar temprano un bloqueo de la familia SSM.
6. Cerrar el diseño y autorizar ejecución antes del piloto. Mantener test fuera del desarrollo. Ninguna de estas puertas se declara superada en esta revisión documental.

Pendientes bibliográficos independientes: cribado completo, puntuación, tres conflictos DOI, confirmación de diez modelos totales y aceptación de cuatro familias, y requisito transversal reducido al omitir Q5/Q6. Esta auditoría no resuelve esos puntos por sí sola.
