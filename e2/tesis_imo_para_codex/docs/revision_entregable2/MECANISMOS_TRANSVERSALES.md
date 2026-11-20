# Mecanismos entre modalidades y transferencia teórica

Actualizado: 18 de noviembre de 2026.
Ámbito: Entregable 2 — Revisión bibliográfica y selección de arquitecturas.

---

## 1. Alcance y delimitación metodológica

La búsqueda sistemática de literatura (protocolo `PROTOCOLO_BUSQUEDA.md`) contempló inicialmente tres preguntas transversales orientadas a rastrear la migración de principios matemáticos desde otras modalidades hacia visión por computador (segmentación semántica):
- **Atención sobre secuencias** (procesamiento de lenguaje natural y traducción hacia visión).
- **Modelos de espacio de estados (SSM)** (series temporales y señales continuas hacia visión bidimensional).
- **Convoluciones dilatadas / multiescala** (procesamiento de voz/audio y señales 1D hacia campos receptivos 2D).

Por instrucción explícita del usuario, las consultas sistemáticas automatizadas Q5 (SSM y series) y Q6 (dilatación y audio) fueron omitidas tras fallos de sesión institucional. En consecuencia, el corpus indexado Q0–Q4 se complementa con la evidencia primaria documental extraída directamente de los artículos fundacionales y preprints de referencia.

> **Advertencia metodológica:** No debe confundirse una analogía funcional con una genealogía histórica demostrada. Por ejemplo, la aplicación de convoluciones dilatadas en *WaveNet* (2016) para síntesis de audio cita antecedentes directos en análisis de imágenes (*algorithme à trous* de Holschneider et al., 1989; Yu & Koltun, 2015). Por tanto, la dilatación representa un mecanismo matemático compartido de expansión de campo receptivo sin pérdida de resolución, no una transferencia unidireccional audio → visión.

---

## 2. Matriz de mecanismos y justificación para rótulos HAZMAT

| Mecanismo matemático | Origen / Dominio de referencia | Arquitectura en el benchmark | Hipótesis teórica para rótulos HAZMAT binarios | Desafío y riesgo de transferencia |
|---|---|---|---|---|
| **Auto-atención espacial multiescala (Self-Attention)** | Modelado de secuencias y NLP (*Vaswani et al., 2017*) | **SegFormer-B0** (MiT-B0 + MLP) | Captura dependencias contextuales de largo alcance para desambiguar el rótulo respecto a fondos industriales complejos (óxido, hendiduras de contenedores). Permite relacionar esquinas distantes sin apilar decenas de capas convolucionales. | Pérdida de resolución espacial fina en etapas tempranas si el parche o la reducción espacial es excesiva; degradación en la delimitación precisa de bordes de etiquetas pequeñas. |
| **Modelos de Espacio de Estados Continuos Discretizados (SSM)** | Teoría de control, procesamiento de señales y series (*Gu & Dao, Mamba 2023*) | **VM-UNet** (VSSM con SSM bidireccional) | Modelado de contexto global con complejidad computacional y de memoria lineal $O(N)$ frente a la cuadrática $O(N^2)$ de los Transformers estándar. Proporciona campo receptivo receptivo global continuo. | Sensibilidad direccional (el escaneo 2D debe simularse mediante múltiples pasadas: horizontal, vertical y diagonales); alta fricción de portabilidad/entorno (kernels CUDA especializados no portables a Windows/AMD estándar). |
| **Pirámides de Convoluciones Dilatadas (ASPP)** | Procesamiento de señales multiescala y síntesis (*Yu & Koltun 2015, WaveNet 2016*) | **DeepLabv3+** (ResNet-18 + ASPP) | Muestreo de características a múltiples tasas de dilatación ($\text{rate} \in \{1, 6, 12, 18\}$), preservando la resolución espacial nativa mientras se amplía el campo receptivo para discriminar escala. | Convoluciones atrous con tasas elevadas degeneran en filtros $1\times 1$ si el objeto es menor que el paso de muestreo ("gridding effect"); exige adaptar las tasas a la resolución $256\times 256$. |
| **Bucle de retroalimentación de tres ramas (Análogo a Control PID)** | Teoría clásica de control y sistemas dinámicos continuos (*Xu et al., 2023*) | **PIDNet-S** | Desacoplamiento explícito de componentes: Rama Proporcional P (detalle de alta resolución), Rama Integral I (contexto de baja frecuencia), y Rama Derivada D (gradiente de alta frecuencia / bordes finos del rombo). | Requiere diseño y cálculo en línea de mapas de verdad terreno para la componente D (fronteras derivadas por Laplaciano/Sobel exclusivamente del train); balance hiperparamétrico sensible entre pérdidas auxiliares. |
| **Atención convolucional multiescala (MSCAN)** | Optimización convolucional inspirada en ViT (*Guo et al., 2022*) | **SegNeXt-T** | Combina la inducción de sesgo local propia de las CNN con campos receptivos extensos mediante convoluciones depthwise desacopladas en tiras ($1\times 7$, $7\times 1$, etc.), evitando el costo de atención densa. | Dependencias de sincronización de estadísticas (SyncBN en implementaciones originales) que requieren adaptación controlada a normalización estándar para ejecución monodispositivo. |
| **Puente Híbrido CNN-Transformer** | Diagnóstico médico multiescala (*Chen et al., 2021*) | **TransUNet 2D** (R50-ViT-B/16) | El encoder CNN extrae texturas y bordes superficiales; el cuello de botella Transformer modela relaciones globales entre partes del rótulo y el vehículo portador. | Gran volumen de parámetros (~105 M); embeddings posicionales fijos sensibles a la geometría de entrada que requieren interpolación bilineal adaptada. |
| **Conexiones densas anidadas y rediseñadas** | Topología de grafos densos (*Zhou et al., 2020*) | **UNet++** (ResNet-18) | Reduce la brecha semántica entre mapas de características del encoder y decoder mediante nodos convolucionales intermedios interconectados, facilitando la agregación multiescala. | Incremento notable de memoria por acumulación de activaciones intermedias; la adaptación en SMP prescinde de la poda de ramas para mantener reproducibilidad cerrada. |

---

## 3. Condiciones de diseño para contrastar las 4 familias

La selección de las siete redes adicionales, sumadas a los baselines del Entregable 1 (**FCN**, **SegNet**, **U-Net**), asegura la cobertura equilibrada de **cuatro familias arquitectónicas distintas**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        COBERTURA DE 4 FAMILIAS                         │
├─────────────────────┬──────────────────────────────────────────────────┤
│ Familia             │ Arquitecturas representativas                    │
├─────────────────────┼──────────────────────────────────────────────────┤
│ 1. CNN Clásica y    │ • FCN-8s (Línea base E1)                         │
│    Multirrama       │ • SegNet (Línea base E1)                         │
│                     │ • U-Net (Línea base E1)                          │
│                     │ • DeepLabv3+ (ASPP + decoder)                    │
│                     │ • UNet++ (Skip connections anidados)             │
│                     │ • PIDNet-S (Tres ramas desacopladas)             │
│                     │ • SegNeXt-T (CNN con atención MSCAN)             │
├─────────────────────┼──────────────────────────────────────────────────┤
│ 2. Transformer Puro │ • SegFormer-B0 (MiT-B0 + MLP, Paper guía)        │
├─────────────────────┼──────────────────────────────────────────────────┤
│ 3. Híbrido CNN-     │ • TransUNet 2D (ResNet-50 + ViT-B/16)            │
│    Transformer      │                                                  │
├─────────────────────┼──────────────────────────────────────────────────┤
│ 4. State Space      │ • VM-UNet (VSSM bidireccional)                   │
│    Models (SSM)     │                                                  │
└─────────────────────┴──────────────────────────────────────────────────┘
```

Esta matriz fundamenta teóricamente que las diferencias observadas en el benchmark no responderán únicamente a la escala de parámetros, sino al **principio matemático** que rige el flujo de información y la preservación espacial de bordes en objetos pequeños.

---

## 4. Síntesis de los tres mecanismos transversales (20/11/2026)

Fuentes con DOI verificado en DataCite (`evidencia_doi_registro.json`); cifras de los artículos del benchmark cotejadas con página en `RESULTADOS_ORIGINALES_7REDES.md`. Q5 y Q6 siguen omitidas: esta síntesis se apoya en las fuentes primarias de cada mecanismo, no en una búsqueda transversal ejecutada, y por eso no cierra la fila 24 del checklist.

| | Atención (secuencias → visión) | Estado selectivo (series → visión) | Dilatación (señal 1D ↔ imagen) |
|---|---|---|---|
| Fuente primaria | Vaswani et al. (2017), `10.48550/arXiv.1706.03762` | Gu y Dao (2023), Mamba, `10.48550/arXiv.2312.00752` | Yu y Koltun (2015), `10.48550/arXiv.1511.07122`; WaveNet, van den Oord et al. (2016), `10.48550/arXiv.1609.03499` |
| Operación | $\mathrm{softmax}(QK^\top/\sqrt{d})\,V$: cada posición pondera a todas las demás | $h_t=\bar A h_{t-1}+\bar B_t x_t,\ y_t=C_t h_t$, con $\bar B_t$, $C_t$ y el paso dependientes de la entrada | $y[i]=\sum_k x[i+r\,k]\,w[k]$: el mismo filtro con huecos de tamaño $r$ |
| Qué transfiere | Relaciones entre posiciones lejanas en una sola capa | Contexto global con costo lineal en la longitud de la secuencia | Campo receptivo grande sin reducir la resolución |
| Costo que trae | Cuadrático en el número de tokens; SegFormer lo reduce con reducción espacial, a cambio de detalle | La imagen debe volverse secuencia: el orden de escaneo es una decisión de diseño; kernels CUDA específicos | Con tasas grandes sobre mapas chicos, los pesos caen fuera del mapa |
| Redes del benchmark | SegFormer-B0 (MiT), TransUNet (ViT-B/16 sobre ResNet-50). SegNeXt-T usa atención **convolucional**, no autoatención | VM-UNet (bloques VSS con escaneo 2D) | DeepLabv3+ (ASPP; r = 6, 12, 18 en el artículo, 12, 24, 36 en la implementación) |
| Evidencia en los artículos originales | En la misma tabla y protocolo (ADE20K, 512 × 512), SegNeXt-T llega a 41,1 mIoU con 4,3 M parámetros y SegFormer-B0 a 37,4 con 3,8 M (SegNeXt, T. 9, p. 9): a esa escala, la autoatención no es necesaria para el contexto | En Synapse, VM-UNet reporta DSC 81,08 frente a 77,48 de TransUNet (VM-UNet, T. II, p. 6), con cifras recopiladas y sin repeticiones | DeepLabv3+ mejora con decodificador y convolución separable en VOC val (T. 5, p. 12); ninguna tabla aísla la dilatación en objetos pequeños |
| Hipótesis para HAZMAT | El contexto global separa el rótulo de fondos con colores y bordes parecidos | Contexto global a menor costo que la atención | Ver el rombo entero y su entorno sin perder la resolución de 1/16 |
| Qué la refuta en el benchmark | SegFormer-B0 y TransUNet sin ventaja de Dice sobre las CNN del mismo presupuesto, sobre todo en el grupo de rótulos pequeños | VM-UNet sin ventaja de Dice sobre TransUNet o con más costo medido por imagen | DeepLabv3+ sin ventaja sobre U-Net en rótulos pequeños |

**Advertencia aritmética para la dilatación.** `scripts/modelos_e2.py` crea DeepLabv3+ con los valores por defecto de `segmentation_models_pytorch`. En la versión 0.6.0 (leída; el repositorio fija 0.5.0, que conviene confirmar) son paso de salida 16 y tasas ASPP (12, 24, 36), no las (6, 12, 18) del artículo. Con entrada de 256 × 256, ASPP opera sobre un mapa de 16 × 16. Con r = 24 y r = 36, los ocho pesos no centrales de cada filtro 3 × 3 caen fuera del mapa en toda posición y leen solo relleno: esas dos ramas se reducen a convoluciones 1 × 1. Con r = 12, cada posición ve a lo sumo un vecino por eje. En este benchmark, la dilatación de DeepLabv3+ aporta poco contexto; un resultado suyo no prueba ni refuta el mecanismo.

**Lo que esta síntesis no afirma.** Las tres columnas son analogías funcionales entre modalidades, no genealogías demostradas (la dilatación ya existía en análisis de imagen antes de WaveNet). Ninguna cifra de los artículos originales se compara con el Dice de HAZMAT-13: los datasets, resoluciones y preentrenamientos son otros. La prueba de cada hipótesis es la fila "qué la refuta", con los resultados del benchmark y sus intervalos.
