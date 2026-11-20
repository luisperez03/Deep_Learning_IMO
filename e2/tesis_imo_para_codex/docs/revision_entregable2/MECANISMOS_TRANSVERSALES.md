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
