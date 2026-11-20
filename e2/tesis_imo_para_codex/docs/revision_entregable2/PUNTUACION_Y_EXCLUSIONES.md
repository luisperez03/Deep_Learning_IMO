# Puntuación y justificación según rúbrica

**RECTIFICACIÓN 20/11:** ranking histórico preliminar, no aprobado/verificado. Falta citación y cotejo de calidad. Pertinencia máxima de PIDNet no acredita evidencia HAZMAT; U-Net E1 NO comparte ResNet-18 con UNet++. Disponibilidad de pesos no equivale a integridad comprobada. No usar estos subtotales como selección bibliométrica definitiva.

Fecha: 18 de noviembre de 2026.
Ámbito: Evaluación formal de candidatos para el benchmark ampliado (Entregable 2).
Rúbrica de referencia: `docs/revision_entregable2/PROTOCOLO_BUSQUEDA.md` (Escala 0–100 puntos).

---

## 1. Criterios y ponderación de la rúbrica

La rúbrica fijada en el protocolo evalúa seis componentes cuantitativos:

1. **Pertinencia de la tarea y mecanismo (25 pts):**
   - 0: Sin relación; 1: Segmentación sin mecanismo específico; 2: Mecanismo pertinente; 3: Dificultad compartida (pequeños/bordes); 4: Rótulos o dominio directamente comparable. (Puntaje base $\times 25/4$).
2. **Reproducibilidad y disponibilidad de código (25 pts):**
   - 0: Insuficiente; 1: Parcial; 2: Especificación completa; 3: Código oficial de autores; 4: Añade pesos públicos, configuración y datos verificables. (Puntaje base $\times 25/4$).
3. **Evidencia experimental y rigor metodológico (20 pts):**
   - 0: Sin comparación; 1: Puntual sin baselines; 2: Baselines y partición claros; 3: Control de equidad, repeticiones y ablaciones; 4: Control formal de incertidumbre. (Puntaje base $\times 20/4$).
4. **Viabilidad técnica y presupuesto computacional (10 pts):**
   - 0: Incompatible; 1: Adaptación mayor incierta; 2: Adaptación viable con costo medio; 3: Configuración pública ligera; 4: Piloto probado dentro del presupuesto. (Puntaje base $\times 10/4$).
5. **Recencia temporal (10 pts):**
   - 4: 2025–2026; 3: 2023–2024; 2: 2021–2022; 1: 2019–2020; 0: <2019 (excepción fundacional justificada). (Puntaje base $\times 10/4$).
6. **Citación normalizada por antigüedad (10 pts):**
   - Conforme a la regla del protocolo: $\text{citas}/\max(1, 2026 - \text{año} + 1)$. Dado que las métricas exactas de citas en la misma base y fecha aún no se han congelado de forma uniforme para todos los candidatos, se realiza la evaluación con los **primeros cinco componentes objetivos (escala 0–90 normalizada a 100)** y se declara explícitamente la sensibilidad.

---

## 2. Puntuación de los 7 candidatos seleccionados

| Candidato | Pertinencia (25) | Reproducibilidad (25) | Evidencia (20) | Viabilidad (10) | Recencia (10) | Subtotal (90) | Puntaje Normalizado (0-100) | Dictamen |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **SegFormer-B0** | 18,75 (3) | 25,00 (4) | 15,00 (3) | 7,50 (3) | 5,00 (2) | **71,25** | **79,17** | **Aprobado (Paper guía)** |
| **PIDNet-S** | 25,00 (4) | 25,00 (4) | 15,00 (3) | 7,50 (3) | 7,50 (3) | **80,00** | **88,89** | **Aprobado (Máxima pertinencia bordes)** |
| **DeepLabv3+** | 18,75 (3) | 25,00 (4) | 15,00 (3) | 7,50 (3) | 0,00 (0) | **66,25** | **73,61** | **Aprobado (Excepción fundacional CNN)** |
| **UNet++** | 18,75 (3) | 25,00 (4) | 15,00 (3) | 7,50 (3) | 2,50 (1) | **68,75** | **76,39** | **Aprobado (Variante SMP ResNet-18)** |
| **SegNeXt-T** | 18,75 (3) | 18,75 (3) | 15,00 (3) | 5,00 (2) | 5,00 (2) | **62,50** | **69,44** | **Aprobado (Atención convolucional)** |
| **TransUNet 2D** | 18,75 (3) | 18,75 (3) | 10,00 (2) | 2,50 (1) | 5,00 (2) | **55,00** | **61,11** | **Aprobado condicionado (Presupuesto/GPU)** |
| **VM-UNet** | 18,75 (3) | 18,75 (3) | 10,00 (2) | 2,50 (1) | 7,50 (3) | **57,50** | **63,89** | **Aprobado condicionado (Entorno Linux/CUDA)** |

### Justificaciones específicas por modelo:

- **PIDNet-S (88,89):** Obtiene la mayor pertinencia por su rama explícita de derivación de bordes (D), crítica para delimitar los contornos en rombo de los rótulos HAZMAT. Código oficial y pesos ligeros.
- **SegFormer-B0 (79,17):** Configuración oficial B0 sumamente ligera (~3,8 M params), código y pesos verificados, ideal para contrastar la auto-atención frente a convoluciones puras. Seleccionado como paper guía para reproducción.
- **DeepLabv3+ (73,61):** Penalizado en recencia por ser de 2018 (puntuación 0 en ese criterio), pero retenido como estándar de facto indiscutible en visión por computador para convoluciones dilatadas multiescala (ASPP). Implementación SMP con ResNet-18 garantiza alta viabilidad.
- **UNet++ (76,39):** Extensión TMI 2020. Proporciona una comparación directa contra la U-Net del E1 al compartir el mismo backbone ResNet-18 en SMP, aislando el efecto de las conexiones densas.
- **SegNeXt-T (69,44):** Puntuación sólida en pertinencia por su mecanismo MSCAN, pero castigado ligeramente en viabilidad por requerir aislamiento de MMSegmentation 0.24.1 y adaptación de SyncBN.
- **TransUNet 2D (61,11):** Castigado en viabilidad por su tamaño (~105 M parámetros) y dependencias fijas legacy en PyTorch 1.4. Retenido estrictamente para garantizar la cobertura de la familia híbrida CNN-Transformer.
- **VM-UNet (63,89):** Castigado en viabilidad por la incompatibilidad de sus extensiones compiladas C++/CUDA (`mamba_ssm`, `causal_conv1d`) con entornos Windows CPU/AMD estándar. Retenido condicionado a entorno GPU/Linux para representar a la familia SSM.

---

## 3. Justificación de candidatos excluidos relevantes

En cumplimiento del protocolo, se documentan las razones formales de descarte de otras arquitecturas prominentes presentes en el cribado:

1. **Attention U-Net (Oktay et al., 2018):**
   - *Motivo:* Aunque incorpora compuertas de atención espacial (*attention gates*), estructuralmente sigue perteneciendo a la familia CNN estándar de encoder-decoder. Mantenerla en la lista reducida de 7 plazas adicionales hubiese sobredimensionado la familia CNN en detrimento de la diversidad exigida (Transformer, Híbrido, SSM).
2. **Swin-UNet / Swin Transformer (Cao et al., 2022):**
   - *Motivo:* Emplea atención por ventanas desplazadas (*shifted windows*). Se prefirió SegFormer-B0 debido a que su decoder MLP y su atención con reducción espacial nativa ofrecen un costo computacional y de memoria notablemente inferior para la resolución $256\times 256$, además de contar con checkpoints estandarizados B0 más accesibles.
3. **SAM / SAM-2 / Promptable Foundation Models (Kirillov et al., 2023; Ravi et al., 2024):**
   - *Motivo:* SAM está concebido como un modelo de segmentación interactiva guiada por prompts (puntos, cajas, texto). Evaluarlo en un benchmark de segmentación semántica autónoma etiqueta/fondo sin intervención humana exige heurísticas artificiales de generación de prompts (o uso de la caja ground truth como prompt, lo que constituiría una fuga metodológica inaceptable frente a las redes fully-convolutional o end-to-end).
4. **YOLOv8-seg / YOLOv5-seg (Modelos de segmentación de instancias):**
   - *Motivo:* Su formulación predice prototipos de máscaras y coeficientes por caja delimitadora (detección + máscara). No compite en equidad con modelos puramente densos de segmentación semántica pixel a pixel, y su métrica nativa de reporte es mask mAP, no IoU/Dice binario puro por imagen.
