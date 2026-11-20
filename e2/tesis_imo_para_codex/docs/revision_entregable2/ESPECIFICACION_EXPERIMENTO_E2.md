# Especificación del Experimento E2: Benchmark de 10 Arquitecturas en Colab

**Actualización v2:** el texto inferior describe la propuesta anterior y contiene garantías objetadas en `REVISION_CAMBIOS_20261118.md`. Para ejecutar, prevalecen `../../config/experimento_e2.json` y `../../COLAB_E2_LEEME.md`: arquitecturas reales, recetas explícitas, lote nominal 2 (cola de 3), preflight obligatorio y checkpoints con firma. No interpretar este documento como evidencia de aprobación docente o ejecución.

Fecha: 18 de noviembre de 2026.
Entorno de ejecución: VS Code conectado a GPU remota (Google Colab).
Tarea: Segmentación semántica binaria etiqueta/fondo de rótulos HAZMAT/IMDG.

---

## 1. Requisitos de resiliencia y persistencia ante desconexiones

Al ejecutar desde Visual Studio Code conectado a un runtime remoto de Colab, las desconexiones de red, caídas de WebSocket o reinicios de máquina virtual son eventos previstos. El diseño del experimento garantiza:

1. **Persistencia atómica por época:**
   - Al finalizar cada época se guarda un archivo temporal `last.tmp` que se reemplaza de forma atómica (`os.replace`) por `last.pt`.
   - Se preserva el mejor modelo en `best.pt` junto a sus predicciones en validación `best_val_predictions.json`.
   - Si la conexión se corta a mitad de una época, al reiniciar el runner detecta `last.pt`, carga el modelo, el optimizador, el contador de época y restaura los estados de los generadores pseudoaleatorios (`torch_rng` y `cuda_rng`), reanudando desde el inicio de esa época sin duplicar pasos ni corromper las métricas.

2. **Sincronización con Google Drive / almacenamiento persistente:**
   - El directorio de salida debe ubicarse en `/content/drive/MyDrive/tesis_imo_experimentos_e2/` (o sincronizarse periódicamente si se corre en disco local `/content/`).
   - Cada combinación `(modelo, semilla)` posee su propia subcarpeta aislada con su `config.json`, `history.json`, `last.pt`, `best.pt` y `result.json`.

3. **Inmutabilidad y verificación de configuración:**
   - Si se reanuda una corrida, el runner compara el `config.json` existente contra los parámetros de invocación. Cualquier cambio en tamaño, tasa o manifest aborta de inmediato para evitar mezclar épocas con hiperparámetros distintos.

---

## 2. Modelos, familias e hiperparámetros del benchmark

El benchmark evalúa 10 modelos bajo inicialización aleatoria (sin preentrenamiento externo en la rama principal), resolución $256\times 256$, lote 1 (o lote físico 2 cuando BatchNorm lo exija), máximo 30 épocas, paciencia 5 y 5 semillas comunes (11, 23, 37, 53, 71):

| # | Modelo | Familia | Implementación / Adaptador | Optimizador base | Tasa inicial ($\eta$) |
|---|---|---|---|---|---|
| 1 | **FCN-8s** | CNN clásica | `SmallFCN` (Línea base E1) | Adam | 0,001 |
| 2 | **SegNet** | CNN clásica | `SmallSegNet` (Línea base E1) | Adam | 0,001 |
| 3 | **U-Net** | CNN clásica | `SmallUNet` (Línea base E1) | Adam | 0,001 |
| 4 | **DeepLabv3+** | CNN multiescala | SMP `DeepLabV3Plus` (ResNet-18, 1 canal logit) | AdamW / Adam | 0,0005 |
| 5 | **UNet++** | CNN anidada | SMP `UnetPlusPlus` (ResNet-18, 1 canal logit) | Adam | 0,0005 |
| 6 | **PIDNet-S** | CNN multirrama | Oficial adaptado (3 ramas: P, I, D de borde) | SGD / AdamW | 0,001 |
| 7 | **SegNeXt-T** | CNN atención | Oficial adaptado (MSCAN-T + LightHamHead) | AdamW | 0,0001 |
| 8 | **SegFormer-B0** | Transformer | Oficial adaptado (MiT-B0 + MLP, Paper guía) | AdamW | 0,00006 |
| 9 | **TransUNet 2D** | Híbrido | Oficial adaptado (R50-ViT-B/16 a $256\times 256$) | SGD con momentum | 0,001 |
| 10| **VM-UNet** | SSM (Mamba) | Oficial adaptado (VSSM bidireccional) | AdamW | 0,001 |

---

## 3. Función de pérdida y protocolo de evaluación

- **Pérdida común:**
  $$\mathcal{L} = \frac{\mathcal{L}_{\text{BCEWithLogits}} + \mathcal{L}_{\text{SoftDice}}}{2}$$
  calculada **únicamente sobre píxeles válidos** (excluyendo estrictamente el padding del letterbox).
- **Métrica de selección (validación):** Dice foreground ponderado de forma idéntica por grupo de similitud:
  $$\text{Dice}_{\text{val}} = \frac{1}{|G_{\text{val}}|} \sum_{g \in G_{\text{val}}} \left( \frac{1}{|g|} \sum_{i \in g} \text{Dice}_i \right)$$
- **Test reservado:** Congelado. No se toca durante la exploración ni para seleccionar checkpoints.

---

## 4. Instrucciones operativas para ejecutar en Colab desde VS Code

1. En VS Code, conectarse a Colab mediante la extensión oficial de Jupyter o un túnel SSH/servidor Jupyter remoto (`https://colab.research.google.com/`).
2. Montar Google Drive para garantizar persistencia:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Ejecutar el script orquestador `scripts/experimento_e2_colab.py`.
