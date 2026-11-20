# Alcance confirmado — 6 de noviembre de 2026

Entrega: domingo 8 de noviembre. Único recurso: portátil local. El investigador
confirmó que se aceptan baselines generales y autorizó continuar con la prueba
mínima tras cerrar aplicaciones. Esto no equivale a validar capacidad para todas
las corridas. Los bloqueos del preflight general siguen vigentes; la autorización
de esta prueba acotada se registra aquí por separado.

La tarea inmediata es segmentación semántica binaria: cada píxel observado es
fondo o etiqueta. No identifica el tipo de mercancía peligrosa y tampoco separa
necesariamente dos etiquetas que se toquen. El corpus seleccionado contiene
etiquetas; no valida por sí solo rechazo de imágenes completamente negativas.

Después: revisar correspondencias semánticas y discrepancias de anotación,
confirmar el mapa de clases HAZMAT a IMO/IMDG y entrenar clasificación de recortes
o segmentación multiclase. Evaluar clasificación primero con recortes anotados
y luego con recortes predichos para distinguir sus errores de los de localización.
Conservar los mismos grupos de partición para evitar fuga entre ambas etapas.
No convertir automáticamente nombres HAZMAT en clases IMO oficiales.

## Implementación inicial para revisar

`scripts/modelos_base.py` contiene tres adaptaciones desde cero con canales
16/32/64, dos reducciones espaciales y convoluciones 3x3 con ReLU:

- FCN: mapa grueso y ampliación bilineal, sin conexiones laterales.
- SegNet: recuperación espacial mediante índices de pooling.
- U-Net: concatenación de características del encoder al decoder.

Familias sustentadas por [FCN](https://arxiv.org/abs/1411.4038),
[SegNet](https://arxiv.org/abs/1511.00561) y
[U-Net](https://arxiv.org/abs/1505.04597). Anchos y profundidad reducidos son
decisiones locales por presupuesto; no se presentan como réplicas originales
ni como una revisión bibliográfica exhaustiva. No se iguala artificialmente el
número de parámetros: se deberá reportar como diferencia entre modelos.

`scripts/prueba_carga.py --model unet` intenta cinco actualizaciones sobre cinco
imágenes de train, con lote 1, resolución 256, dos hilos, semilla 11, Adam y
BCE + Dice. No lee test ni guarda pesos. Repetir con `fcn` y `segnet` solamente
cuando la reserva y el entorno estén validados. Se registra versión de torch,
parámetros y tiempo por paso. Una prueba tan corta no estima confiablemente el
tiempo total de épocas ni el consumo térmico sostenido.

Estado actual: sintaxis Python verificada; ejecución numérica pendiente de
entorno PyTorch y memoria disponible. La prueba rechaza iniciar bajo 2 GiB
libres; no debe reducirse ese límite solo para forzar la ejecución. No hay
resultados de modelos ni evaluación comparativa todavía.
