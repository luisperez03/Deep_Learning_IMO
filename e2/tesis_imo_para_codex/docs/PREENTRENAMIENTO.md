# Preparación técnica antes de los modelos base

**Estado: datos derivados y protocolo preparados; entrenamiento deshabilitado.**
Esta versión es una propuesta reproducible para revisión, no una certificación de
calidad perfecta ni una selección bibliográfica final de modelos.

![Calidad](../resultados/preentrenamiento/graficos/01_calidad.png)

![Particiones](../resultados/preentrenamiento/graficos/02_particiones.png)

![Grupos](../resultados/preentrenamiento/graficos/03_grupos.png)

## Resultado verificado

- 1451 pares técnicamente elegibles; 132 en cuarentena.
- 102 imágenes sin máscara utilizable.
- Train: 1015 imágenes / 567 grupos.
- Validación: 218 imágenes / 122 grupos.
- Prueba: 218 imágenes / 122 grupos.
- Enlaces candidatos conocidos que cruzan particiones: 0.
- Grupos de similitud en todo el corpus: 937; tamaño máximo 3.

La cuarentena propaga alertas a las imágenes relacionadas: discrepancias semánticas,
posibles omisiones y cajas con muy poca cobertura de máscara. Es una política
conservadora que puede excluir siluetas útiles; no afirma que todos esos casos sean
errores. Está registrada y se puede revisar sin tocar los originales.

## Artefactos de esta versión

`resultados/preentrenamiento/manifest.csv` registra grupo, estado, motivo, split,
rutas y hashes. `train.csv`, `val.csv`, `test.csv` son sus vistas de trabajo.
`masks_binary/` conserva copias 0/1 a resolución original; no son cajas convertidas
en máscaras. Solo se cargan los archivos listados en el manifiesto, nunca glob
de todas las copias derivadas.

Normalización calculada únicamente sobre train, después de letterbox bilineal
256×256 y excluyendo padding: `normalization_train.json`. El cargador convierte
RGB a CHW float32, mantiene máscaras 0/1 y devuelve la zona válida. No se activa
augmentación ni se cachea el dataset completo en RAM.

El diagnóstico en train compara 192, 256 y 320: ningún primer plano completo
desaparece ni queda por debajo de 16 píxeles totales. Esto NO demuestra que cada
etiqueta pequeña individual sobreviva: el informe es por imagen, no por componente.
256 es una resolución candidata sujeta a revisar detalle y costo.

## Identidad del test reservado

SHA256 de test.csv: `7340a0cab140a8180b4a669260809eb79e86be7cb43ef87f5cac9bbd22317e81`.

El generador rechaza cambios silenciosos de entradas/configuración y de IDs del
test existente. Cualquier depuración posterior exige una nueva versión documentada.
El cargador niega el acceso al test por defecto. No hay evaluación de modelos.

## Validaciones antes de autorizar entrenamiento

1. Revisar agrupamiento conservador y política de cuarentena. No afirmar ausencia
   absoluta de fuga: faltan escenas verificadas y variantes no detectadas por hashes.
2. Aprobar margen delta, métricas, hipótesis, arquitectura y presupuesto experimental.
3. Resolver selección de tres baselines mediante literatura; los candidatos actuales
   no están demostrados como los tres más recurrentes en HAZMAT.
4. Validar recursos del PC y entorno PyTorch CPU aislado. No instalar ni arrancar
   modelos sin acordarlo. Plan de prueba mínima descrito en PC_Y_PRESUPUESTO.md.

Guías: `PREENTRENAMIENTO_REPRODUCIBLE.md`, `PROTOCOLO_PREVIO.md`,
`PC_Y_PRESUPUESTO.md`, `PROCEDENCIA_Y_LITERATURA.md`.
