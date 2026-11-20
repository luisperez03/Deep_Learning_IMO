# Réplica del paper guía y limitación cuantificada (ítems 32 y 33)

Estado al 2026-11-20: **protocolos fijados y código probado con datos simulados; ninguna de las dos
partes se ha ejecutado con datos reales.** No hay todavía un mIoU de la réplica ni un veredicto de la
limitación. Lo que sigue dice qué corre, en qué orden, cuánto tarda y qué falta.

## Archivos

| Archivo | Qué es |
|---|---|
| `config/replica_segformer.json` | Protocolo de la réplica, escrito antes de cualquier resultado: checkpoint, datos, preprocesamiento, métrica, IC, criterio y análisis de la desviación (D1-D5). Diseño del paso 2. |
| `config/limitacion_segformer_area.json` | Protocolo de la limitación: enunciado, mecanismo, redes piloto, terciles, estadísticos, criterio y la evidencia previa de test que ya se había visto. |
| `scripts/replica_segformer_ade.py` | Evaluación de SegFormer-B0 en ADE20K: preprocesamiento, mIoU acumulado, bootstrap por imagen, veredicto. |
| `scripts/crear_colab_replica.py` | Generador determinista del cuaderno; incrusta el script y el protocolo con su SHA-256. |
| `colab/Replica_SegFormer_ADE20K.ipynb` | Cuaderno de Colab que ejecuta la réplica (generado, no editar a mano). |
| `scripts/extraer_validacion_por_imagen.py` | Saca el Dice por imagen de validación guardado en `best_epoch_*.pt` de cada corrida del benchmark. |
| `scripts/limitacion_area_segformer.py` | Análisis de la limitación por terciles de área, con modo `--simular`. |
| `tests/test_replica_segformer.py`, `tests/test_colab_replica.py`, `tests/test_limitacion_area.py` | 24 pruebas; pasan en el entorno local del proyecto. |

## Parte 1: réplica (ítem 32)

**Qué se replica.** El resultado clave del artículo: SegFormer-B0 en ADE20K, validación de 2.000
imágenes, mIoU en escala única = 37,4 %. Es una réplica de **evaluación**: se evalúa el checkpoint
publicado con el preprocesamiento de la configuración oficial (`AlignedResize` a (2048, 512) con
proporción y múltiplos de 32, imagen completa, sin volteo, normalización de mmseg). Valida el pipeline
de datos y métrica; no reproduce el entrenamiento.

**Vía elegida y por qué.** Checkpoint de Hugging Face `nvidia/segformer-b0-finetuned-ade-512-512`,
revisión `489d5cd8`, `model.safetensors` con SHA-256 verificado en el cuaderno. Es la conversión del
checkpoint de NVlabs que hizo Hugging Face y corre con `transformers`, que Colab ya trae. La vía
original (mmcv-full 1.x + mmseg 0.11 de NVlabs) ata extensiones compiladas a versiones antiguas de
PyTorch y CUDA. El riesgo que se acepta, que la conversión no sea fiel, queda declarado como D3. El
preprocesamiento se implementó a mano con cv2, como mmcv, y no con el procesador por defecto de
Hugging Face (512x512 sin proporción), que se evalúa aparte como desviación D1.

**Criterio a priori.** D = mIoU de la réplica − 37,4. `replicado` si el IC95 de D cae entero en
[−1; +1]; `compatible` si |D| ≤ 1 pero el IC se sale; `no_replicado` si |D| > 1. IC95 por bootstrap de
las 2.000 imágenes (2.000 réplicas, semilla 20261120), recalculando el mIoU acumulado. Si no
coincide, se informan D1-D5 y no se toca el criterio.

**Cómo correr.**

1. Abrir `colab/Replica_SegFormer_ADE20K.ipynb` en Colab con GPU T4.
2. Leer los términos de ADE20K (https://ade20k.csail.mit.edu/terms/) y poner
   `ACEPTO_TERMINOS_ADE20K = True` en la celda 1.
3. Ejecutar las celdas en orden. La celda 4 baja `ADEChallengeData2016.zip` (967.382.037 bytes) de la
   fuente oficial, comprueba el tamaño, calcula el SHA-256, extrae solo validación y exige 2.000 pares.
   La fuente no publica un hash: el de la primera corrida se copia a `SHA256_ADE_FIJADO` para las siguientes.
4. Bajar `replica_segformer.zip` y guardar su contenido en `resultados/replica_segformer/`.

**Tiempo en T4 (estimado, sin medir).** Descarga 3-10 min, extracción 1-2 min, inferencia 3-6 min por
variante (dos variantes), bootstrap menos de 1 min: entre 15 y 30 min en total. La celda 5 mide 50
imágenes y extrapola antes de la corrida completa; anotar ese número en el informe.

**Qué se probó y qué no.** Con un ADE20K simulado de 3 imágenes y un modelo que lee la clase del
color, se probaron el preprocesamiento (tamaños como mmcv), la reducción de etiquetas, la métrica contra
fuerza bruta, la acumulación frente a la media por imagen, el bootstrap, el veredicto y las celdas 1, 3,
6 y 7 del cuaderno. **No se probaron** las celdas 2, 4, 5 y 8 (entorno de Colab, descarga de ADE20K,
checkpoint de Hugging Face y Drive): necesitan red y descargas que no se hicieron en este equipo.

**Paso 2 (réplica parcial de entrenamiento): solo diseño.** En `config/replica_segformer.json`,
sección `paso_2`: codificador `nvidia/mit-b0` (ImageNet-1k), receta del artículo con lote 8 y 8.000
iteraciones (5 % del calendario), semilla 11. Costo estimado sin medir de 1,5 a 2,5 h en T4. No se
implementó ni se ejecutó; no cabe con verificación antes del domingo y no debe presentarse como hecho.

## Parte 2: limitación cuantificada (ítem 33)

**Limitación.** SegFormer-B0 pierde desempeño en rótulos pequeños: el codificador MiT trabaja a 1/4,
1/8, 1/16 y 1/32 y la cabeza MLP predice a 1/4 (en el benchmark, `Segformer` de
segmentation_models_pytorch 0.6.0 con `upsampling=4`), sin conexiones a resolución completa.

**Métrica, subconjunto y magnitud.** Dice por grupo en validación (122 grupos, 218 imágenes), por
terciles del área media del rótulo en el grupo (cortes 0,0195 y 0,0411 de fracción de la imagen;
41, 40 y 41 grupos). Magnitud: Δ = Dice T3 − Dice T1, con IC95 por bootstrap de grupos estratificado
(5.000 réplicas), pareado entre redes. Piloto: U-Net, DeepLabv3+ y SegNeXt-T contra SegFormer-B0;
contraste C = Δ SegFormer − Δ comparador, Holm sobre los tres. Veredicto a priori: `especifica`,
`compartida`, `parcial` o `no_confirmada` (definiciones en el protocolo, δ = 0,02).

**Lo que ya se había visto.** El análisis exploratorio posterior al test
(`resultados/integracion_e2/analisis.json`) muestra en test una caída Q4 − Q1 de 0,051 en
SegFormer-B0, menor que en U-Net (0,082), DeepLabv3+ (0,083) y SegNeXt-T (0,117), con SegFormer-B0 por
debajo en todos los cuartiles. Eso apunta a un déficit de nivel y no de tamaño. El protocolo lo declara
y fija la prueba en validación sin cambiar el enunciado; el veredicto se informa sea cual sea. No es un
preregistro anterior a los experimentos definitivos: el benchmark ya estaba entrenado.

**Datos que faltan.** El Dice por imagen de validación no está en el repositorio: vive en la clave
`validation` de los `best_epoch_*.pt` de las 50 corridas (Drive del benchmark). Los
`*_seed*.json` de test tampoco están aquí, y el único archivo por imagen local
(`resultados/complementos_locales_e2/bordes_seed11_por_imagen.json`) es de test con una semilla; no se usa.

**Cómo correr.**

```
python scripts/extraer_validacion_por_imagen.py --corridas <carpeta del experimento con <red>_seed<semilla>/result.json> --salida resultados/validacion_por_imagen_e2
python scripts/limitacion_area_segformer.py --por-imagen resultados/validacion_por_imagen_e2 --salida resultados/limitacion_segformer_area
```

El extractor necesita torch (Colab o el entorno local), verifica el SHA-256 de cada checkpoint contra
`result.json` y que la media por grupos reproduzca `best_val_group_Dice`. El análisis exige las cinco
semillas de las cuatro redes piloto y las 218 imágenes en cada corrida; si falta algo, se detiene.

**Tiempo.** Extracción: lo que tarde en leer 50 checkpoints desde Drive (sin medir; los de TransUNet
son los más pesados). Análisis: unos 4 s en este equipo con 5.000 réplicas (medido con datos simulados).

**Prueba.** `--simular especifica|compartida|nula` genera Dice por imagen sobre los grupos reales de
validación con un efecto conocido; las pruebas comprueban que el veredicto recupera cada escenario,
que los IC son reproducibles y que faltantes o imágenes repetidas detienen el análisis. Las salidas
simuladas llevan la marca SIMULADO.

## Qué falta

1. Ejecutar el cuaderno en Colab y subir `resultados/replica_segformer/` (ítem 32).
2. Correr el extractor sobre las 50 corridas y luego el análisis (ítem 33).
3. Llevar los dos veredictos al informe con su IC, sin reinterpretarlos.
4. Opcional: obtener el `.pth` oficial de NVlabs para cerrar D3; paso 2 si hay tiempo de GPU después del domingo.
