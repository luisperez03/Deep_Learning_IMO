# Modelo original del Entregable 3: actualización (2026-11-27)

Actualización posterior: la causa de la discrepancia quedó reproducida en `../RESOLUCION_DISCREPANCIA.json`.
Con OpenCV 4.11: 21/21 comparaciones geométricas y pytest 11 aprobadas/1 omitida; sin modificar geometría
ni tolerancia. La reanudación CPU volvió a verificarse con pesos idénticos en ambos brazos.
Usar la secuencia v4, que incluye OpenCV en la identidad y comienza en una carpeta nueva.
El relato siguiente corresponde a la detección inicial con OpenCV 4.10.

Se verificó entrenamiento reducido en CPU (torch 2.5.1+cpu), dos épocas y cuatro imágenes por
partición para los brazos dbnet y dbnet_at. Pausar tras la primera época y reanudar produce los
mismos pesos y métricas que la ejecución continua. `--pause-after-epoch` permite el piloto sin
cambiar la receta ni su huella; repetir una pausa ya alcanzada no avanza otra época.

Pytest real: 10 pruebas aprobadas, una fallida (rombo de 10 px, referencia histórica de DBNet)
y una omitida (falta `entregable_2/codigo/etiquetas.py`). La causa geométrica sigue pendiente.
El cuaderno `training/colab/Piloto_Original_E3_v3.ipynb` incrusta el código actualizado y se
detiene tras dos épocas de cada brazo. Es diagnóstico; no se ejecutó GPU ni la campaña final.
Ver `../AL_DESPERTAR_E3.txt`, `../DIAGNOSTICO_GEOMETRIA.json` y `../VERIFICACION_REANUDACION.json`.

## Registro histórico recibido (2026-11-23)

Generado con: código propio de `e3/src`, `e3/training` y `e3/tests`, datos de `hazmat13_drive/VOC2012` y CSV de
`repo_dl_imo/e2` (solo lectura), entorno `datos/deep_learning/env_e2` (torch 2.14.1 CPU). Sin GPU, sin preentrenamiento.
Por verificar: ver "Pendiente".

## Qué se hizo

- `src/geometria_at.py`: autónomo. Reescribe lo que usaba de `entregable_2/codigo/etiquetas.py` y de
  `entregable_3/codigo/etiquetas_at.py` (C1, C2, refinamiento en píxeles, `VARIANTES`, regla y posproceso de DBNet).
  El inglete ya no usa pyclipper: `inglete_poligono` mueve cada lado d hacia afuera e interseca los consecutivos
  (límite de inglete 2, corte a 2·d de la esquina cuando se pasa); `rellenar_inglete` lo usa en convexos y, en
  no convexos, rellena la unión de piezas convexas (el polígono, un rectángulo por lado y una cuña por vértice convexo).
- `src/original_model.py`: `UNetAT` (mismos anchos 16/32/64 que `SmallUNet`, sin BatchNorm, capas creadas en el
  mismo orden: con la misma semilla global tienen los mismos pesos iniciales que la U-Net de E2), dos cabezas, B = sigmoid(k(P−T)),
  `predict_proba`, `predict_mask`, validación de entrada y `perdida_db`.
- `training/entrenar_original.py`: bucle de `train_model` de E2 con `UNetAT`, objetivos cacheados en RAM, Dice por
  grupo en validación después del posproceso de la variante, mismos checkpoints, `result.json` y reanudación.
- `tests/test_original_model.py`, `tests/conftest.py`, `tests/correr_sin_pytest.py`.

## Pruebas

Comando: `env_e2\python.exe e3\tests\correr_sin_pytest.py` (63 s). Resultado: 10 pasaron, 1 saltada, 0 fallos.

| Prueba | Resultado |
|---|---|
| (i) sobreajuste, 4 imágenes sintéticas 64×64, 250 pasos, Adam 3e-3 | pasó: pérdida 5,28 → 0,17; Dice del posproceso C2 por imagen 0,92 a 1,00 |
| (ii) misma semilla, mismas salidas (también el generador global intacto) | pasó |
| (iii) formas de salida y entradas inválidas (canales, múltiplo de 4, rango, tipo, dtype, variante, cabeza) | pasó |
| (iv) gradiente no nulo y finito en todos los parámetros (cabezas 1x1 y 3x3); derivada de B | pasó |
| (v) rectángulos aspecto 1, 2, 4 con C2 sobre la regla de DBNet, Dice ≥ 0,99 (8 posiciones subpíxel); DBNet en aspecto 4 < 0,90 | pasó |
| (v) rombos de lado 10 a 96 | pasó con umbrales menores que los pedidos (ver riesgos) |
| (v) coincidencia con `resultados/geometria_sintetica.json` (hecho con pyclipper), tolerancia 0,005 | pasó: el inglete propio reproduce las cifras previas |
| (v) inglete propio contra pyclipper, IoU ≥ 0,98 en convexos | pasó: 1,000 en cuadrado, rectángulo, rombo, hexágono y pentágono; 0,989 a 0,996 en triángulo de 70° |
| (v) extra: punta de 30° y forma de L (no convexo) contra pyclipper, IoU ≥ 0,95 | pasó: 0,963 a 0,973 y 0,999 |
| (v) extra: variante `dbnet` idéntica a E2 (`db_objetivos`, `db_post`) | pasó |
| (vi) guardar y cargar `state_dict`, salidas iguales | pasó |
| (vi) exportación ONNX e ida y vuelta (tolerancia 1e-4) | saltada: faltan `onnx` y `onnxruntime` |

Humo de entrenamiento (CPU, 8 imágenes de train y 8 de val, 2 épocas, semilla 11, ambos brazos):

`env_e2\python.exe e3\training\entrenar_original.py --cpu-smoke --allow-local-output --max-images 8 --epochs 2 --seeds 11 --variante dbnet dbnet_at --output C:\Proyectos\estudio\datos\deep_learning\e3_humo`

Salida en `datos/deep_learning/e3_humo`. Corre completo y escribe `result.json` (118.290 parámetros). Tiempo por
época: 2,5 a 2,7 s en `dbnet` y 3,6 a 4,0 s en `dbnet_at` (entrenamiento 2,0 s; el resto es validación con posproceso).
Volver a lanzar el mismo comando devuelve el resultado guardado; con código distinto, se niega por la huella. El Dice de
validación del humo (0,108) es el de una predicción casi vacía tras 2 épocas: prueba que el circuito funciona, no el modelo.

## Decisiones

- **Pesos de la pérdida:** 5·Ls + 10·Lt + Lb, los de `DBNet.perdida` de E2 y los de `diseno.md` 3.1; coinciden. Ls es BCE con OHEM 1:3, Lt es L1
  dentro de la banda y Lb es Dice sobre B.
- **k = 50**, como DBNet y E2.
- **Cabezas:** conv 1x1 por cabeza (por defecto), para que el modelo tenga 17 parámetros más que la U-Net de E2 y
  cualquier ventaja no se deba a capacidad. La opción `cabeza='3x3'` existe (+4.700 parámetros) y no se usa.
- **g0 = 4 px:** paso del mapa más profundo (1/4 de 256), por la arquitectura y no por los datos.
- `forward` en entrenamiento devuelve además `logit_prob`, que la BCE estable necesita.
- **Brazos entrenados:** `dbnet` (etiquetas y posproceso de DBNet) y `dbnet_at` (C1 + C2). Las otras 6 variantes salen por inferencia.
- **Sin aumentos**, como el cargador de E2 (`load_example` no los hace); `diseno.md` 4.2 menciona aumentos solo para su prueba de sobreajuste con ResNet.
- **Arnés de E2:** el preflight de E2 (`preflight.json`, 10 modelos) y `verify_data` (imágenes en `repo_dl_imo/e2/VOC2012`, que no existe en el
  repositorio de solo lectura) no se pueden usar. Se reprodujeron: SHA-256 de imágenes y máscaras contra los CSV, fuga de grupos, `verify_sources`,
  y un preflight propio (forward y backward finitos) por variante. No se saltó ninguna verificación.
  Las imágenes se leen desde `--voc` (por defecto `datos/hazmat13_drive`).

## Qué falta por correr y verificar

- Ejecutar en GPU el entrenamiento completo (2 brazos × 5 semillas, 30 épocas) y medir tiempo por época real: la validación con
  C2 tarda unos 0,3 s por imagen en CPU, o sea alrededor de 1 min por época con las 218 imágenes.
- Ejecutar la prueba ONNX donde estén `onnx` y `onnxruntime` (no hay acceso a PyPI aquí). La exportación no se pudo ni intentar.
- Correr las pruebas con pytest real (solo se corrieron con el corredor propio).
- Precisión completa de las 6 variantes de la ablación en validación con los checkpoints entrenados.
- Mover `training/` de `e3` al repositorio donde vaya a vivir, si cambia la ruta por defecto del arnés (`--e2-scripts`, `--voc`).

## Riesgos

- **Rombos con Dice ≥ 0,99:** la propiedad exacta del teorema no se sostiene en píxeles para todos los tamaños. Con C2 sobre la
  regla de DBNet, los rectángulos de aspecto 1, 2 y 4 dan 1,000. Los rombos de lado 10 y 64 dan ≥ 0,99 con las dos
  reglas, pero lados de 12, 16 (C2), 24 (DBNet-AT), 32 y 48 (C2) y 96 quedan entre 0,89 y 0,99 por el anillo de ±1 px (`diseno.md` 3.3 y 3.5).
  Con C1 incluido (`dbnet_at`), los rectángulos dan 0,94 a 0,96, no 0,99. El test (v) exige lo que sí se sostiene y lo
  documenta; no se afinó nada para que pasara.
- **Inglete en no convexos:** la unión de piezas es el desplazamiento exacto del conjunto y no recorta bucles como Vatti, y
  las esquinas de menos de 60° se cortan a 2·d, con diferencias menores contra pyclipper (IoU 0,96 a 0,97 en una punta de 30°).
- **Datos en RAM:** unos 1,3 GB para train (imágenes y objetivos); cuidar la memoria en Colab si se entrenan varios procesos.
- **El estado de `e3/` es compartido:** la carpeta ya tenía archivos de otra tarea (`app.py`, `experiments/`, `pages/`, etc.). Solo se escribió en
  `src/geometria_at.py`, `src/original_model.py`, `training/entrenar_original.py`, `tests/` y este archivo.
