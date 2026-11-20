# Complemento GPU del E2: cómo se corre y qué falta

Estado al 2026-11-20: **preparado y probado en CPU, sin ejecutar en GPU.** Ningún número de este complemento existe
todavía. Todo análisis que salga de aquí es exploratorio: el test ya fue observado en E1 y en la evaluación cerrada.

## Archivos

| Archivo | Qué hace |
|---|---|
| `scripts/metricas_diagnostico_e2.py` | Brier, NLL, ECE fijo y adaptativo, temperatura, HD95/ASSD, precisión/recall, perturbaciones, conjuntos de etiquetas empíricos y predicción conforme Mondrian por píxel |
| `scripts/diagnosticos_gpu_e2.py` | Etapas `clean`, `robustness`, `interpret` sobre los 50 checkpoints; firma, hashes, reanudación, tiempos por tarea; `--limit` y `--seeds` para el piloto |
| `scripts/resumir_diagnosticos_gpu_e2.py` | Intervalos por bootstrap de grupos, brecha train–val, curvas de robustez, ensembles, conformal calibrado en validación, fidelidad y estabilidad con IC; verifica también los hashes de los NPZ |
| `scripts/empaquetar_complemento_gpu_e2.py` | Arma `paquete_complemento_gpu_e2.zip` desde el ZIP de test cerrado sin reescribir el protocolo cerrado |
| `scripts/crear_colab_complemento_gpu_e2.py` | Genera `colab/Complemento_GPU_E2.ipynb` con los archivos del complemento embebidos y sus SHA256 |
| `scripts/prueba_cpu_diagnosticos_e2.py` | Prueba de humo en CPU con modelos sin entrenar |
| `scripts/prueba_local_complemento_e2.py` | Prueba local de punta a punta (ZIP cerrado simulado → paquete → extracción → CPU → resumen) |
| `config/complementos_e2_20261120.json` | Protocolo complementario; se le agregó la sección `conformal` antes de cualquier ejecución |

`preparar_test_e2.py` no se ejecutó ni se modificó. Ningún archivo cubierto por los hashes de
`config/protocolo_test_e2_cerrado.json` cambió.

## Qué corre cada celda del cuaderno

| Celda | Contenido | Tiempo |
|---|---|---|
| 1 | Monta Drive, fija rutas (`RUNS`, `OUTPUT = MyDrive/tesis_imo_e2_diagnosticos_v1`), busca los dos ZIP, `nvidia-smi` | segundos |
| 2 | Escribe los archivos embebidos (verifica SHA256), arma el paquete con el empaquetador (verifica el SHA del ZIP cerrado `ce8b6ecc…`, el del ZIP de entrenamiento `954f4fe5…`, el manifiesto cerrado, los 55 archivos del protocolo y cada imagen y máscara de train/val contra el CSV), lo extrae y vuelve a verificar todo. Guarda `paquete_usado.json` en Drive y se niega a seguir si `OUTPUT` tiene resultados de otra versión de los scripts | estimado 3–8 min (≈ 210 MiB; lee el ZIP cerrado y el de entrenamiento desde Drive) |
| 3 | Entorno aislado `/content/imo-e2-env`: Python 3.11, `setuptools==75.8.0`, torch 2.5.1+cu121, `requirements-e2.txt`, transformers 4.46.3, causal-conv1d 1.5.0.post8 y mamba-ssm 2.2.4 (ruedas cp311); `LD_LIBRARY_PATH` con `torch/lib` y `nvidia/*/lib`; comprueba que las versiones son **exactamente** las de `training_environment.packages` y guarda el `freeze` | estimado 5–10 min |
| 4 | **Piloto**: las tres etapas con la semilla 11 de las 10 redes, `--limit 8` (IG `--limit 2`), en `/content/piloto_diagnosticos`; extrapola con los segundos que registra cada tarea y escribe `OUTPUT/PILOTO_TIEMPOS.json` marcado como ESTIMADO | estimado 10–20 min |
| 5 | Etapa `clean`: 50 corridas × (1015 + 218 + 218) imágenes; NPZ float16 de val y test del top 3 | lo dice el piloto |
| 6 | Etapa `robustness`: 450 tareas; la variable `MODELS` permite repartirla entre sesiones | lo dice el piloto |
| 7 | Etapa `interpret`: 3 redes × 5 semillas × 100 casos de validación, IG 16 pasos + 6 pasadas de fidelidad | lo dice el piloto |
| 8 | Resumen (`RESUMEN_DIAGNOSTICOS.json`); corre con etapas incompletas y marca `partial` con la lista de faltantes | estimado 5–15 min (CPU de Colab, lee NPZ de Drive) |
| 9 | Descarga un ZIP con los JSON, logs y `freeze` (sin NPZ ni pesos) | segundos |

Cada lanzamiento de etapa verifica de nuevo los 50 checkpoints y todos los datos: es un costo fijo por lanzamiento
que el piloto mide aparte (`verification_seconds`). Tras un corte de sesión se corren las celdas 1 a 3 y la etapa
interrumpida; las tareas terminadas se validan por firma y hash y se saltan. Los logs quedan en `OUTPUT/logs/` y
los tiempos por lanzamiento en `OUTPUT/tiempos/`.

**La GPU queda atada a la carpeta.** La firma incluye el dispositivo: si una sesión nueva entrega otra GPU, la etapa
se detiene con un mensaje que dice cuál era la registrada. Opciones: esperar la misma GPU o usar otra carpeta de
salida (y entonces correr todo allí, porque latencia y memoria no son comparables entre GPU).

## Cambios a los scripts frente a la versión recibida

- **Conformal por clase (Mondrian), nuevo.** Puntaje `1 − p(clase verdadera)` con la media de las 5 semillas **sin
  temperatura** (la temperatura se ajustó en validación; usarla haría depender el puntaje de los datos de
  calibración). Un umbral para rótulo y otro para fondo, calibrados en validación con hasta 2.000 píxeles por grupo
  y clase, α = 0,10. En test se mide cobertura y tamaño de conjunto por clase, por tercil de área y por grupo
  (p5, mediana, fracción bajo 0,90, IC de la media por grupos). El umbral único marginal se reporta como referencia,
  para mostrar que el fondo se lleva la garantía.
- **Cuantil conforme exacto.** Se usa el estadístico de orden k = ⌈(n+1)(1−α)⌉; `np.quantile(..., 'higher')` en k/n
  tomaba a veces el k+1 (lo detectó una prueba unitaria), y el producto en coma flotante `(n+1)(1−α)` puede pasar
  del entero (20 × 0,9 = 18,000000000000004), por eso se resta 1e-12 antes del techo.
- **Resumidor:** verifica el SHA256 de los NPZ de probabilidades y de atribuciones contra el JSON firmado; IC por
  bootstrap de casos para fidelidad (borrado e inserción contra aleatorio, y fracción de casos en que IG gana) y
  para la estabilidad entre semillas (Spearman medio por caso); caída de Dice por nivel de severidad con IC;
  `np.trapz` con respaldo a `np.trapezoid`.
- **Runner:** dispositivo parametrizable (solo para la prueba en CPU; `main()` sigue exigiendo CUDA), `--limit` y
  `--seeds` para el piloto (`--limit` entra en la firma, así que un piloto no se puede mezclar con la corrida
  definitiva), segundos por tarea y por partición en cada JSON, test cargado una vez por proceso en robustez
  (antes se releía 9 veces por corrida), el candado no se reescribe en cada lanzamiento y el mensaje de firma
  distinta dice qué GPU estaba registrada. Se quitó el sombreado de la máscara `v` dentro de la comprensión de
  diccionario de las métricas calibradas (funcionaba, pero por una regla de alcance poco evidente).

Revisado contra la guía y sin cambio: Brier/NLL/ECE fijo y adaptativo con 15 bins y temperatura en validación
(acotada a [0,05; 10] con aviso si toca el borde); robustez con 3 severidades por perturbación, sin tocar el padding;
HD95/ASSD y precisión/recall por imagen en val y test para las 50 corridas; brecha train–val con el Dice de train
en modo evaluación e IC por bootstrap independiente de grupos; IG del top 3 por validación con 100 casos
estratificados por área del rótulo, elegidos antes de ver predicciones.

## Qué se probó y cómo

- `tests/test_metricas_diagnostico_e2.py`: 9 pruebas, pasan (6 previas y 3 nuevas: cuantil conforme con muestra
  finita, Mondrian contra umbral único con una clase minoritaria difícil, padding excluido de los conjuntos).
- `test_evaluacion_e2.py`, `test_integracion_e2.py` y `test_preparacion.py` pasan.
  `test_complementos_locales_e2.py` y `test_e2.py` no corren en el entorno local disponible (faltan `scikit-learn`
  y `ml-collections`); no se instalaron paquetes.
- `scripts/prueba_local_complemento_e2.py` con las imágenes de `hazmat13_drive/VOC2012`: ver la sección siguiente.

Resultado de `python scripts/prueba_local_complemento_e2.py --voc-root <hazmat13_drive> --work <tmp> --limit 2`
(entorno local `env_e2`, CPU, 2026-11-20):

- ZIP cerrado simulado con las 218 imágenes de test; el empaquetador conservó las 493 entradas y los 55 archivos del
  protocolo cerrado con su hash, agregó 2.473 entradas (1.233 imágenes de train/val desde carpeta, 1.233 máscaras
  derivadas que coincidieron byte a byte con el CSV, y los archivos del complemento); paquete de 207,6 MiB. Tras
  extraer, 0 discrepancias contra `PACKAGE_MANIFEST_COMPLEMENTO.json`.
- Dentro del paquete extraído, con modelos sin entrenar y 2 imágenes por partición: 8 redes × 5 semillas (40
  corridas) en `clean` y `robustness` (24 curvas = 8 redes × 3 perturbaciones) y FCN y U-Net en `interpret`.
  Primera pasada 550 s; la segunda, que debía reanudar, 74 s, sin recalcular nada (clean 16 s, robustez 21 s,
  IG 0,5 s: solo construir modelos y verificar firmas).
- El resumidor produjo curvas, ensembles, conformal e interpretabilidad para FCN y U-Net, y marcó `partial` con
  TransUNet y VM-UNet como faltantes. Las coberturas conformes que salieron (0,94 y 0,77) no dicen nada: modelos
  sin entrenar y 2 imágenes de calibración.
- Al alterar un JSON de robustez (Dice cambiado a mano), el resumidor se detuvo con «Incompatible/corrupto».
- TransUNet no se construyó (falta `ml-collections` en el entorno local); VM-UNet se excluye en CPU por diseño.

## Qué revisar al terminar

1. `RESUMEN_DIAGNOSTICOS.json` con `status: complete` y `missing` vacío.
2. Dice limpio de test por red contra los JSON de la evaluación cerrada: deben quedar muy cerca, no idénticos bit a
   bit (otra GPU, otra versión de cuDNN). Una diferencia mayor que la tercera cifra decimal es un error, no ruido.
3. `temperature_at_bound`: una temperatura en el borde del intervalo invalida la recalibración de esa corrida.
4. Conformal: cobertura de rótulo y de fondo cerca de 0,90 en `mondrian_by_class`; mirar `by_group.p5` y la fracción
   de grupos bajo la nominal antes de afirmar nada. La garantía es aproximada (píxeles no intercambiables, validación
   ya usada para elegir checkpoints).
5. Interpretabilidad: `max_IG_completeness_abs_error` pequeño frente a `score_change_from_baseline`; fidelidad con
   IC que excluya 0 antes de decir que IG supera al orden aleatorio; `undefined_constant_map_pairs`.
6. Recursos: los FLOPs del profiler son un conteo parcial (sin operadores SSM de Mamba ni NMF de SegNeXt), no FLOPs
   certificados; la latencia es de la GPU que tocó.
7. Las categorías de error son geométricas automáticas, no una revisión semántica humana.

## Qué queda sin probar

- Nada se ejecutó en CUDA: ni la instalación del entorno, ni la carga de los 50 checkpoints, ni VM-UNet, ni el
  profiler con CUDA, ni los tiempos reales.
- El empaquetador no se corrió con los ZIP reales (están en el Drive del proyecto): en local se probó con un ZIP
  cerrado simulado de la misma estructura (mismos 55 archivos verificados contra el protocolo, test de
  `hazmat13_drive`) y con train/val tomados de una carpeta y máscaras derivadas, no del ZIP de entrenamiento. La
  rama que lee `paquete_colab_e2_v2.zip` solo se probó en lectura de código. El cuaderno exige el SHA oficial
  `ce8b6ecc…` del ZIP cerrado.
- TransUNet (falta `ml-collections` en local) y VM-UNet (exige CUDA) no se construyeron; las otras 8 redes sí. El
  ensemble, el conformal y la interpretabilidad se probaron de punta a punta con FCN y U-Net, no con TransUNet.
- La prueba en CPU usó 2 imágenes por partición e IG de 2 pasos; con 100 casos y 16 pasos solo se ejercita en GPU.
- El cuaderno se compiló celda a celda (sintaxis) pero no se ejecutó en Colab.
