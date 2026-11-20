# Complemento de entrenamiento del E2

> **Estado: diseñado y probado en CPU con pocas imágenes. NO ejecutado en GPU.** Ningún número de
> calidad de este documento es un resultado del complemento; las horas son **estimadas**.
> Todo lo que produzca es **exploratorio**: el test del benchmark ya fue observado. Aquí el test no
> se carga y todas las elecciones se hacen con validación.

## Qué cubre

Los controles de la guía que exigen volver a entrenar y que el checklist marca F o P:

| bloque | control de la guía | checklist | diseño |
|---|---|---|---|
| sobreajuste | 11.1: verificar la implementación sobreajustando un lote pequeño | 41 | 10 redes, lote fijo de 4 imágenes de train, 500 pasos con la receta de cada red; pasa si el Dice del lote (modo evaluación) ≥ 0,90, umbral fijado antes de correr |
| ablación | 4.6: sobre el mejor modelo, aislar ≥ 3 componentes con las mismas semillas e inferencia sobre cada diferencia | 61 | TransUNet (mejor Dice de validación) sin Transformer, sin conexiones de salto y sin término Dice; semillas 11, 23, 37, 53, 71 |
| permutación | Tabla A: repetir el pipeline completo permutando *y* en train, B ≥ 1000 o B ≥ 200 si es costoso; p = (1 + #{s_b ≥ s_obs})/(B + 1) | 41, 56 | máscaras de train barajadas entre imágenes; FCN con B = 200 (control del pipeline) y TransUNet con B = 19 (control del mejor modelo, declarado bajo el mínimo) |
| hpo | 4.1, 11.1, 11.2: mismo número de intentos y presupuesto por modelo, espacios comparables, solo validación, registro de todo | 37 | búsqueda aleatoria con una secuencia común de intentos para las 10 redes; T = 4 recomendado antes del domingo, T = 8 diseñado |

Configuración completa y justificaciones: `config/complemento_entreno_e2.json`.

### Por qué esas tres ablaciones

El artículo de TransUNet (Chen et al., 2021, arXiv:2102.04306) construye su argumento con justo estas
piezas: compara R50-U-Net, R50-ViT-CUP y TransUNet, estudia el número de conexiones de salto (0, 1 y 3)
y entrena con la mezcla 0,5 entropía cruzada + 0,5 Dice. Las tres variantes se obtienen con perillas del
código oficial fijado en `third_party/e2/transunet`, sin reescribir la red:

- **sin_transformer**: `transformer.num_layers = 12 → 0`. Queda la ResNet-50 híbrida, la proyección de
  parches, el embedding posicional, la LayerNorm final, el decodificador y las tres conexiones de salto.
  Es el contraste CNN frente a TransUNet del artículo. Mide qué aportan los 12 bloques de autoatención.
  Medido en la prueba en CPU: la red pasa de 105.322.001 a 20.267.537 parámetros (sin saltos:
  103.810.577). Quitar el Transformer quita también el 81 % de la capacidad; la diferencia mezcla
  "autoatención" y "tamaño", y así debe leerse.
- **sin_saltos**: `n_skip = 3 → 0`. Codificador completo, decodificador en cascada sin concatenar mapas
  del codificador (el caso n_skip = 0 del artículo).
- **sin_dice**: misma red, pérdida BCE sola (peso 1) en lugar de 0,5 BCE + 0,5 Dice. Al quitar el
  término también cambia la escala del gradiente; la receta (SGD, lr 0,001) no se reajusta porque es la
  fija del benchmark. Se declara como parte de lo que mide la variante.

El runner comprueba en cada corrida que la variante quitó de verdad el componente (número de bloques
Transformer, canales de entrada del decodificador) y se detiene si no.

La referencia es TransUNet completo: por defecto, las cinco corridas originales del benchmark (mismo
paquete, mismo entorno); con `CONTROL=True` se reentrena también completo en esta misma carpeta y el
resumidor lo prefiere. La inferencia por variante es: diferencia media pareada por semilla con IC95 t y
prueba t pareada; Wilcoxon exacto (con 5 pares su p mínimo bilateral es 0,0625, así que sola nunca baja
de 0,05); bootstrap pareado por grupo de validación (2000 remuestreos, 122 grupos) e inversión de signos
por grupo; Holm sobre las tres variantes.

### Permutación: qué se baraja y cuánto cabe

Cada réplica b baraja las 1015 máscaras de train entre imágenes con una permutación fija por
(semilla 20261121, b). Cada imagen conserva su zona válida, así que la parte de la máscara ajena que cae
en el relleno no cuenta. Validación usa sus máscaras verdaderas. El pipeline es el del benchmark
(mismo orden de lotes que la semilla 11, 30 épocas, paciencia 5, mejor época por Dice de validación);
s_b es el mejor Dice de validación de la réplica y s_obs el de la corrida original con semilla 11.

B = 200 para TransUNet no cabe (estimado entre 34 y 170 horas de L4). Por eso dos bloques:

- **pipeline_fcn, B = 200**: el mínimo de la guía para modelos costosos, con la red más barata. Prueba
  el pipeline (datos, partición, pérdida, evaluación), que es el mismo para las 10 redes.
- **transunet, B = 19**: control específico del mejor modelo; p mínimo 1/20 = 0,05. Está por debajo del
  mínimo de la guía y así se reporta.

B se fija con la estimación del piloto y antes de ver cualquier s_b. Si una sesión se corta, se reporta
el B alcanzado y su p mínimo.

Un s_b alto no es necesariamente fuga: con máscaras barajadas la red aún puede aprender la posición y el
tamaño típicos de un rótulo (prior espacial). Por eso el número informativo es s_obs frente a la
distribución de s_b, no s_b frente a cero.

### Búsqueda de hiperparámetros: igual para las 10

- Intento 0 = la receta del benchmark. Intentos 1..T−1 = tasa de la receta × 10^u con u ~ U(−1, 1) y
  weight decay en {0; 0,0001; 0,01}. **El mismo u y el mismo weight decay para las 10 redes** en cada
  intento: espacios de igual amplitud relativa, como pide la sección 11.1.
- Presupuesto por intento: semilla 11, hasta 10 épocas, paciencia 3, validación por grupos.
- Las tareas corren por intento y luego por red: cualquier corte deja un prefijo de intentos completo
  para las 10, con el mismo presupuesto.
- Se guardan todas las configuraciones evaluadas (`hpo_<red>_tNN/result.json`) y el resumidor reporta
  el ganador, su posición, la mejora sobre la receta con el mismo presupuesto y el rango entre intentos.
- **Fase 2, diseñada y no programada**: reentrenar la ganadora de cada red con las cinco semillas y el
  protocolo completo (unas 17 h de L4 estimadas). No cabe antes del domingo.
- **Límites declarados**: una semilla y 10 épocas por intento (el ganador lleva ruido de semilla y
  favorece lo que aprende rápido); el espacio cubre tasa y weight decay, los dos hiperparámetros comunes a
  los 10 optimizadores, y no lote, dropout, profundidad ni ancho, que la guía lista como espacio mínimo.

**Recorte propuesto si no cabe**: T = 4 (receta + 3 intentos) para las 10 antes del domingo. Si ni eso
cabe, no se recorta para unas redes y no para otras: se reporta el prefijo T completo alcanzado.

## Archivos

| archivo | qué es |
|---|---|
| `config/complemento_entreno_e2.json` | diseño, semillas, B, T, espacio, referencias y tiempos del benchmark |
| `scripts/complemento_entreno_e2.py` | runner: `piloto`, `estimar`, `sobreajuste`, `ablacion`, `permutacion`, `hpo` |
| `scripts/resumir_complemento_entreno_e2.py` | tablas e inferencia → `RESUMEN.json` y `RESUMEN.md` |
| `scripts/crear_colab_complemento_entreno.py` | genera el cuaderno y `colab/Complemento_Entrenamiento_E2.verificacion.json` |
| `colab/Complemento_Entrenamiento_E2.ipynb` | cuaderno de Colab (no editar a mano: regenerar) |
| `scripts/prueba_cpu_complemento_entreno.py` | prueba funcional en CPU |
| `resultados/complemento_entreno_e2/prueba_cpu.json` | informe de la prueba en CPU |

El runner reutiliza `experimento_e2_colab.py` (pérdida, `training_batches`, `evaluate_validation`,
semillas, checkpoints atómicos, verificación de fuentes y datos) y `modelos_e2.py`. El cuaderno exige
que esos dos y `dataset_ready.py` tengan los SHA-256 registrados en `config/protocolo_test_e2_cerrado.json`.
Dos diferencias con `train_model`, sin efecto numérico buscado: una caché en RAM de los ejemplos ya
preprocesados (la prueba en CPU comprueba que da los mismos arreglos que `load_example`) y que las
permutaciones y la HPO no guardan pesos al cerrar (solo JSON), para no llenar Drive.

## Cómo se corre

1. Subir `paquete_colab_e2_v2.zip` (el mismo del benchmark) a Colab o a la raíz de Drive y abrir
   `colab/Complemento_Entrenamiento_E2.ipynb` con GPU L4.
2. Celdas 1 a 6: Drive, paquete v2 verificado, archivos nuevos verificados, entorno igual al del
   benchmark (compara el `pip freeze`), `preflight_e2.py` de las 10 redes, lanzador con registro en
   `logs/`.
3. **Piloto** (unos 6 minutos estimados): primera época de tres corridas reales de la ablación y dos
   épocas de la primera permutación de FCN. Escribe `estimacion_computo.json`. Esas épocas se reutilizan.
4. Fijar `B` y `T` con esa estimación y lanzar los bloques en este orden de prioridad, cambiando su
   bandera `EJECUTAR_... = True`.
5. Celda de resumen: `RESUMEN.md` y `RESUMEN.json` en Drive.

Salida: `/content/drive/MyDrive/tesis_imo_e2_complemento_entreno_v1/`, una carpeta por corrida
(`abl_<variante>_s<semilla>`, `perm_<red>_bNNN`, `hpo_<red>_tNN`) con `config.json`, `history.json`,
`best_val_predictions.json` y `result.json`; `sobreajuste/<red>.json`.

**Reanudación**: cada corrida guarda `last.pt` al final de cada época y se reanuda desde ahí con la misma
huella (tarea, código, datos y entorno). Una corrida cerrada se salta. Si cambian el código o el
entorno, la huella no coincide y se exige otra carpeta: no se mezclan corridas. La huella de código
incluye **todos** los `scripts/*.py` del paquete (como en el benchmark): añadir o editar cualquiera a
mitad de un bloque impide reanudar. La prueba en CPU lo comprobó por accidente: un script nuevo creado
durante la prueba hizo que la reanudación se rechazara, como debía.

**Dos sesiones a la vez**: `FRAGMENTO='0/2'` en una y `'1/2'` en la otra reparten las tareas de un
bloque sin pisarse (misma carpeta de Drive). No lanzar dos sesiones con el mismo fragmento.

## Cuánto tarda (horas de L4, ESTIMADAS)

Base: segundos por época de las 50 corridas del benchmark (`auditoria_e2_recibida`), que incluyen leer
imágenes de disco. El piloto corrige con la GPU real y con la caché.

| bloque | corridas | estimado | supuesto |
|---|---|---|---|
| sobreajuste | 10 × 500 pasos | ≈ 0,25 h | un paso con 4 imágenes ≈ 2 pasos del benchmark |
| ablación | 15 | ≈ 10 h; hasta 12,8 h | costo de TransUNet completo (102 s/época) y 24,4 épocas medias; hasta 30 |
| control completo (opcional) | 5 | ≈ 3,5 h | lo mismo |
| permutación FCN, B = 200 | 200 | 6,5 a 33 h sin caché; el piloto lo fija | 19,6 s/época, entre 6 (paciencia) y 30 épocas |
| HPO, T = 4 | 40 | 2,3 a 5,8 h | 522 s por época sumando las 10, entre 4 y 10 épocas |
| HPO, T = 8 | 80 | 4,6 a 11,6 h | lo mismo |
| permutación TransUNet, B = 19 | 19 | 3,2 a 16 h | 102 s/época, entre 6 y 30 épocas |

En las redes compactas casi todo el tiempo del benchmark era leer y redimensionar imágenes (FCN y
TransUNet difieren en 1000 veces los parámetros y solo 5 veces el tiempo). Con la caché la permutación
de FCN debería quedar muy por debajo de la cota; es una hipótesis que el piloto confirma o descarta.

**Lo que cabe antes del domingo con una L4**, en orden: sobreajuste, ablación, permutación FCN
(con el B que dé el piloto), HPO con T = 4 y, si sobra, la permutación de TransUNet. Con dos sesiones en
paralelo, la ablación se reparte con `FRAGMENTO`.

## Qué revisar al terminar

- `preflight.json`: las 10 en `passed`. `environment.freeze.txt` frente al del benchmark: la celda de
  entorno lista los paquetes distintos; si hay alguno, declararlo y preferir `CONTROL=True`.
- Sobreajuste: la pérdida baja y el Dice del lote supera 0,90. Una red que no pasa con su receta es un
  hallazgo que se reporta, no algo que se arregla tocando la receta.
- Ablación: 15 `result.json` con `completed`; épocas y mejor época por corrida (si muchas llegan a 30, la
  paciencia no actuó). Leer el signo y el IC de cada diferencia, no solo el p.
- Permutación: B alcanzado, p y p mínimo posible; distribución de s_b. Si algún s_b se acerca a s_obs,
  revisar fuga antes de interpretar el benchmark.
- HPO: T completo para las 10; si el ganador es la receta (t = 0) en la mayoría, la receta fija no era
  una desventaja; si gana con margen grande en alguna red, esa red quedó en desventaja en el benchmark.
- Ninguna corrida abierta (`last.pt` sin `result.json`) al cerrar.

## Qué queda sin probar

- **Nada se ha ejecutado en GPU.** La prueba en CPU (`resultados/complemento_entreno_e2/prueba_cpu.json`)
  corre una época sobre 6 imágenes de train y 4 de validación en todas las variantes, la permutación de
  FCN y de TransUNet, dos intentos de HPO en tres redes, el sobreajuste con 3 pasos, el piloto con
  reanudación y el resumidor. Solo comprueba que el código corre; no mide calidad ni tiempo.
- VM-UNet no se probó (exige CUDA y mamba-ssm): entra en el sobreajuste y en la HPO solo en GPU.
- La prueba en CPU usó otras versiones (torch 2.14, smp 0.6) y un `ConfigDict` mínimo en lugar de
  `ml-collections`, que no está en el entorno local; en Colab se usan las versiones fijadas.
- La lectura de los mejores checkpoints originales con `weights_only=True` (para el bootstrap por grupos
  de la ablación) no se probó con los archivos reales; si falla, el resumidor lo dice y usa solo la
  inferencia por semilla.
- Que TransUNet completo reentrenado aquí reproduzca los números originales: no se garantiza igualdad
  bit a bit entre sesiones de GPU; por eso existe `CONTROL`.
- Las horas son estimadas; las fija el piloto.
- El paquete v2 se identifica por el SHA-256 que usa el cuaderno del benchmark; no se ha comprobado aquí
  contra el ZIP que está en Drive.
