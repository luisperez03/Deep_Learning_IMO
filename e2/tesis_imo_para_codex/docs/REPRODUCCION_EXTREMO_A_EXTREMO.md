# Reproducción extremo a extremo

Tres niveles, de menor a mayor costo. El primero no pide GPU y se comprobó en un clon limpio el 2026-11-22
(`resultados/reproduccion_completa_e2/LECTURA.md`); los otros dos repiten trabajo de GPU y se describen con los
cuadernos que lo hicieron.

## Qué se necesita y de dónde sale

| Pieza | De dónde | Para qué nivel | Cómo se verifica |
|---|---|---|---|
| Código, configuraciones, manifiestos, resultados derivados y libro | `git clone --branch cierre-e2 https://github.com/luisperez03/Deep_Learning_IMO.git` (proyecto en `e2/tesis_imo_para_codex`) | 1, 2, 3 | commit de git |
| Imágenes y máscaras (carpeta `VOC2012`, HAZMAT-13) | Repositorio de autores [github.com/mrl-amrl/HAZMAT13](https://github.com/mrl-amrl/HAZMAT13), según su README (procedencia y licencia en `docs/DATA_CARD_HAZMAT13.md`) | 1, 2, 3 | SHA-256 de cada imagen y máscara en `resultados/preentrenamiento/{train,val,test}.csv`; los scripts se detienen si no coincide |
| Test cerrado `resultados_test_e2_final/` (50 JSON, resúmenes y 2.180 máscaras de semilla 11) | Drive del proyecto, a pedido. No va al repositorio: las máscaras derivan de imágenes de terceros | 1 | Huella SHA-256 del conjunto `e9d87e51…78cf82` (2.234 archivos); firma del protocolo y `payload_sha256` de cada JSON, que verifica `auditar_test_e2.py` |
| Corridas `tesis_imo_e2_arquitecturas_reales_v2/` (checkpoints e historias de las 50 corridas) | Drive del proyecto, a pedido | 1 (solo dos figuras de curvas), 2 | Hash de cada checkpoint en `config/protocolo_test_e2_cerrado.json` |
| Paquetes de Colab (`paquete_colab_e2_v2.zip`, `paquete_test_e2_cerrado.zip`, `paquete_complemento_gpu_e2.zip`) | Se arman desde el repositorio con `scripts/empaquetar_colab.py` y `scripts/empaquetar_complemento_gpu_e2.py`; el de test es el congelado del Drive | 2, 3 | SHA-256 esperado escrito en la primera celda de cada cuaderno |

La carpeta de datos va como **hermana** del proyecto, porque los manifiestos usan rutas relativas a `e2/`:

```
e2/
├── tesis_imo_para_codex/           <- el proyecto (del clon)
├── VOC2012/                        <- HAZMAT-13
├── resultados_test_e2_final/       <- test cerrado
└── tesis_imo_e2_arquitecturas_reales_v2/   <- opcional en el nivel 1
```

## Entornos

| Uso | Versiones declaradas | Probado el 2026-11-22 |
|---|---|---|
| Análisis e informe (nivel 1) | `requirements-libro.txt` (numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.3, Pillow 11.3.0) más scikit-learn, threadpoolctl y torch CPU, que importan la auditoría y los complementos | Python 3.12.16, numpy 2.5.5, scipy 1.18.1, Pillow 13.0.0, matplotlib 3.11.2, scikit-learn 1.9.1, torch 2.14.1 CPU |
| Libro | `jupyter-book==1.0.4.post1` | jupyter-book 0.15.1 (Python 3.9) |
| Modelos (tests de `test_e2.py`, niveles 2 y 3) | `requirements-e2.txt` (torch 2.5.1, `ml-collections` 1.0.0, `einops` 0.8.1, …) | En Colab, torch 2.5.1+cu121 (`resultados_test_e2_final/environment.freeze.txt`) |

## Nivel 1: el informe desde los artefactos (sin GPU, unos 8 minutos)

Un comando, desde un proyecto cualquiera del repositorio (no hace falta que sea el clon: crea el suyo):

```
python scripts/reproduccion_completa_e2.py --destino CARPETA_NUEVA --voc RUTA/VOC2012
       --test RUTA/resultados_test_e2_final --python PYTHON_DEL_INFORME
       [--python-alt PYTHON_CON_TORCH] [--historias RUTA/tesis_imo_e2_arquitecturas_reales_v2]
       [--jupyter-book RUTA/jupyter-book]
```

Qué hace: clona la rama de GitHub, enlaza los datos como hermanos, regenera las 1.451 máscaras binarias (verificadas
por SHA-256), corre `auditar_test_e2.py`, `graficar_test_e2.py`, `analisis_integracion_e2.py`,
`integrar_entregable2.py` y `completar_analisis_locales_e2.py`, los 12 archivos de tests, recalcula el Dice desde las
máscaras con código propio, compila el libro con `-W` y ejecuta los cuadernos que no piden GPU.

Qué sale:

- `CARPETA_NUEVA/repo/e2/tesis_imo_para_codex/`: el proyecto con todas las tablas, figuras y JSON regenerados, y el
  libro en `libro/_build/html/index.html`.
- `resultados/reproduccion_completa_e2/REPORTE.json` en el proyecto desde el que se lanzó: cada paso con su código de
  salida, cada archivo reescrito comparado con su versión de GitHub (JSON, discrepancia numérica máxima; Markdown,
  líneas distintas; PNG, píxeles distintos), el recálculo independiente, la reinferencia en GPU publicada y la
  ejecución de cada cuaderno.

Qué esperar: discrepancia 0 en tablas, contrastes y recálculo desde máscaras; diferencias pequeñas en los complementos
de EDA y en el rasterizado de las figuras si las versiones de Pillow o matplotlib no son las declaradas. Sin
`--historias`, las dos figuras de curvas de entrenamiento quedan como no reproducidas.

La orden corta del proyecto, `scripts/reproducir_entregable2.py`, regenera en el mismo sitio sin clonar; supone que
`resultados/auditoria_test_e2/comparacion_validacion_test.png` ya existe (lo produce `graficar_test_e2.py`).

## Nivel 2: repetir la inferencia de test (GPU, Colab)

Con los checkpoints en Drive:

1. `colab/Evaluacion_Final_E2_Colab.ipynb` con `paquete_test_e2_cerrado.zip`: vuelve a correr `evaluar_e2_cerrado.py`
   sobre las 50 corridas y escribe un `resultados_test_e2_final/` nuevo (sin optimizador ni ajuste: checkpoints y
   umbral 0,5 fijados). Luego, el nivel 1 con `--test` apuntando a esa carpeta.
2. O `colab/Complemento_GPU_E2.ipynb`, etapa `clean`: es la repetición que ya se hizo; su Dice medio por red difiere
   del test cerrado en 1,3 × 10⁻⁵ como máximo (`resultados/diagnosticos_gpu_e2/TABLAS.md`). Diferencias de ese orden
   son el no determinismo de cuDNN, no un error.

## Nivel 3: reentrenar (GPU, Colab, unas 16 horas registradas)

1. `scripts/preparar_base.py` (particiones y normalización desde train; no debe cambiar los manifiestos: compararlos
   con git).
2. `scripts/empaquetar_colab.py` arma `paquete_colab_e2_v2.zip`; su SHA-256 debe ser el de la primera celda de
   `colab/Benchmark_E2_Colab_v2.ipynb`.
3. `colab/Benchmark_E2_Colab_v2.ipynb`: 10 redes × 5 semillas (11, 23, 37, 53, 71) en una NVIDIA L4; reanuda por
   corrida completada. Las 50 corridas sumaron 16,4 horas de entrenamiento y validación
   (`resultados/auditoria_e2_recibida/auditoria.json`).
4. Nivel 2 y nivel 1 sobre las corridas nuevas.

El reentrenamiento en GPU no es determinista bit a bit: las cifras nuevas serán cercanas, no iguales. La comparación
justa es por red y semilla con la tabla de validación (`resultados/auditoria_e2_recibida/`) y con la dispersión entre
semillas. El test ya se observó: un reentrenamiento que se use para decidir algo con el test es exploratorio.
