# Diseño experimental e implementación

## Tres maneras de reconstruir una máscara

Las implementaciones de `scripts/modelos_base.py` comparten bloques de dos convoluciones 3 × 3 con relleno y ReLU, canales 16/32/64, dos reducciones espaciales por máximos y una cabeza 1 × 1. La salida tiene un canal. Con lote unitario se prescindió de BatchNorm. Los pesos se aprendieron desde cero, sin aumentos.

```{include} mapas_redes.inc
```

FCN tiene 72.145 parámetros y centra el presupuesto en extraer contexto. Al ampliar directamente la salida gruesa puede suavizar contornos. SegNet tiene 85.761 parámetros y conserva posiciones de máximos para reconstruir la geometría con *unpooling*. U-Net tiene 118.273 parámetros y transporta también los valores de las características superficiales. Esto permite representar detalles que el *pooling* elimina, a costa de almacenar mapas más grandes. Estas son adaptaciones locales. Por ejemplo, la FCN del artículo explora fusiones laterales y nuestra FCN prescinde de ellas.

Una convolución calcula $h_j=\mathrm{ReLU}(b_j+\sum_i W_{ji}*x_i)$. Para un núcleo de lado $k$, con $C_i$ canales de entrada y $C_o$ de salida, aporta $(k^2 C_i+1)C_o$ parámetros. En un mapa de $H\times W$ su costo es del orden de $HWk^2C_iC_o$ operaciones por imagen. Se suma por capa; la retropropagación conserva el orden y añade gradientes y activaciones intermedias. La tabla de tiempos mide épocas con entrenamiento y validación incluidos.

```{include} fundamentos.inc
```

## Metodología experimental

**De la ecuación al entrenamiento.** En `colab_runner.loss_fn`, `valid` selecciona los píxeles observados antes de sumar. El último renglón asigna el mismo peso a BCE y a la pérdida Dice.

```{literalinclude} ../scripts/colab_runner.py
:language: python
:pyobject: loss_fn
```

**De validación al checkpoint.** En `run`, `score` es el Dice medio por grupo en validación. Una mejora estricta actualiza `best.pt`; en caso de empate se conserva el primer máximo.

```{literalinclude} ../scripts/colab_runner.py
:language: python
:lines: 167-172
:dedent: 8
```

| Elemento | Valor aplicado en las 15 corridas |
|---|---|
| Entrada y lote | 256 × 256, lote 1 |
| Optimizador | Adam, tasa fija 0,001 |
| Pérdida | ½ BCE con logits + ½ pérdida Dice suave, solo píxeles válidos |
| Presupuesto | Máximo 30 épocas, paciencia 5 sin mejora en Dice de validación por grupo |
| Semillas comunes | 11, 23, 37, 53 y 71 |
| Selección | Mejor checkpoint de cada corrida por validación; una evaluación final en test con umbral 0,5 |

La pérdida Dice suave es $1-(2\sum py+\epsilon)/(\sum p+\sum y+\epsilon)$, donde $p$ es la salida sigmoide, $y$ la máscara binaria y $\epsilon=10^{-6}$. El orden de imágenes depende de la semilla y la época. Se fijaron los generadores de Python, NumPy y PyTorch, también en CUDA. El modo determinista registra advertencias cuando una operación carece de implementación determinista. La profesora cerró esta etapa sin ajuste de hiperparámetros. Los planes antiguos con una segunda tasa o más épocas no describen las corridas.

## Hardware y procedencia

El portátil Ryzen 5 7520U ejecutó en CPU las tres redes con semilla 11 y FCN con semilla 23. SegNet con semilla 23 completó la primera época en CPU y las diez restantes en T4. Las otras diez corridas se ejecutaron en T4. Para auditar aquella migración, el historial completo conserva la época CPU y sus segundos. Toda la evaluación final se efectuó en Colab.

El entorno CPU registró Python 3.12.14, PyTorch 2.15.0+cpu, NumPy 2.3.5 y Pillow 12.3.0. El entrenamiento en T4 registró Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0, NumPy 2.4.6 y Pillow 11.3.0. Las versiones del evaluador y los SHA-256 de los 15 checkpoints están en `IMO_test_final.zip → procedencia.json`. Las dependencias del EDA original y del entrenamiento CPU constan en `requirements-inicio.lock.txt` y `requirements-cpu.lock.txt`.

## Reproducir el informe desde los archivos guardados

Los {download}`diagramas, galería y recuentos de capas <../scripts/ilustrar_libro.py>` se regeneran desde la raíz con `.\.venv-cpu\Scripts\python.exe scripts\ilustrar_libro.py`. El script obtiene las dimensiones de salida del código PyTorch con tensores simbólicos de tamaño 1 × 3 × 256 × 256 y coteja las huellas de cada ejemplo de la galería con el manifiesto de train.

El informe se puede auditar con el {download}`código de las tres redes <../scripts/modelos_base.py>`, el {download}`análisis estadístico <../scripts/analisis_entregable.py>`, el {download}`complemento del EDA <../scripts/eda_entregable.py>` y las {download}`dependencias del libro <../requirements-libro.txt>`.

Se necesita Python 3.12, la raíz `tesis_imo_para_codex`, su carpeta hermana `VOC2012`, ambos archivos comprimidos de resultados y los directorios locales de resultados. Para comprobar otra copia de los datos, usar los SHA-256 por archivo del manifiesto. El corpus de los autores se obtiene desde el repositorio citado en el primer capítulo; su equivalencia con esta copia exige verificación.

Desde la raíz del proyecto, en PowerShell, estos comandos crean un entorno independiente, regeneran el complemento de EDA y las tablas y construyen el libro.

```powershell
python -m venv .venv-libro
.\.venv-libro\Scripts\python.exe -m pip install -r requirements-libro.txt
.\.venv-libro\Scripts\python.exe scripts\eda_entregable.py
.\.venv-libro\Scripts\python.exe scripts\analisis_entregable.py
.\.venv-libro\Scripts\jupyter-book.exe build libro -W
```

Abrir `libro/_build/html/index.html`. Para recompilar solo el texto basta el último comando. La configuración desactiva la ejecución de cuadernos. El EDA original se puede examinar desde el capítulo 3 o ejecutar en `notebooks/01_eda_voc.ipynb` en su entorno histórico. El flujo de modelado conservado pasa por `preparar_base.py` (manifiestos y máscaras), `dataset_ready.py` (carga y métricas), `modelos_base.py` y `colab_runner.py` (redes y entrenamiento). `evaluar_test_final.py` produjo el ZIP cerrado. El análisis del libro solo lee sus CSV.

## Reproducir las figuras del cierre

Las tres figuras ya están incluidas en resultados. En el proyecto local, el entorno CPU histórico basta para reconstruirlas desde los tres checkpoints de semilla 11. Se verifican las huellas, se usa `eval()` con gradientes desactivados y se cotejan las cuatro métricas por caso con las filas cerradas de T4. El generador escribe exclusivamente los paneles y su tabla. La auditoría completa lee los artefactos sin ejecutar las redes.

```powershell
.\.venv-libro\Scripts\python.exe scripts\auditar_cierre.py
.\.venv-cpu\Scripts\python.exe scripts\paneles_cierre.py
.\.venv-libro\Scripts\jupyter-book.exe build libro -W
```

El {download}`generador de paneles <../scripts/paneles_cierre.py>` y la {download}`auditoría de integridad <../scripts/auditar_cierre.py>` usan las rutas de la copia local descritas arriba. Las fotografías son para revisión académica local; la redistribución del corpus requiere confirmar la licencia.

### Ruta alternativa para inspección en Colab

El ZIP del test guarda métricas y procedencia. Para ver las máscaras de tres casos, se prepararon `colab/04_IMO_Figuras_Colab.ipynb` y `colab/IMO_figuras_colab.zip`. El paquete contiene solo las tres fotos elegidas, sus anotaciones, los tres mejores checkpoints de semilla 11 y el código de inferencia y preprocesamiento.

1. Abrir el cuaderno en VS Code y seleccionar el kernel de Colab con GPU.
2. Usar **Upload to Colab** para subir únicamente `IMO_figuras_colab.zip` a `/content`. El cuaderno comprueba su SHA-256.
3. Ejecutar la celda y descargar `/content/paneles_IMO.zip`. Incluye la comparación frente a la anotación y las nueve máscaras individuales. No requiere Drive.
4. Este cuaderno histórico produce las máscaras individuales y una lámina simple. El generador local anterior incorpora además los mapas de error y contornos que se muestran en esta edición.

Si se reconstruye el paquete, ejecutar `.\.venv-libro\Scripts\python.exe scripts\preparar_figuras_colab.py` y usar juntos el cuaderno y ZIP recién generados.
