# Primer bloque del entregable 1

Este flujo usa VOC2012 para explorar segmentación binaria de etiquetas de peligro.
Los originales permanecen en la carpeta hermana `../VOC2012`. No se crean datos de
entrenamiento, splits finales ni modelos en este bloque.

## Ejecución desde esta carpeta

Recomendado: Python 3.11 o posterior compatible con las versiones fijadas.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-inicio.lock.txt
.venv\Scripts\python -m unittest discover -s tests -p test_inicio_voc.py -v
.venv\Scripts\python -m unittest discover -s tests -p test_revision_voc.py -v
.venv\Scripts\python scripts/inicio_voc.py --data-root ".."
.venv\Scripts\python scripts/revisar_voc.py --data-root ".."
.venv\Scripts\python scripts/crear_cuaderno_inicio.py
```

No hace falta instalar Jupyter para regenerar el cuaderno y sus salidas: el último
script ejecuta sus celdas Python y guarda un `.ipynb` estándar. Para interactuar con
él puede abrirse en un entorno Jupyter con el mismo Python. El generador ejecuta
únicamente las celdas definidas en su propio código, no cuadernos de terceros.

La auditoría lee todas las imágenes y máscaras de clase de VOC. Puede tomar unos minutos.
Las rutas se resuelven respecto al script y a `--data-root`; no dependen del usuario
de Windows. Resultados derivados se sobrescriben al repetir, nunca los originales.

## Productos

- `docs/01_problema.md`: definición y alcance provisional.
- `resultados/inicio_voc/inventario.csv`: una fila por identificador, estado, colores,
  fracción de etiqueta, dimensiones, partición original y hashes de imagen/máscara.
- `resultados/inicio_voc/resumen.json`: cobertura, errores, duplicados exactos y
  discrepancias entre conjuntos de clases de XML y máscaras.
- `resultados/inicio_voc/ejecucion.json`: versiones, semilla, configuración y hashes.
- `resultados/inicio_voc/panel_*.jpg`: muestra aleatoria reproducible de 30 pares.
- `resultados/inicio_voc/muestra_visual.csv`: plantilla de revisión humana; regenerarla
  reinicia sus campos. Guardar revisiones completadas en un archivo separado versionado.
- `notebooks/01_eda_voc.ipynb`: EDA inicial con salidas ejecutadas.
- `resultados/inicio_voc/graficos/*.png`: 16 figuras exportables, generadas también
  dentro del cuaderno. Las figuras usan texto sin tildes para evitar glifos ausentes
  en la fuente incluida en Pillow, sin depender de fuentes del equipo.
- `resultados/revision_voc/`: candidatos de similitud, estadísticos de miniaturas,
  metadatos de origen, paneles comparativos y 12 máscaras a resolución original.
- `config/revision_voc.json`: umbrales exploratorios y prioridades de revisión.
- `docs/REVISION_ETAPA_2.md`: interpretación y revisión exploratoria del asistente.

## Interpretación y límites

`ok` significa par técnicamente legible, del mismo tamaño, sin colores desconocidos
y con primer plano. No certifica alineación, cobertura de todas las etiquetas ni
exactitud semántica. Las máscaras vacías se señalan para revisión, no se declaran
inválidas por definición. Colores desconocidos nunca se convierten a etiquetas.

Se comprueban duplicados binarios exactos y píxeles RGB idénticos por SHA256.
La segunda etapa busca candidatos con dHash y aHash de 64 bits (distancias <= 6),
comparando todos los pares. Las coincidencias no confirman identidad ni escena.
La adjudicación de candidatos y los grupos definitivos siguen pendientes.
Los splits originales se describen, no se aprueban.
No hay aún conjunto de prueba congelado: el EDA de todo el corpus es exploratorio.
Antes de modelar se fijarán grupos, splits y protocolo; los parámetros de
preprocesamiento se estimarán solo en entrenamiento.

La bitácora de ejecución contiene hora y plataforma, por lo que puede cambiar entre
ejecuciones. Con mismos archivos, configuración y versiones, el inventario, la
selección visual y los estadísticos deben coincidir. El hash del cuaderno puede
cambiar si se modifica su generador; se registra el código que lo produjo.

Este flujo es independiente del auditor general heredado, que aún necesita ajustes
de rutas, reconocimiento de SegmentationClassRaw y PyYAML. No afirma que sus pruebas
hayan pasado. Licencia, procedencia, publicación del dataset y Jupyter Book final
siguen pendientes; no publicar imágenes ni paneles por ahora.
