# Proyecto de grado — localización y clasificación de etiquetas IMO

> **Preparación previa a modelos:** consultar `PREENTRENAMIENTO_REPRODUCIBLE.md`
> y `docs/PREENTRENAMIENTO.md`. Se generaron manifiestos agrupados y máscaras
> binarias derivadas. Los modelos no se ejecutan; hardware y protocolo requieren
> validación con el investigador.

> **Actualización 2026-11-06:** el primer bloque de VOC ya está implementado y
> ejecutado. Empezar por [INICIO_REPRODUCIBLE.md](INICIO_REPRODUCIBLE.md) y
> `notebooks/01_eda_voc.ipynb`. Las instrucciones de instalación que siguen pertenecen
> al paquete original; no mover los datasets para seguirlas. El flujo focalizado
> admite la estructura actual de carpetas hermanas.

## Instalación en tu carpeta actual de Windows
1. Abre `Descargas/DEEP LEARNING` (donde ya están tus tres carpetas de imágenes).
2. **Extrae el contenido del ZIP dentro de `DEEP LEARNING`**, no dentro de una cuarta carpeta. Los archivos `AGENTS.md`, `README.md` y `ejecutar_auditoria.bat` deben quedar al mismo nivel de los tres datasets.
3. Abre esa carpeta raíz completa en Codex (no solo una de las tres carpetas).
4. Pega en Codex el contenido de `PROMPT_PARA_CODEX.txt`.

Árbol esperado:
```
DEEP LEARNING/
├── Hazmat Pacards.v5i.coco/    # tus imágenes originales
├── VOC2012/                   # tus imágenes originales
├── SeaFront_v1_0_0/           # tus imágenes originales
├── AGENTS.md
├── CONTEXTO_ACADEMICO.md
├── METODOLOGIA.md
├── DECISIONES.md
├── PROMPT_PARA_CODEX.txt
├── README.md
├── requirements.txt
├── ejecutar_auditoria.bat
├── config/datasets.json
├── scripts/auditar_datasets.py
├── scripts/crear_muestrarios.py
└── tests/test_auditoria.py
```

## Ejecutar auditoría manualmente (alternativa a Codex)
En Windows, con Python 3.10 o posterior instalado: doble clic en `ejecutar_auditoria.bat`. La primera ejecución crea `.venv` e instala Pillow y PyYAML (necesita conexión para instalar dependencias). Luego ejecuta auditoría **sin alterar originales**.

Alternativa por terminal desde `DEEP LEARNING`:
```
py -3 -m pip install -r requirements.txt
py -3 scripts/auditar_datasets.py
py -3 scripts/crear_muestrarios.py
```

Salida (creada automáticamente):
- `resultados/auditoria/INFORME_AUDITORIA.md`: interpretación general, incidencias y preguntas.
- `resultados/auditoria/resumen.json`: metadatos estructurados (para Codex).
- `resultados/auditoria/clases.csv`: clases e instancias observadas.
- `resultados/auditoria/muestra_imagenes.csv`: tamaños, modos y estado de apertura de muestras.
- `resultados/auditoria/incidencias.csv`: anomalías relevantes.
- `resultados/auditoria/muestrarios/*.jpg`: paneles visuales con ejemplos anotados para revisar qué significan realmente las clases.

## Comandos útiles
```
# Revisar más imágenes por dataset (por defecto 250)
py -3 scripts/auditar_datasets.py --muestra-imagenes 1000

# Revisar todas las imágenes con Pillow (más lento)
py -3 scripts/auditar_datasets.py --todas-imagenes

# Detectar archivos de imagen idénticos por hash SHA256 (puede tardar)
py -3 scripts/auditar_datasets.py --hash-duplicados

# Ejecutar pruebas automáticas
py -3 -m unittest discover -s tests -v
```

## Alcance/limitaciones del auditor
- Reconoce COCO, YOLO, Pascal VOC y máscaras por directorios reconocibles. Algunos formatos no estándar exigirán adaptar el código.
- El primer reporte es **exploratorio**, no la validación visual ni semántica definitiva.
- Si un dataset carece de archivos de anotación reconocidos, el sistema no inventa etiquetas.
- La integridad de imagen se examina en una muestra por defecto, no en el 100%; el informe lo indica.
- La existencia de un directorio de máscaras no garantiza que sean máscaras IMO adecuadas; verificar sus valores, clases y correspondencia.
- La herramienta no convierte datos ni entrena modelos.
