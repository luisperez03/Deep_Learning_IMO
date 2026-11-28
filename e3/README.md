# Entregable 3: UNet-AT y benchmark de segmentación de rótulos HAZMAT

Código, libro y dashboard del Entregable 3 de Deep Learning: segmentación binaria de rótulos de mercancías peligrosas
en HAZMAT-13, benchmark de diez redes y modelo original UNet-AT. Estado: borrador; el modelo original aún no se
entrena con el protocolo completo (ver `MODELO_ORIGINAL_ESTADO.md` y `DASHBOARD_ESTADO.md`).

## Estructura

| Ruta | Contenido |
|---|---|
| `src/` | Modelo original (`original_model.py`), geometría de C1 y C2 (`geometria_at.py`), datos y gráficos del dashboard |
| `training/` | Entrenamiento del original, exportación ONNX y generadores de los cuadernos de Colab |
| `experiments/` | Estándar, protocolo, potencia, tabla maestra, procedencia y revisión de novedad |
| `data/` | Tablas derivadas del benchmark (sin imágenes) y `DATA_STATEMENT.md` |
| `app.py`, `pages/`, `assets/` | Dashboard Dash multipágina |
| `book/` | Informe en Jupyter Book |
| `tests/` | Pruebas del modelo, del dashboard y de consistencia |
| `Dockerfile`, `.github/workflows/ci.yml` | Contenedor e integración continua |
| `MODEL_CARD.md`, `CITATION.cff` | Ficha del modelo y cita |

## Pruebas

    pip install -r requirements.txt pytest
    pytest tests/test_dashboard.py tests/test_consistencia.py

El modelo original necesita torch (CPU), scipy, opencv-python-headless, onnx y onnxruntime:

    pytest tests/test_original_model.py

## Dashboard

    python app.py        (http://127.0.0.1:8050/)

## Libro

    pip install -r book/requirements.txt
    jupyter-book build book/

El resultado queda en `book/_build/html/`. Las celdas leen CSV versionados; no entrenan nada.

## Contenedor

    docker build -t e3 .
    docker run --rm e3

La imagen no se ha construido todavía. Más comandos (libro y dashboard) en el encabezado del `Dockerfile`.

## Entrenamiento

Versiones en `requirements-train.txt`; cuadernos de Colab en `training/colab/`. Detalle y semillas en
`experiments/protocolo.md`.

## Licencia y datos

La licencia del código la decide el equipo y todavía no existe `LICENSE`. La licencia de las imágenes de HAZMAT-13
está sin confirmar: ver `data/DATA_STATEMENT.md`.
