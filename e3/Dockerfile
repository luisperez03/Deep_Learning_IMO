# Entregable 3: contenedor de reproducibilidad (pruebas livianas, libro y dashboard).
# No se ha construido todavía: Docker no está verificado en la máquina de desarrollo.
#
# Construir:            docker build -t e3 .
# Pruebas (defecto):    docker run --rm e3
# Libro:                docker run --rm -v "$PWD/book/_build:/app/book/_build" e3 jupyter-book build book/
# Dashboard:            docker run --rm -p 8050:8050 e3 gunicorn app:server -b 0.0.0.0:8050 --workers 1 --threads 2
#
# El entrenamiento (torch, GPU) no corre aquí: ver requirements-train.txt y training/colab/.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero, para aprovechar la caché de capas
COPY requirements.txt /tmp/requirements.txt
COPY book/requirements.txt /tmp/book-requirements.txt
RUN pip install -r /tmp/requirements.txt -r /tmp/book-requirements.txt pytest

COPY . /app

# Usuario sin privilegios
RUN useradd --create-home app && chown -R app:app /app
USER app

EXPOSE 8050

# Pruebas que no necesitan torch.
CMD ["python", "-m", "pytest", "tests/test_dashboard.py", "tests/test_consistencia.py", "-q"]
