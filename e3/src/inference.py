"""Inferencia en vivo (STUB).

Estado: el artefacto ONNX de cada red aun no existe (bloque B del plan: exportacion con prueba de ida y vuelta).
Mientras no exista `<E3_MODELOS_ONNX>/<modelo>.onnx`, `predecir` devuelve el estado "pendiente" y la app sigue
funcionando. Reglas de diseno que se mantienen cuando se complete:

- Solo onnxruntime en CPU. Nunca se importa torch ni el codigo de entrenamiento.
- Carga bajo demanda y como maximo dos sesiones vivas (`functools.lru_cache(maxsize=2)`).
- Privacidad: la imagen se procesa en memoria; no se escribe a disco ni se registra.

Aviso de entrada fuera de dominio (heuristica PROVISIONAL, no calibrada). Reglas, en orden:
  R1  lado menor < 128 px                      resolucion insuficiente para un rotulo pequeno (el modelo trabaja a 256 px).
  R2  relacion de aspecto fuera de [0,3; 3,0]   encuadres extremos (panoramas, tiras) no aparecen en el corpus (tamanos comunes 388x520, 749x1000, 485x1000).
  R3  desviacion tipica de la luminancia < 0,03  imagen casi uniforme: no hay estructura que segmentar.
  R4  luminancia media < 0,08 o > 0,92           imagen casi negra o casi blanca.
Los umbrales son de sentido comun, no se estimaron con el conjunto de entrenamiento. Al exportar el ONNX se deben
sustituir por percentiles de entrenamiento (2,5 % y 97,5 %) de las mismas medidas. Un aviso no bloquea la prediccion.
"""
from __future__ import annotations

import base64
import io
import json
import hashlib
import time
from pathlib import Path
from functools import lru_cache
from typing import Optional

import numpy as np

from . import data as D

MAX_BYTES = 5 * 1024 * 1024
FORMATOS = {"image/png": "PNG", "image/jpeg": "JPEG"}
LADO = 256
# Media y DE de entrenamiento (resultados/preentrenamiento/normalization_train.json del E2, RGB en 0..1).
MEDIA = (0.6040505855357633, 0.5607523535908079, 0.5072384738005108)
DE = (0.2325267080314736, 0.22660245690198325, 0.2396770034095485)

MENSAJE_PENDIENTE = "Artefacto ONNX pendiente"


def decodificar(contents: str) -> tuple[Optional[str], Optional[bytes]]:
    """Separa el prefijo `data:<tipo>;base64,` de un dcc.Upload. Devuelve (tipo, bytes) o (None, None)."""
    try:
        cab, b64 = contents.split(",", 1)
        tipo = cab.split(":", 1)[1].split(";", 1)[0].lower()
        return tipo, base64.b64decode(b64, validate=True)
    except Exception:  # noqa: BLE001 - entrada del usuario: cualquier fallo es "no valida"
        return None, None


def validar_archivo(contents: Optional[str], filename: Optional[str]) -> tuple[bool, str, Optional[bytes]]:
    """Valida formato (PNG o JPG), tamano (<= 5 MB) y que el contenido sea realmente una imagen."""
    if not contents:
        return False, "No se recibió ningún archivo.", None
    tipo, raw = decodificar(contents)
    if raw is None:
        return False, "El archivo no se pudo leer.", None
    if len(raw) > MAX_BYTES:
        return False, f"El archivo pesa {len(raw) / 1048576:.1f} MB; el máximo es 5 MB.", None
    if tipo not in FORMATOS:
        return False, "Formato no admitido: solo PNG o JPG.", None
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != FORMATOS[tipo]:
                return False, "El contenido no coincide con el formato declarado.", None
    except Exception:  # noqa: BLE001
        return False, "El archivo no es una imagen válida.", None
    return True, "Archivo válido.", raw


def avisos_fuera_de_dominio(raw: bytes) -> list[str]:
    """Aplica las reglas R1 a R4 (ver el encabezado). Devuelve la lista de avisos, vacia si no salta ninguna."""
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as im:
        w, h = im.size
        g = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    av = []
    if min(w, h) < 128:
        av.append(f"R1: lado menor de {min(w, h)} px (< 128 px); la resolución puede ser insuficiente.")
    asp = w / h
    if not 0.3 <= asp <= 3.0:
        av.append(f"R2: relación de aspecto {asp:.2f} fuera de [0,30; 3,00].")
    if float(g.std()) < 0.03:
        av.append("R3: imagen casi uniforme (desviación de luminancia < 0,03).")
    m = float(g.mean())
    if m < 0.08 or m > 0.92:
        av.append(f"R4: luminancia media {m:.2f} fuera de [0,08; 0,92].")
    return av


def preprocesar(raw: bytes) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Letterbox a 256x256 con interpolacion bilineal y normalizacion de entrenamiento. Devuelve (1,3,256,256) float32
    y la caja (x0, y0, x1, y1) del contenido dentro del lienzo. El relleno vale 0 tras normalizar.
    Coincidencia con dataset_ready.load_example comprobada en tests de integración."""
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = LADO / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        im = im.resize((nw, nh), Image.BILINEAR)
    a = (np.asarray(im, dtype=np.float32) / 255.0 - np.array(MEDIA, np.float32)) / np.array(DE, np.float32)
    lienzo = np.zeros((LADO, LADO, 3), np.float32)
    x0, y0 = (LADO - nw) // 2, (LADO - nh) // 2
    lienzo[y0:y0 + nh, x0:x0 + nw] = a
    return lienzo.transpose(2, 0, 1)[None], (x0, y0, x0 + nw, y0 + nh)


def metadatos(modelo):
    if modelo not in D.nombres():
        raise ValueError('Modelo desconocido')
    ruta = D.dir_modelos() / modelo / 'metadata.json'
    return json.loads(ruta.read_text(encoding='utf-8')) if ruta.is_file() else {}


@lru_cache(maxsize=1)
def _sesion(modelo: str):
    """Sesion de onnxruntime en CPU, cargada bajo demanda. Devuelve None si falta el artefacto o la biblioteca."""
    meta = metadatos(modelo)
    nombre = meta.get('artefacto')
    if meta.get('estado') != 'ok' or not (meta.get('ida_y_vuelta') or {}).get('aprueba') or not nombre:
        return None
    if Path(nombre).name != nombre:
        raise ValueError('Nombre de artefacto inválido')
    ruta = D.dir_modelos() / modelo / nombre
    if not ruta.is_file():
        return None
    if hashlib.sha256(ruta.read_bytes()).hexdigest() != meta['artefacto_sha256']:
        raise ValueError('El artefacto no coincide con su firma')
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    op = ort.SessionOptions()
    op.intra_op_num_threads = 2
    return ort.InferenceSession(str(ruta), sess_options=op, providers=["CPUExecutionProvider"])


def predecir(modelo: str, raw: bytes) -> dict:
    """Inferencia CPU de artefactos verificados; no guarda la imagen recibida."""
    sesion = _sesion(modelo)
    if sesion is None:
        meta = metadatos(modelo)
        mensaje = meta.get('motivo_sin_artefacto', MENSAJE_PENDIENTE)
        if (meta.get('respaldo_torchscript') or {}).get('aprueba'):
            mensaje += '. Respaldo TorchScript verificado para uso fuera del dashboard'
        return {"estado": "pendiente", "mensaje": mensaje, "modelo": modelo}
    from PIL import Image
    entrada, (x0, y0, x1, y1) = preprocesar(raw)
    inicio = time.perf_counter()
    prob = sesion.run(None, {sesion.get_inputs()[0].name: entrada})[0]
    ms = (time.perf_counter() - inicio) * 1000
    if prob.shape != (1, 1, LADO, LADO) or not np.isfinite(prob).all():
        raise ValueError('Salida del modelo inválida')
    pred = prob[0, 0, y0:y1, x0:x1] >= 0.5
    with Image.open(io.BytesIO(raw)) as imagen:
        imagen = imagen.convert('RGB')
        mascara = Image.fromarray(pred.astype(np.uint8) * 255).resize(imagen.size, Image.Resampling.NEAREST)
        base = np.asarray(imagen).copy()
        seleccion = np.asarray(mascara) > 0
        base[seleccion] = (0.55 * base[seleccion] + 0.45 * np.array([255, 80, 30])).astype(np.uint8)
    def png(im):
        buffer = io.BytesIO()
        im.save(buffer, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    meta = metadatos(modelo)
    return {'estado': 'ok', 'modelo': modelo, 'latencia_ms': ms,
            'mascara': png(mascara), 'superposicion': png(Image.fromarray(base)),
            'mensaje': f"{meta.get('arquitectura', 'E2 histórico')}, semilla {meta['semilla']}; "
                       f"inferencia CPU {ms:.1f} ms. Umbral 0,5; probabilidades sin calibración",
            'arquitectura': meta.get('arquitectura'), 'semilla': meta['semilla']}
