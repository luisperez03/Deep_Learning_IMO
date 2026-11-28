"""Acceso a los datos del dashboard.

Toda cifra sale de la tabla maestra (data/tabla_maestra.csv), que genera experiments/generar_tabla_maestra.py a
partir de los resultados del E2. Las demas tablas de data/ las genera experiments/generar_tablas_dashboard.py.
Este modulo solo lee; no calcula metricas.

Variables de entorno (todas opcionales):
  E3_DATOS_IMAGENES  carpeta con las fotografias (VOC2012/JPEGImages). No se incluyen en el repo: la licencia de
                     HAZMAT-13 para las imagenes esta sin confirmar. Sin ella, la galeria muestra un aviso.
  E3_MODELOS_ONNX    carpeta con los artefactos <modelo>.onnx (por defecto e3/models).
  E3_MODEL_CARD_URL  enlace a la model card.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
TABLA_MAESTRA = DATA_DIR / "tabla_maestra.csv"

VERSION = "0.2.0 (comparación local)"
FECHA = "2026-11-27"
MODEL_CARD_URL = os.environ.get(
    "E3_MODEL_CARD_URL", "/assets/ficha_modelo.html")

PARTICIONES = {"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}

VARIABLES_EDA = {
    "area_pct": "Fracción de área del rótulo (% de la imagen)",
    "regiones": "Regiones anotadas por imagen (n)",
    "borde": "Rótulo anotado en el borde (0 = no, 1 = sí)",
    "lum_rotulo": "Luminancia media del rótulo (0 a 1)",
    "lum_anillo": "Luminancia media del entorno inmediato (0 a 1)",
    "contraste": "Contraste rótulo-entorno (0 a 1)",
    "clase": "Categoría de peligro",
    "origen": "Carpeta de anotación",
    "particion": "Partición",
}
CLASES_ES = {
    "corrosive": "Corrosivo", "dangerous": "Peligroso (misceláneo)", "explosive": "Explosivo",
    "flammable": "Gas inflamable", "flammable-solid": "Sólido inflamable",
    "infectious-substance": "Sustancia infecciosa", "inhalation-hazard": "Peligro por inhalación",
    "non-flammable-gas": "Gas no inflamable", "organic-peroxide": "Peróxido orgánico", "oxygen": "Comburente (oxígeno)",
    "poison": "Tóxico", "radioactive": "Radiactivo", "spontaneously-combustible": "Combustión espontánea",
    "mixta": "Varias categorías en la imagen",
}
NUMERICAS = ["area_pct", "regiones", "borde", "lum_rotulo", "lum_anillo", "contraste"]
CATEGORICAS = ["clase", "origen", "particion"]


@lru_cache(maxsize=None)
def tabla_maestra() -> pd.DataFrame:
    return pd.read_csv(TABLA_MAESTRA)


def redes() -> pd.DataFrame:
    t = tabla_maestra()
    return t[t.tipo == "red"].reset_index(drop=True)


def fila(modelo: str) -> pd.Series:
    t = tabla_maestra()
    return t[t.modelo == modelo].iloc[0]


def nombres() -> dict:
    t = tabla_maestra()
    return dict(zip(t.modelo, t.nombre))


@lru_cache(maxsize=None)
def meta_tabla() -> dict:
    return json.loads((DATA_DIR / "tabla_maestra_meta.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def eda_imagenes() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "eda_imagenes.csv")


@lru_cache(maxsize=None)
def dice_por_caso() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "dice_por_caso.csv")


@lru_cache(maxsize=None)
def dice_por_area() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "dice_por_area.csv")


@lru_cache(maxsize=None)
def limitacion_area() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "limitacion_area.csv")


@lru_cache(maxsize=None)
def calidad_shift() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "calidad_shift.csv")


@lru_cache(maxsize=None)
def calidad_resumen() -> dict:
    return json.loads((DATA_DIR / "calidad_resumen.json").read_text(encoding="utf-8"))


def filtrar(df: pd.DataFrame, particion: str | None, area: tuple | list | None) -> pd.DataFrame:
    """Aplica los filtros globales del EDA: particion ('todas' o clave) y rango de fraccion de area en %."""
    out = df
    if particion and particion != "todas":
        out = out[out.particion == particion]
    if area:
        lo, hi = float(area[0]), float(area[1])
        out = out[(out.area_pct >= lo) & (out.area_pct <= hi)]
    return out


def dir_imagenes() -> Path | None:
    """Carpeta con las fotografias, o None si no esta configurada o no existe."""
    p = os.environ.get("E3_DATOS_IMAGENES")
    if not p:
        return None
    d = Path(p)
    return d if d.is_dir() else None


def dir_modelos() -> Path:
    return Path(os.environ.get("E3_MODELOS_ONNX", BASE / "artifacts"))


def tabla_artefactos() -> list[dict]:
    """Tabla de selección/inferencia desde los mismos metadatos que habilitan el ONNX."""
    filas = []
    for ruta in sorted(dir_modelos().glob('*/metadata.json')):
        m = json.loads(ruta.read_text(encoding='utf-8'))
        filas.append({'Modelo': nombres().get(m.get('modelo'), m.get('modelo')),
                      'Arquitectura': m.get('arquitectura', 'E2 histórico'), 'Semilla': m.get('semilla'),
                      'Dice val. media 5 semillas': round(m['media_val_configuracion'], 4) if 'media_val_configuracion' in m else None,
                      'Dice val. artefacto': round(m['val_dice_memoria'], 4) if 'val_dice_memoria' in m else None,
                      'Estado': m.get('estado')})
    return filas


def frontera_pareto(df: pd.DataFrame) -> list[str]:
    """Modelos no dominados en (menos parametros, mas Dice de test). Estimaciones puntuales."""
    d = df.sort_values(["parametros", "dice_test"], ascending=[True, False])
    mejor, out = -1.0, []
    for _, r in d.iterrows():
        if r.dice_test > mejor:
            out.append(r.modelo)
            mejor = r.dice_test
    return out


def fmt(x: float, dec: int = 4) -> str:
    """Numero con coma decimal (convencion del texto de la interfaz)."""
    return f"{x:.{dec}f}".replace(".", ",")


def fmt_miles(n: float) -> str:
    return f"{int(n):,}".replace(",", ".")
