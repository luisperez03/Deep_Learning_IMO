"""Consistencia entre libro, dashboard y tabla maestra.

Garantiza que las cifras del benchmark del capitulo de resultados del libro y las del dashboard salen de la misma
tabla (data/tabla_maestra.csv), sin medias copiadas a mano. No necesita torch ni la red.
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

E3 = Path(__file__).resolve().parents[1]
CSV = E3 / "data" / "tabla_maestra.csv"
META = E3 / "data" / "tabla_maestra_meta.json"
RESULTADOS = E3 / "book" / "07_resultados.md"
COLUMNAS_MEDIAS = ["dice_val", "dice_test", "de_semillas", "ic95_inf", "ic95_sup", "iou_test"]


def _prosa(texto: str) -> str:
    """Texto del capitulo sin los bloques de codigo (code-cell y otros)."""
    return re.sub(r"```.*?```", "", texto, flags=re.S)


def test_tabla_maestra_existe_y_tiene_las_diez_redes():
    t = pd.read_csv(CSV)
    redes = t[t["tipo"] == "red"]
    assert len(redes) == 10
    assert redes["dice_test"].notna().all()


def test_meta_existe_con_hashes_de_entradas():
    meta = json.loads(META.read_text(encoding="utf-8"))
    hashes = meta["entradas_sha256"]
    assert hashes, "la meta no lista las entradas"
    for nombre, h in hashes.items():
        assert re.fullmatch(r"[0-9a-f]{64}", h), f"hash invalido en {nombre}"


def test_meta_hash_del_csv_si_lo_declara():
    meta = json.loads(META.read_text(encoding="utf-8"))
    clave = next((k for k in ("tabla_maestra_sha256", "csv_sha256", "sha256") if k in meta), None)
    if clave is None:
        pytest.skip("tabla_maestra_meta.json no declara aun el hash del CSV (pendiente en generar_tabla_maestra.py)")
    assert meta[clave] == hashlib.sha256(CSV.read_bytes()).hexdigest()


def test_el_libro_lee_la_tabla_maestra():
    texto = RESULTADOS.read_text(encoding="utf-8")
    celdas = re.findall(r"```\{code-cell\}.*?```", texto, flags=re.S)
    assert any("tabla_maestra.csv" in c and "read_csv" in c for c in celdas), \
        "el capitulo de resultados debe leer data/tabla_maestra.csv en una celda"


def test_el_libro_no_copia_a_mano_las_medias_del_benchmark():
    prosa = _prosa(RESULTADOS.read_text(encoding="utf-8"))
    t = pd.read_csv(CSV)
    for col in COLUMNAS_MEDIAS:
        for v in t[col].dropna():
            for forma in (f"{v:.4f}".replace(".", ","), f"{v:.4f}"):
                assert forma not in prosa, f"cifra copiada a mano ({col} = {forma}) en 07_resultados.md"


def test_el_dashboard_lee_la_misma_tabla():
    data_py = (E3 / "src" / "data.py").read_text(encoding="utf-8")
    assert 'DATA_DIR / "tabla_maestra.csv"' in data_py
    modelos = (E3 / "pages" / "modelos.py").read_text(encoding="utf-8")
    assert "tabla_maestra()" in modelos


def test_dashboard_y_libro_ven_los_mismos_valores():
    """La funcion del dashboard devuelve exactamente la tabla que lee el libro."""
    import sys
    if str(E3) not in sys.path:
        sys.path.insert(0, str(E3))
    from src import data as D

    dash = D.tabla_maestra().reset_index(drop=True)
    libro = pd.read_csv(CSV).reset_index(drop=True)
    comunes = [c for c in COLUMNAS_MEDIAS if c in dash.columns]
    assert comunes, "el dashboard no expone columnas de medias"
    pd.testing.assert_frame_equal(dash[comunes], libro[comunes], check_dtype=False)
