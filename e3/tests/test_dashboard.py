"""Pruebas del dashboard: tabla maestra, registro de paginas, layouts, inferencia stub y restricciones de despliegue."""
import base64
import io
import json
import sys
from pathlib import Path

import pytest

E3 = Path(__file__).resolve().parent.parent
if str(E3) not in sys.path:
    sys.path.insert(0, str(E3))

import app as app_mod  # noqa: E402
import dash  # noqa: E402
from plotly.utils import PlotlyJSONEncoder  # noqa: E402

from src import data as D  # noqa: E402
from src import inference, plots  # noqa: E402


def _png(w=300, h=400, ruido=True) -> str:
    from PIL import Image
    import numpy as np
    a = (np.random.default_rng(0).integers(0, 255, (h, w, 3)) if ruido else np.full((h, w, 3), 128)).astype("uint8")
    b = io.BytesIO()
    Image.fromarray(a).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def test_tabla_maestra_es_la_fuente():
    assert D.TABLA_MAESTRA == E3 / "data" / "tabla_maestra.csv" and D.TABLA_MAESTRA.is_file()
    t = D.tabla_maestra()
    assert len(D.redes()) == 10
    assert set(t.tipo) == {"red", "baseline", "original"}
    assert abs(D.fila("transunet_2d").dice_test - 0.8626) < 1e-9
    assert D.fila("original").estado == "pendiente"
    assert (D.redes().calibracion_ece == "pendiente").all()


def test_generadores_existen():
    for n in ("generar_tabla_maestra.py", "generar_tablas_dashboard.py"):
        assert (E3 / "experiments" / n).is_file()


def test_paginas_registradas_y_layouts():
    rutas = {p["path"] for p in dash.page_registry.values()}
    assert {"/", "/eda", "/modelos"} <= rutas
    for p in dash.page_registry.values():
        lay = p["layout"]() if callable(p["layout"]) else p["layout"]
        assert lay is not None
        json.dumps(lay, cls=PlotlyJSONEncoder)  # serializable
    assert app_mod.server is app_mod.app.server


def test_servidor_responde_y_callbacks_registrados():
    c = app_mod.server.test_client()
    for ruta in ("/", "/eda", "/modelos"):
        assert c.get(ruta).status_code == 200
    deps = c.get("/_dash-dependencies")
    assert deps.status_code == 200 and len(deps.get_json()) >= 6
    assert c.get("/").headers["X-Content-Type-Options"] == "nosniff"


def test_modelos_lee_la_tabla_maestra():
    mod = next(m for n, m in sys.modules.items() if n.endswith("pages.modelos") or n == "modelos")
    filas = mod.datos_tabla()
    por_red = {f["Red"]: f for f in filas}
    assert por_red["TransUNet"]["Dice test"] == round(float(D.fila("transunet_2d").dice_test), 4)
    assert por_red["Modelo original"]["Dice test"] == "pendiente"
    assert len(filas) == 12


def test_pareto_coincide_con_e2():
    assert set(D.frontera_pareto(D.redes())) == set(D.redes()[D.redes().pareto_e2].modelo)


def test_graficos_se_construyen():
    plots.registrar_plantilla()
    red, df = D.redes(), D.eda_imagenes()
    for f in (plots.fig_hist_area(df), plots.fig_balance(df), plots.fig_clases(df), plots.fig_intensidad(df), plots.fig_ecdf(df),
              plots.fig_destino(), plots.fig_ks(), plots.fig_pipeline(), plots.fig_dice_ic(red, D.fila("baseline_clasico")),
              plots.fig_pareto(red), plots.fig_cd(red, D.meta_tabla()["nemenyi_cd"]), plots.fig_limitacion(D.limitacion_area()),
              plots.fig_explorador(df, "area_pct", "contraste", "particion"), plots.fig_explorador(df, "clase", None, "ninguno"),
              plots.fig_hist_area(df.iloc[0:0])):
        assert f.layout.title.text
    dc = D.dice_por_caso()
    plots.fig_dice_caso(dc[dc.particion == "test"], ["fcn", "unet"])
    plots.fig_dice_area(D.dice_por_area(), ["fcn"], D.calidad_resumen()["cortes_area_train_frac"])


def test_filtros_eda():
    df = D.eda_imagenes()
    assert set(D.filtrar(df, "val", None).particion) == {"val"}
    assert D.filtrar(df, "todas", [0, 2]).area_pct.max() <= 2


def test_validacion_de_carga():
    assert inference.validar_archivo(_png(), "a.png")[0]
    assert not inference.validar_archivo("data:text/plain;base64,aGk=", "a.txt")[0]
    assert not inference.validar_archivo("basura", "a.png")[0]
    grande = "data:image/png;base64," + base64.b64encode(b"0" * (5 * 1024 * 1024 + 1)).decode()
    ok, msg, _ = inference.validar_archivo(grande, "g.png")
    assert not ok and "5 MB" in msg
    falso = "data:image/png;base64," + base64.b64encode(b"no soy imagen").decode()
    assert not inference.validar_archivo(falso, "f.png")[0]


def test_fuera_de_dominio_y_artefacto_ausente(tmp_path, monkeypatch):
    monkeypatch.setenv('E3_MODELOS_ONNX', str(tmp_path))
    inference._sesion.cache_clear()
    _, _, raw = inference.validar_archivo(_png(300, 400), "a.png")
    assert inference.avisos_fuera_de_dominio(raw) == []
    _, _, pequena = inference.validar_archivo(_png(60, 60), "p.png")
    assert any(a.startswith("R1") for a in inference.avisos_fuera_de_dominio(pequena))
    _, _, plana = inference.validar_archivo(_png(300, 400, ruido=False), "u.png")
    assert any(a.startswith("R3") for a in inference.avisos_fuera_de_dominio(plana))
    r = inference.predecir("fcn", raw)
    assert r["estado"] == "pendiente" and "pendiente" in r["mensaje"]
    x, caja = inference.preprocesar(raw)
    assert x.shape == (1, 3, 256, 256) and caja[2] - caja[0] <= 256
    inference._sesion.cache_clear()


def test_sin_torch_y_cache_de_sesiones():
    assert "torch" not in sys.modules
    assert inference._sesion.cache_info().maxsize == 1
    req = (E3 / "requirements.txt").read_text()
    assert "torch" not in req and "onnxruntime==" in req and "gunicorn==" in req
    r = (E3 / "render.yaml").read_text()
    assert "gunicorn app:server --workers 1 --threads 2 --timeout 120" in r and "3.11.9" in r
