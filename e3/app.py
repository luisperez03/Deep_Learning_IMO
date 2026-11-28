"""Dashboard del Entregable 3: segmentacion de rotulos de mercancias peligrosas (HAZMAT-13).

Ejecucion local:   python app.py            (http://127.0.0.1:8050/)
Produccion:        gunicorn app:server --workers 1 --threads 2 --timeout 120
No importa torch ni codigo de entrenamiento: la inferencia (src/inference.py) usa solo onnxruntime en CPU.
"""
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import dash  # noqa: E402
import dash_bootstrap_components as dbc  # noqa: E402
from dash import Dash, Input, Output, State, html  # noqa: E402

from src import plots, ui  # noqa: E402

plots.registrar_plantilla()

app = Dash(
    __name__, use_pages=True, pages_folder=str(BASE / "pages"), assets_folder=str(BASE / "assets"),
    external_stylesheets=[dbc.themes.FLATLY], title=ui.TITULO, suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
               {"name": "description", "content": "Benchmark de segmentación de rótulos de mercancías peligrosas (HAZMAT-13)."}],
)
server = app.server


@server.after_request
def cabeceras(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    return resp


def _layout():
    return html.Div([ui.navbar(), dbc.Container(dash.page_container, className="pb-3"), ui.footer()])


app.layout = _layout


@app.callback(Output("navbar-collapse", "is_open"), Input("navbar-toggler", "n_clicks"), State("navbar-collapse", "is_open"))
def _alternar_menu(n, abierto):
    return (not abierto) if n else abierto


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8050")), debug=False)
