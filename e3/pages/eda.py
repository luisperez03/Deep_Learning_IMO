"""Pagina EDA: filtros globales (particion y fraccion de area) que actualizan todos los graficos."""
import base64
import io

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, dcc, html

from src import data as D
from src import plots, ui

dash.register_page(__name__, path="/eda", name="EDA", order=1, title="EDA | HAZMAT-13")

N_GALERIA = 6
OPC_PART = [{"label": "Todas las particiones", "value": "todas"}] + [
    {"label": v, "value": k} for k, v in D.PARTICIONES.items()]
OPC_VAR = [{"label": v, "value": k} for k, v in D.VARIABLES_EDA.items()]


def layout():
    q = D.calidad_resumen()
    c = q["constantes_prosa"]
    ru = q["ruido_etiqueta"]
    sh = q["shift"]

    def kpi(valor, etiqueta, nota):
        return dbc.Col(dbc.Card(dbc.CardBody([html.Div(etiqueta, className="kpi-etiqueta"), html.Div(valor, className="kpi-valor"),
                                              html.Small(nota, className="text-muted")])), md=3, sm=6, className="mb-3")

    filtros = dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([html.Label("Partición", htmlFor="f-part", className="fw-semibold"),
                 dcc.Dropdown(id="f-part", options=OPC_PART, value="todas", clearable=False)], md=4),
        dbc.Col([html.Label("Fracción de área del rótulo (% de la imagen)", htmlFor="f-area", className="fw-semibold"),
                 dcc.RangeSlider(id="f-area", min=0, max=30, step=0.5, value=[0, 30],
                                 marks={0: "0", 5: "5", 10: "10", 15: "15", 20: "20", 25: "25", 30: "30"},
                                 tooltip={"placement": "bottom", "always_visible": False})], md=8),
    ])), className="filtros-globales mb-3")

    return html.Div([
        html.H1("Análisis exploratorio de datos", className="mt-3"),
        html.P("Los filtros de esta barra actualizan todos los gráficos de la página, salvo donde se indica.", className="lead"),
        html.Div(filtros, className="sticky-filtros"),
        html.P(id="eda-resumen", className="fw-semibold"),

        ui.seccion("Variable objetivo y balance"),
        dbc.Row([
            dbc.Col([ui.grafico("g-hist-area"), ui.interpretacion("", "t-hist-area")], lg=6),
            dbc.Col([ui.grafico("g-balance"), ui.interpretacion("", "t-balance")], lg=6),
        ]),
        dbc.Row([dbc.Col([ui.grafico("g-clases"), ui.interpretacion("", "t-clases")], lg=12)]),

        ui.seccion("Panel de imágenes", "Intensidades de todas las imágenes filtradas y galería de una muestra aleatoria."),
        dbc.Row([
            dbc.Col([ui.grafico("g-intensidad"), ui.interpretacion("", "t-intensidad")], lg=6),
            dbc.Col([ui.grafico("g-gris"), ui.interpretacion("", "t-gris")], lg=6),
        ]),
        dbc.Button("Nueva muestra aleatoria", id="b-muestra", color="primary", outline=True, n_clicks=0, className="mb-2"),
        dcc.Loading(html.Div(id="galeria")),

        ui.seccion("Calidad de los datos", "Cifras del Entregable 2. No dependen de los filtros."),
        dbc.Row([
            kpi(f"{c['pares_similares_candidatos']['valor']}", "Pares casi duplicados",
                f"dHash y aHash ≤ 6 bits; {c['enlaces_que_cruzan_particiones_actual']['valor']} enlaces cruzan particiones tras agrupar"),
            kpi(f"{c['cuarentena']['valor']}", "Imágenes en cuarentena", "discrepancia semántica o revisión manual"),
            kpi(f"{c['sin_mascara']['valor']}", "Sin máscara utilizable", "excluidas, no tratadas como fondo"),
            kpi(f"{D.fmt(ru['tasa'] * 100, 1)} %", "Ruido de etiqueta",
                f"{ru['con_error']} de {ru['muestra']}; IC95 Wilson {D.fmt(ru['ic95_wilson'][0] * 100, 1)} a "
                f"{D.fmt(ru['ic95_wilson'][1] * 100, 1)} %"),
        ]),
        dbc.Row([
            dbc.Col([ui.grafico("g-destino", plots.fig_destino()),
                     ui.interpretacion("De las 1.685 fotografías, 1.451 son elegibles y se reparten en tres particiones; "
                                       "234 quedan fuera por falta de máscara o por cuarentena.")], lg=6),
            dbc.Col([ui.grafico("g-ks", plots.fig_ks()),
                     ui.interpretacion(f"Ningún rasgo muestra diferencia entre entrenamiento y validación (menor p con Holm = "
                                       f"{D.fmt(sh['ks_min_p_holm'], 2)}); la AUC adversarial es {D.fmt(sh['adversarial_auc_train_val'], 2)} "
                                       f"(p = {D.fmt(sh['adversarial_p'], 2)}). No rechazar no prueba ausencia de shift.")], lg=6),
        ]),
        dbc.Row([dbc.Col([ui.grafico("g-ecdf"), ui.interpretacion("", "t-ecdf")], lg=12)]),

        ui.seccion("Explorador", "Elige las variables; el gráfico respeta los filtros globales."),
        dbc.Row([
            dbc.Col([html.Label("Variable X", htmlFor="x-var"), dcc.Dropdown(id="x-var", options=OPC_VAR, value="area_pct", clearable=False)], md=4),
            dbc.Col([html.Label("Variable Y (opcional)", htmlFor="y-var"), dcc.Dropdown(id="y-var", options=OPC_VAR, value="contraste", clearable=True)], md=4),
            dbc.Col([html.Label("Color", htmlFor="c-var"),
                     dcc.Dropdown(id="c-var", options=[{"label": "Sin color", "value": "ninguno"}] + [
                         {"label": D.VARIABLES_EDA[k], "value": k} for k in ("particion", "origen")],
                         value="particion", clearable=False)], md=4),
        ], className="mb-2"),
        ui.grafico("g-explorador"),
        ui.interpretacion("", "t-explorador"),
    ])


def _q(s):
    return np.percentile(s, [25, 50, 75])


@callback(
    Output("eda-resumen", "children"), Output("g-hist-area", "figure"), Output("t-hist-area", "children"),
    Output("g-balance", "figure"), Output("t-balance", "children"), Output("g-clases", "figure"), Output("t-clases", "children"),
    Output("g-intensidad", "figure"), Output("t-intensidad", "children"),
    Input("f-part", "value"), Input("f-area", "value"),
)
def actualizar_objetivo(part, area):
    df = D.filtrar(D.eda_imagenes(), part, area)
    total = len(D.eda_imagenes())
    resumen = f"{len(df)} de {total} imágenes con los filtros actuales."
    if df.empty:
        v = "Sin imágenes con estos filtros."
        return (resumen, plots.fig_hist_area(df), v, plots.fig_balance(df), v, plots.fig_clases(df), v,
                plots.fig_intensidad(df), v)
    q1, med, q3 = _q(df.area_pct)
    t1 = (f"La mitad de las imágenes tiene el rótulo entre {D.fmt(q1, 2)} % y {D.fmt(q3, 2)} % del área "
          f"(mediana {D.fmt(med, 2)} %): predominan los objetivos pequeños.")
    m = df.area_pct.mean()
    t2 = (f"En promedio solo {D.fmt(m, 2)} % de los píxeles es rótulo; un modelo que dijera siempre «fondo» tendría "
          f"exactitud cercana a {D.fmt(100 - m, 1)} % y Dice cero.")
    cl = df.clase.map(D.CLASES_ES).value_counts()
    t3 = f"La categoría más frecuente es «{cl.index[0]}» ({cl.iloc[0]} imágenes) y la menos frecuente «{cl.index[-1]}» ({cl.iloc[-1]})."
    dif = (df.lum_rotulo - df.lum_anillo).abs().median()
    t4 = (f"La luminancia del rótulo difiere de la de su entorno en una mediana de {D.fmt(dif, 3)} (escala 0 a 1).")
    return (resumen, plots.fig_hist_area(df), t1, plots.fig_balance(df), t2, plots.fig_clases(df), t3,
            plots.fig_intensidad(df), t4)


def _miniatura(ruta, lado=260):
    from PIL import Image
    with Image.open(ruta) as im:
        im = im.convert("RGB")
        g = np.asarray(im.convert("L").resize((128, 128)), dtype=np.uint8)
        im.thumbnail((lado, lado))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    h, _ = np.histogram(g, bins=32, range=(0, 256))
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), h / h.sum() * 100


@callback(
    Output("galeria", "children"), Output("g-gris", "figure"), Output("t-gris", "children"),
    Input("f-part", "value"), Input("f-area", "value"), Input("b-muestra", "n_clicks"),
)
def galeria(part, area, n):
    df = D.filtrar(D.eda_imagenes(), part, area)
    vacia = plots._vacio("Intensidades de las imágenes de la muestra")
    carpeta = D.dir_imagenes()
    if carpeta is None:
        aviso = dbc.Alert(["La galería no está disponible: no se redistribuyen las fotografías porque la licencia de HAZMAT-13 "
                           "para las imágenes está sin confirmar. En local, defina la variable de entorno ",
                           html.Code("E3_DATOS_IMAGENES"), " con la carpeta de imágenes (JPEGImages)."], color="secondary")
        return aviso, vacia, "Sin imágenes disponibles, no hay histograma de intensidades de la muestra."
    if df.empty:
        return dbc.Alert("Sin imágenes con estos filtros.", color="secondary"), vacia, ""
    muestra = df.sample(min(N_GALERIA, len(df)), random_state=int(n or 0))
    cols, hists, ids = [], [], []
    for _, r in muestra.iterrows():
        ruta = carpeta / f"{r.id}.jpg"
        if not ruta.is_file():
            continue
        src, h = _miniatura(ruta)
        alt = (f"Fotografía {r.id} de la partición {D.PARTICIONES[r.particion].lower()}; categoría "
               f"{D.CLASES_ES.get(r.clase, r.clase).lower()}; el rótulo ocupa {D.fmt(r.area_pct, 1)} % de la imagen")
        cols.append(dbc.Col(html.Figure([html.Img(src=src, alt=alt, className="img-fluid rounded"),
                                         html.Figcaption(f"{r.id} · {D.PARTICIONES[r.particion]} · {D.fmt(r.area_pct, 1)} %",
                                                         className="small text-muted")]), md=4, sm=6))
        hists.append(h)
        ids.append(r.id)
    if not cols:
        return dbc.Alert("No se encontraron los archivos en la carpeta configurada.", color="warning"), vacia, ""
    return (dbc.Row(cols), plots.fig_hist_gris(np.array(hists), ids),
            "Cada curva es la distribución de intensidades de una imagen de la muestra; difieren en brillo y en dispersión.")


@callback(Output("g-ecdf", "figure"), Output("t-ecdf", "children"), Input("f-area", "value"))
def shift(area):
    df = D.filtrar(D.eda_imagenes(), "todas", area)
    if df.empty:
        return plots.fig_ecdf(df), "Sin imágenes con este rango de área."
    med = df.groupby("particion").area_pct.median()
    txt = ("Medianas de área: " + "; ".join(f"{D.PARTICIONES[p].lower()} {D.fmt(med[p], 2)} %" for p in D.PARTICIONES if p in med)
           + ". Las curvas casi se superponen: no hay shift visible en el tamaño del rótulo (este gráfico ignora el filtro de partición).")
    return plots.fig_ecdf(df), txt


@callback(Output("g-explorador", "figure"), Output("t-explorador", "children"),
          Input("f-part", "value"), Input("f-area", "value"), Input("x-var", "value"), Input("y-var", "value"),
          Input("c-var", "value"))
def explorador(part, area, x, y, color):
    df = D.filtrar(D.eda_imagenes(), part, area)
    fig = plots.fig_explorador(df, x, y, color)
    if df.empty or not x:
        return fig, "Sin datos con los filtros actuales."
    if y and x in D.NUMERICAS and y in D.NUMERICAS:
        r = float(df[x].corr(df[y], method="spearman"))
        return fig, (f"Correlación de Spearman entre ambas variables: {D.fmt(r, 2)} (n = {len(df)}). Es descriptiva y no implica causalidad.")
    if y and y in D.NUMERICAS:
        return fig, f"Cada caja resume {D.VARIABLES_EDA[y].lower()} por nivel de la variable categórica (n = {len(df)})."
    return fig, f"Distribución de las {len(df)} imágenes filtradas según la variable elegida."
