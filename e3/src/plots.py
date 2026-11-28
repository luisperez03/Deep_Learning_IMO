"""Graficos del dashboard con una plantilla Plotly unica y una paleta de cinco colores.

Accesibilidad: el color nunca es el unico canal. Las series se distinguen ademas por simbolo, patron de linea o
etiqueta directa, y todo grafico lleva titulo, ejes con unidades y hover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from . import data as D

# Paleta (maximo cinco colores). Valores escogidos para distinguirse tambien en escala de grises.
AZUL, NARANJA, VERDE, GRIS, VINO = "#1F4E79", "#D9822B", "#2A9D8F", "#6C757D", "#A4343A"
PALETA = [AZUL, NARANJA, VERDE, VINO, GRIS]
SIMBOLOS = ["circle", "square", "diamond", "triangle-up", "x", "cross", "star", "hexagon", "triangle-down", "pentagon"]
PART_COLOR = {"train": AZUL, "val": NARANJA, "test": VERDE}
PART_SIMBOLO = {"train": "circle", "val": "square", "test": "diamond"}
FAMILIA_SIMBOLO = {
    "Híbrido CNN-Transformer": "star", "Convolucional": "circle", "Espacio de estados (Mamba)": "hexagon",
    "Convolucional codificador-decodificador": "square", "Convolucional con ASPP": "diamond",
    "Convolucional anidada": "triangle-up", "Convolucional de tres ramas": "triangle-down",
    "Convolucional con atención": "cross", "Transformer": "x",
}


def registrar_plantilla() -> None:
    pio.templates["e3"] = go.layout.Template(layout=dict(
        font=dict(family="Lato, 'Segoe UI', Arial, sans-serif", size=13, color="#2C3E50"),
        colorway=PALETA, paper_bgcolor="white", plot_bgcolor="white",
        title=dict(font=dict(size=15), x=0.0, xanchor="left"),
        xaxis=dict(gridcolor="#E5E8EB", zerolinecolor="#C9CED3", linecolor="#9AA3AB", ticks="outside"),
        yaxis=dict(gridcolor="#E5E8EB", zerolinecolor="#C9CED3", linecolor="#9AA3AB", ticks="outside"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=70, b=55), hoverlabel=dict(font_size=12),
    ))
    pio.templates.default = "e3"


def _fig(titulo: str, x: str | None = None, y: str | None = None, alto: int = 360) -> go.Figure:
    f = go.Figure()
    f.update_layout(template="e3", title=titulo, height=alto)
    if x:
        f.update_xaxes(title=x)
    if y:
        f.update_yaxes(title=y)
    return f


def _vacio(titulo: str, alto: int = 320) -> go.Figure:
    f = _fig(titulo, alto=alto)
    f.add_annotation(text="Sin datos con los filtros actuales", showarrow=False, font=dict(size=14, color=GRIS))
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False)
    return f


def _clase_es(s: pd.Series) -> pd.Series:
    return s.map(D.CLASES_ES).fillna(s)


# ----------------------------------------------------------------------------- EDA
def fig_hist_area(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _vacio("Distribución de la fracción de área del rótulo")
    f = _fig("Distribución de la fracción de área del rótulo", "Área del rótulo (% de la imagen, escala logarítmica)",
             "Imágenes (n)")
    bins = np.logspace(np.log10(0.3), np.log10(30), 31)
    for p, nom in D.PARTICIONES.items():
        s = df[df.particion == p].area_pct
        if s.empty:
            continue
        cuentas, _ = np.histogram(s, bins=bins)
        centros = np.sqrt(bins[:-1] * bins[1:])
        f.add_trace(go.Bar(x=centros, y=cuentas, name=nom, marker_color=PART_COLOR[p], opacity=0.75,
                           marker_pattern_shape={"train": "", "val": "/", "test": "x"}[p],
                           hovertemplate=nom + "<br>Área ≈ %{x:.2f} %<br>%{y} imágenes<extra></extra>"))
    f.update_layout(barmode="overlay")
    f.update_xaxes(type="log")
    return f


def fig_balance(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _vacio("Balance de píxeles: rótulo frente a fondo")
    f = _fig("Balance de píxeles: rótulo frente a fondo", "Partición", "Píxeles de la imagen (%, media por imagen)")
    g = df.groupby("particion").area_pct.mean()
    for p in [k for k in D.PARTICIONES if k in g.index]:
        v = float(g[p])
        f.add_trace(go.Bar(x=[D.PARTICIONES[p]], y=[v], name="Rótulo (clase positiva)", marker_color=AZUL,
                           legendgroup="r", showlegend=p == g.index[0],
                           text=[f"{v:.2f} %".replace(".", ",")], textposition="outside",
                           hovertemplate="%{x}<br>Rótulo: %{y:.2f} % de los píxeles<extra></extra>"))
    f.update_layout(yaxis_range=[0, max(10.0, float(g.max()) * 1.4)])
    return f


def fig_clases(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _vacio("Imágenes por categoría de peligro")
    f = _fig("Imágenes por categoría de peligro", "Imágenes (n)", None, alto=430)
    c = df.assign(cat=_clase_es(df.clase)).groupby(["cat", "particion"]).size().unstack(fill_value=0)
    c = c.loc[c.sum(axis=1).sort_values().index]
    for p, nom in D.PARTICIONES.items():
        if p in c:
            f.add_trace(go.Bar(y=c.index, x=c[p], name=nom, orientation="h", marker_color=PART_COLOR[p],
                               marker_pattern_shape={"train": "", "val": "/", "test": "x"}[p],
                               hovertemplate="%{y}<br>" + nom + ": %{x} imágenes<extra></extra>"))
    f.update_layout(barmode="stack")
    f.update_layout(margin=dict(l=190, r=20, t=70, b=55))
    return f


def fig_intensidad(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _vacio("Intensidad del rótulo y de su entorno")
    f = _fig("Intensidad del rótulo y de su entorno", "Partición", "Luminancia media (0 = negro, 1 = blanco)")
    for col, nom, color, sim in (("lum_rotulo", "Rótulo", AZUL, "circle"), ("lum_anillo", "Entorno inmediato", NARANJA, "square")):
        f.add_trace(go.Box(x=df.particion.map(D.PARTICIONES), y=df[col], name=nom, marker_color=color,
                           marker_symbol=sim, boxmean=True,
                           hovertemplate=nom + "<br>%{x}: %{y:.3f}<extra></extra>"))
    f.update_layout(boxmode="group")
    return f


def fig_hist_gris(hist: np.ndarray, ids: list[str]) -> go.Figure:
    """Histograma de intensidades (32 bins de escala de grises) de las imagenes de la galeria."""
    f = _fig("Intensidades de las imágenes de la muestra", "Intensidad en escala de grises (0 a 255)",
             "Píxeles (% de la imagen)", alto=320)
    x = np.arange(32) * 8 + 4
    for i, (h, iid) in enumerate(zip(hist, ids)):
        f.add_trace(go.Scatter(x=x, y=h, name=iid, mode="lines+markers",
                               line=dict(color=PALETA[i % 5], dash=["solid", "dash", "dot", "dashdot"][i % 4]),
                               marker=dict(symbol=SIMBOLOS[i % 10], size=5),
                               hovertemplate=iid + "<br>Intensidad ≈ %{x}<br>%{y:.2f} % de los píxeles<extra></extra>"))
    return f


def fig_ecdf(df: pd.DataFrame, col: str = "area_pct") -> go.Figure:
    f = _fig("Función de distribución empírica por partición (shift)", D.VARIABLES_EDA[col], "Proporción acumulada")
    for p, nom in D.PARTICIONES.items():
        s = np.sort(df[df.particion == p][col].to_numpy())
        if s.size:
            f.add_trace(go.Scatter(x=s, y=np.arange(1, s.size + 1) / s.size, name=f"{nom} (n = {s.size})",
                                   mode="lines", line=dict(color=PART_COLOR[p], shape="hv",
                                                           dash={"train": "solid", "val": "dash", "test": "dot"}[p], width=2.5),
                                   hovertemplate=nom + "<br>%{x:.2f}<br>Acumulado: %{y:.2f}<extra></extra>"))
    if col == "area_pct":
        f.update_xaxes(type="log")
    return f


def fig_destino() -> go.Figure:
    q = D.calidad_resumen()
    c = q["constantes_prosa"]
    pi = q["particiones_imagenes"]
    items = [("Entrenamiento", pi["train"], AZUL, ""), ("Validación", pi["val"], NARANJA, "/"),
             ("Prueba", pi["test"], VERDE, "x"), ("Cuarentena", c["cuarentena"]["valor"], VINO, "."),
             ("Sin máscara", c["sin_mascara"]["valor"], GRIS, "+")]
    f = _fig("Destino de las 1.685 fotografías originales", None, "Imágenes (n)", alto=330)
    for nom, v, col, pat in items:
        f.add_trace(go.Bar(x=[nom], y=[v], name=nom, marker_color=col, marker_pattern_shape=pat, showlegend=False,
                           text=[D.fmt_miles(v)], textposition="outside",
                           hovertemplate=nom + ": %{y} imágenes<extra></extra>"))
    f.update_layout(yaxis_range=[0, 1250])
    return f


def fig_ks() -> go.Figure:
    ks = D.calidad_shift().sort_values("D")
    f = _fig("Shift entre entrenamiento y validación: estadístico KS por rasgo", "Estadístico D de Kolmogorov-Smirnov (0 a 1)",
             None, alto=400)
    f.add_trace(go.Bar(y=ks.rasgo, x=ks.D, orientation="h", marker_color=AZUL, customdata=np.stack([ks.p, ks.p_holm], 1),
                       hovertemplate="%{y}<br>D = %{x:.3f}<br>p = %{customdata[0]:.3f}; p Holm = %{customdata[1]:.3f}<extra></extra>"))
    f.update_layout(margin=dict(l=150, r=20, t=70, b=55), xaxis_range=[0, 1])
    return f


def fig_explorador(df: pd.DataFrame, x: str, y: str | None, color: str | None) -> go.Figure:
    et = D.VARIABLES_EDA
    if df.empty or not x:
        return _vacio("Explorador de variables")
    d = df.copy()
    for c in ("clase",):
        if c in d:
            d[c] = _clase_es(d[c])
    d["particion"] = d.particion.map(D.PARTICIONES)
    num_x = x in D.NUMERICAS
    col = color if color and color != "ninguno" else None
    cats = sorted(d[col].astype(str).unique()) if col else [None]

    def sub(c):
        return d if c is None else d[d[col].astype(str) == c]

    if not y:
        f = _fig(f"Distribución de «{et[x]}»", et[x] if num_x else et[x], "Imágenes (n)")
        for i, c in enumerate(cats):
            s = sub(c)
            if num_x:
                f.add_trace(go.Histogram(x=s[x], name=str(c) if c else "Todas", marker_color=PALETA[i % 5], opacity=0.7,
                                         marker_pattern_shape=["", "/", "x", ".", "+"][i % 5], nbinsx=30,
                                         hovertemplate="%{x}<br>%{y} imágenes<extra></extra>"))
            else:
                v = s[x].astype(str).value_counts()
                f.add_trace(go.Bar(x=v.index, y=v.values, name=str(c) if c else "Todas", marker_color=PALETA[i % 5],
                                   marker_pattern_shape=["", "/", "x", ".", "+"][i % 5],
                                   hovertemplate="%{x}<br>%{y} imágenes<extra></extra>"))
        f.update_layout(barmode="overlay" if num_x else "group")
        return f
    num_y = y in D.NUMERICAS
    if num_x and num_y:
        f = _fig(f"«{et[y]}» según «{et[x]}»", et[x], et[y], alto=420)
        for i, c in enumerate(cats):
            s = sub(c)
            f.add_trace(go.Scattergl(x=s[x], y=s[y], mode="markers", name=str(c) if c else "Imágenes",
                                     marker=dict(color=PALETA[i % 5], symbol=SIMBOLOS[i % 5], size=6, opacity=0.65),
                                     text=s.id, hovertemplate="%{text}<br>x = %{x}<br>y = %{y}<extra></extra>"))
        return f
    if (not num_x) and num_y:
        f = _fig(f"«{et[y]}» por «{et[x]}»", et[x], et[y], alto=420)
        for i, c in enumerate(cats):
            s = sub(c)
            f.add_trace(go.Box(x=s[x].astype(str), y=s[y], name=str(c) if c else "Todas", marker_color=PALETA[i % 5]))
        f.update_layout(boxmode="group")
        return f
    # y categorica: conteo cruzado
    f = _fig(f"«{et[x]}» frente a «{et[y]}»", et[x], "Imágenes (n)", alto=420)
    for i, yy in enumerate(sorted(d[y].astype(str).unique())):
        s = d[d[y].astype(str) == yy]
        v = s[x].astype(str).value_counts() if not num_x else s[x].round(0).astype(str).value_counts()
        f.add_trace(go.Bar(x=v.index, y=v.values, name=yy, marker_color=PALETA[i % 5],
                           marker_pattern_shape=["", "/", "x", ".", "+"][i % 5]))
    f.update_layout(barmode="stack")
    return f


# ----------------------------------------------------------------------------- Contexto
def fig_pipeline() -> go.Figure:
    pasos = [("Fotografías", "1.685 JPG\nHAZMAT-13"), ("Control de calidad", "102 sin máscara\n132 en cuarentena"),
             ("Partición por grupos", "1.015 / 218 / 218\n(sin enlaces entre particiones)"),
             ("Preproceso", "Letterbox 256 px\nnormalización de train"),
             ("Entrenamiento", "10 redes\nsemillas fijas\nBCE + Dice"),
             ("Evaluación", "Dice, IoU\nIC95 por grupos"),
             ("Exportación", "ONNX\n(pendiente)"), ("Dashboard", "Contexto, EDA,\nModelos")]
    f = go.Figure()
    n = len(pasos)
    f.update_layout(template="e3", title="Pipeline del proyecto, de la fotografía al dashboard", height=250,
                    margin=dict(l=10, r=10, t=60, b=10))
    for i, (t, s) in enumerate(pasos):
        x0 = i * 1.0
        pend = "pendiente" in s
        f.add_shape(type="rect", x0=x0 + 0.04, x1=x0 + 0.86, y0=0.15, y1=0.85, line=dict(color=VINO if pend else AZUL, width=2,
                    dash="dash" if pend else "solid"), fillcolor="#F3F6F9", layer="below")
        f.add_annotation(x=x0 + 0.45, y=0.7, text=f"<b>{i + 1}. {t}</b>", showarrow=False, font=dict(size=11), width=95)
        f.add_annotation(x=x0 + 0.45, y=0.38, text=s.replace("\n", "<br>"), showarrow=False, font=dict(size=9.5, color="#4A5560"),
                         width=95)
        if i < n - 1:
            f.add_annotation(x=x0 + 1.0, y=0.5, ax=x0 + 0.87, ay=0.5, xref="x", yref="y", axref="x", ayref="y",
                             showarrow=True, arrowhead=2, arrowwidth=1.6, arrowcolor=GRIS, text="")
    f.update_xaxes(visible=False, range=[-0.02, n - 0.1])
    f.update_yaxes(visible=False, range=[0, 1])
    return f


# ----------------------------------------------------------------------------- Modelos
def fig_dice_ic(red: pd.DataFrame, base: pd.Series | None = None) -> go.Figure:
    d = red.sort_values("dice_test")
    f = _fig("Dice en prueba por red, con IC del 95 % por grupos", "Dice de etiqueta en prueba (0 a 1)", None, alto=440)
    f.add_trace(go.Scatter(
        x=d.dice_test, y=d.nombre, mode="markers", name="Media de cinco semillas",
        error_x=dict(type="data", symmetric=False, array=d.ic95_sup - d.dice_test, arrayminus=d.dice_test - d.ic95_inf,
                     color=GRIS, thickness=1.6),
        marker=dict(color=AZUL, size=10, symbol=[FAMILIA_SIMBOLO.get(x, "circle") for x in d.familia]),
        customdata=np.stack([d.ic95_inf, d.ic95_sup, d.de_semillas, d.familia], 1),
        hovertemplate="%{y}<br>Dice = %{x:.4f}<br>IC95 [%{customdata[0]:.4f}; %{customdata[1]:.4f}]"
                      "<br>DE entre semillas = %{customdata[2]:.4f}<br>%{customdata[3]}<extra></extra>"))
    if base is not None:
        f.add_vline(x=float(base.dice_test), line_dash="dash", line_color=VINO,
                    annotation_text=f"Baseline clásico {D.fmt(base.dice_test, 3)}", annotation_position="bottom right")
    f.update_layout(margin=dict(l=130, r=20, t=70, b=55), xaxis_range=[0.55, 0.92])
    return f


def fig_pareto(red: pd.DataFrame) -> go.Figure:
    f = _fig("Frontera de Pareto: Dice de prueba frente a parámetros", "Parámetros entrenables (escala logarítmica)",
             "Dice de etiqueta en prueba", alto=430)
    fr = D.frontera_pareto(red)
    on = red[red.modelo.isin(fr)].sort_values("parametros")
    f.add_trace(go.Scatter(x=on.parametros, y=on.dice_test, mode="lines", name="Frontera de Pareto",
                           line=dict(color=VINO, dash="dash", width=2, shape="hv"), hoverinfo="skip"))
    for fam, g in red.groupby("familia"):
        f.add_trace(go.Scatter(
            x=g.parametros, y=g.dice_test, mode="markers+text", name=fam, text=g.nombre, textposition="top center",
            textfont=dict(size=10), marker=dict(size=12, symbol=FAMILIA_SIMBOLO.get(fam, "circle"),
                                                color=[VINO if m in fr else AZUL for m in g.modelo],
                                                line=dict(color="white", width=1)),
            customdata=np.stack([g.ic95_inf, g.ic95_sup, [("Sí" if m in fr else "No") for m in g.modelo]], 1),
            hovertemplate="%{text}<br>Parámetros = %{x:,}<br>Dice = %{y:.4f}<br>IC95 [%{customdata[0]:.4f}; %{customdata[1]:.4f}]"
                          "<br>En la frontera: %{customdata[2]}<extra></extra>"))
    f.update_xaxes(type="log")
    f.update_layout(legend=dict(orientation="h", y=-0.28, x=0), margin=dict(l=60, r=20, t=70, b=120))
    return f


def fig_cd(red: pd.DataFrame, cd: float) -> go.Figure:
    d = red.sort_values("rango_grupos", ascending=False).reset_index(drop=True)
    f = _fig("Rangos medios por grupo y diferencia crítica de Nemenyi", "Rango medio (1 = mejor Dice)", None, alto=440)
    f.add_trace(go.Scatter(x=d.rango_grupos, y=d.nombre, mode="markers", name="Rango medio",
                           marker=dict(size=11, color=AZUL, symbol=[FAMILIA_SIMBOLO.get(x, "circle") for x in d.familia]),
                           hovertemplate="%{y}<br>Rango medio = %{x:.2f}<extra></extra>"))
    for _, r in d.iterrows():
        f.add_trace(go.Scatter(x=[r.rango_grupos, r.rango_grupos + cd], y=[r.nombre, r.nombre], mode="lines",
                               line=dict(color=NARANJA, width=3), showlegend=False,
                               hovertemplate=f"{r.nombre}: no se distingue de las redes con rango hasta "
                                             f"{r.rango_grupos + cd:.2f}<extra></extra>"))
    f.add_trace(go.Scatter(x=[], y=[], mode="lines", line=dict(color=NARANJA, width=3),
                           name=f"Intervalo de longitud CD = {D.fmt(cd, 2)}"))
    f.update_layout(margin=dict(l=130, r=20, t=70, b=55))
    return f


def fig_dice_caso(dc: pd.DataFrame, modelos: list[str]) -> go.Figure:
    nom = D.nombres()
    f = _fig("Dice por imagen (media de las cinco semillas)", "Dice de etiqueta por imagen (0 a 1)", "Imágenes (n)")
    for i, m in enumerate(modelos):
        if m in dc:
            f.add_trace(go.Histogram(x=dc[m], name=nom.get(m, m), xbins=dict(start=0, end=1, size=0.04),
                                     marker_color=PALETA[i % 5], opacity=0.65,
                                     marker_pattern_shape=["", "/", "x", ".", "+"][i % 5],
                                     hovertemplate=nom.get(m, m) + "<br>Dice %{x}<br>%{y} imágenes<extra></extra>"))
    f.update_layout(barmode="overlay")
    return f


def fig_dice_area(est: pd.DataFrame, modelos: list[str], cortes: list[float]) -> go.Figure:
    nom = D.nombres()
    etiquetas = [f"Q{i + 1}" for i in range(4)]
    etiquetas[0] += f" (< {D.fmt(cortes[0] * 100, 1)} %)"
    etiquetas[1] += f" ({D.fmt(cortes[0] * 100, 1)} a {D.fmt(cortes[1] * 100, 1)} %)"
    etiquetas[2] += f" ({D.fmt(cortes[1] * 100, 1)} a {D.fmt(cortes[2] * 100, 1)} %)"
    etiquetas[3] += f" (> {D.fmt(cortes[2] * 100, 1)} %)"
    f = _fig("Dice de prueba según el tamaño del rótulo", "Cuartil de área anotada (cortes de entrenamiento)", "Dice de etiqueta", alto=400)
    for i, m in enumerate(modelos):
        g = est[est.modelo == m].sort_values("cuartil")
        if g.empty:
            continue
        f.add_trace(go.Scatter(x=[etiquetas[q - 1] for q in g.cuartil], y=g.Dice, mode="lines+markers", name=nom.get(m, m),
                               line=dict(color=PALETA[i % 5], dash=["solid", "dash", "dot", "dashdot", "longdash"][i % 5]),
                               marker=dict(symbol=SIMBOLOS[i % 5], size=9),
                               error_y=dict(type="data", symmetric=False, array=g.ic95_sup - g.Dice, arrayminus=g.Dice - g.ic95_inf,
                                            thickness=1, color=PALETA[i % 5]),
                               hovertemplate=nom.get(m, m) + "<br>%{x}<br>Dice = %{y:.4f}<extra></extra>"))
    f.update_layout(margin=dict(l=60, r=20, t=70, b=75))
    return f


def fig_limitacion(lim: pd.DataFrame) -> go.Figure:
    nom = D.nombres()
    d = lim.assign(nombre=lim.modelo.map(nom)).sort_values("delta_t3_t1")
    f = _fig("Piloto de área pequeña: pérdida de Dice del tercil menor frente al mayor", "Dice(T3) − Dice(T1), validación", None, alto=400)
    f.add_trace(go.Bar(y=d.nombre, x=d.delta_t3_t1, orientation="h", marker_color=AZUL,
                       error_x=dict(type="data", symmetric=False, array=d.ic95_delta_sup - d.delta_t3_t1,
                                    arrayminus=d.delta_t3_t1 - d.ic95_delta_inf, color=GRIS),
                       hovertemplate="%{y}<br>Δ = %{x:.3f}<extra></extra>"))
    f.update_layout(margin=dict(l=130, r=20, t=70, b=55))
    return f
