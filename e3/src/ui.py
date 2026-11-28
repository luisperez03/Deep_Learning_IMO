"""Componentes comunes: barra de navegacion, pie, tarjetas y envoltorio de graficos."""
from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from . import data as D

TITULO = "HAZMAT-13: segmentación de rótulos de mercancías peligrosas"


def navbar() -> dbc.Navbar:
    paginas = sorted(dash.page_registry.values(), key=lambda p: p.get("order", 99))
    enlaces = [dbc.NavItem(dbc.NavLink(p["name"], href=p["path"], active="exact")) for p in paginas]
    return dbc.Navbar(
        dbc.Container([
            html.A(html.Span("HAZMAT-13", className="fw-bold text-white"), href="/", className="navbar-brand"),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(dbc.Nav(enlaces, navbar=True, className="ms-auto"), id="navbar-collapse", navbar=True),
        ]),
        color="primary", dark=True, fixed="top", className="shadow-sm", id="navbar",
    )


def footer() -> html.Footer:
    return html.Footer(dbc.Container(dbc.Row([
        dbc.Col([
            html.Strong("Política de privacidad. "),
            "Las imágenes que se cargan para la predicción se procesan en memoria y no se almacenan ni se registran. "
            "La aplicación no usa cookies de seguimiento ni analítica.",
        ], md=8),
        dbc.Col(f"Versión {D.VERSION} · {D.FECHA}", md=4, className="text-md-end"),
    ]), className="py-3"), className="pie mt-5")


def interpretacion(texto, id: str | None = None) -> html.P:
    kw = {"id": id} if id else {}
    return html.P(texto, className="interpretacion", **kw)


def tarjeta(titulo: str, *hijos, id: str | None = None, color: str | None = None) -> dbc.Card:
    kw = {"id": id} if id else {}
    return dbc.Card([dbc.CardHeader(titulo, className="fw-semibold"), dbc.CardBody(list(hijos))],
                    className="mb-3 h-100", color=color, outline=bool(color), **kw)


def grafico(id: str, figura=None, alto: int | None = None) -> dcc.Graph:
    kw = {"figure": figura} if figura is not None else {}
    return dcc.Graph(id=id, config={"displaylogo": False, "responsive": True,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                     style={"height": f"{alto}px"} if alto else {}, **kw)


def pendiente(texto: str) -> dbc.Alert:
    return dbc.Alert([html.Strong("Pendiente. "), texto], color="secondary", className="py-2 mb-2 marcador-pendiente")


def seccion(titulo: str, sub: str | None = None) -> html.Div:
    return html.Div([html.H3(titulo, className="mt-4"), html.P(sub, className="text-muted") if sub else None])
