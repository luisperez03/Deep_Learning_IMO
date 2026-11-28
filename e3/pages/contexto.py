"""Pagina Contexto: problema, preguntas e hipotesis, Data Card, pipeline, limitaciones, autores y version."""
import dash
import dash_bootstrap_components as dbc
from dash import html

from src import data as D
from src import plots, ui

dash.register_page(__name__, path="/", name="Contexto", order=0, title="Contexto | HAZMAT-13")


def _tabla(filas, cabecera=None, cls="tabla-simple"):
    head = html.Thead(html.Tr([html.Th(c) for c in cabecera])) if cabecera else None
    cuerpo = html.Tbody([html.Tr([html.Td(c) for c in f]) for f in filas])
    return dbc.Table([head, cuerpo] if head else [cuerpo], bordered=False, hover=True, responsive=True, size="sm", className=cls)


def layout():
    best = D.redes().sort_values("dice_test", ascending=False).iloc[0]
    base = D.fila("baseline_clasico")
    q = D.calidad_resumen()
    pi = q["particiones_imagenes"]

    problema = ui.tarjeta(
        "Problema y relevancia",
        html.P("En una inspección portuaria interesa localizar el rótulo de mercancías peligrosas en la fotografía de un "
               "contenedor antes de leer su símbolo o asignarle una clase. La tarea de este proyecto es la segmentación "
               "binaria: para cada píxel, decidir si pertenece a un rótulo de peligro."),
        html.P(["Relevancia: el accidente del puerto de Tianjin (12 de agosto de 2015), originado en un contenedor con "
                "nitrocelulosa seca sobrecalentada, dejó 165 fallecidos, 8 desaparecidos y 798 heridos según el informe "
                "de investigación del Consejo de Estado de China (febrero de 2016), tal como lo recogió ",
                html.Em("China Daily"), " el 8 de febrero de 2016."], className="mb-2"),
        html.P("La segmentación no sustituye la inspección humana; es un paso previo de apoyo y no está validada como "
               "herramienta de seguridad.", className="text-muted mb-0"),
    )

    preguntas = ui.tarjeta(
        "Preguntas de investigación e hipótesis",
        html.H6("Entregable 2 (benchmark de diez redes)"),
        _tabla([
            ["RQ1", "¿TransUNet mejora el Dice frente a la FCN de referencia?", "H0: Δ = 0; α = 0,05; mejora práctica si supera δ = 0,02"],
            ["RQ2", "¿Hay diferencias entre las diez redes bajo la misma receta?", "H0 global de rangos intercambiables (Friedman); nueve contrastes contra FCN con Holm"],
            ["RQ3", "¿Una red menor queda dentro de ±0,02 de TransUNet?", "TOST de equivalencia por grupos; Pareto como descripción"],
            ["RQ4", "¿El área anotada se asocia con dificultad compartida?", "Descriptiva: Dice por cuartil de área, IC por grupos"],
        ], ["Id.", "Pregunta", "Hipótesis o criterio"]),
        html.Small(f"Respuesta del E2: la mejor media es {best.nombre} ({D.fmt(best.dice_test)}), pero su ventaja sobre FCN "
                   "tiene un IC pareado que incluye cero; solo U-Net resulta equivalente a FCN. Los contrastes posteriores "
                   "al test son exploratorios.", className="text-muted"),
        html.Hr(),
        html.H6("Modelo original (Entregable 3)"),
        ui.pendiente("Pregunta, hipótesis y criterio de decisión del modelo original: se completan al congelar el protocolo "
                     "(bloque C del plan)."),
    )

    ficha = ui.tarjeta(
        "Data Card: HAZMAT-13",
        _tabla([
            ["Fuente", "Laboratorio MRL-AMRL, Universidad Azad de Qazvin (Irán), con la Universidad de Leeds; "
                       "Sharifi, Zibaei y Rezaei (2021), Machine Learning with Applications, 100104"],
            ["Licencia", "El repositorio es MIT; la licencia de las imágenes y anotaciones no está declarada. "
                         "Por eso este sitio no redistribuye imágenes. Consulta a los autores: pendiente de respuesta."],
            ["Tamaño (n)", f"1.685 fotografías; 1.451 pares elegibles: {D.fmt_miles(pi['train'])} entrenamiento, "
                           f"{pi['val']} validación, {pi['test']} prueba (122 grupos de similitud en validación y prueba)"],
            ["Variable objetivo", "Máscara binaria: unión de las máscaras de las 13 categorías de peligro (1 = rótulo, 0 = fondo); "
                                  f"media de {D.fmt(D.eda_imagenes().area_pct.mean(), 2)} % de píxeles positivos por imagen"],
            ["Tarea", "Segmentación semántica binaria a 256 × 256 px (letterbox)"],
            ["Métrica primaria", "Dice de etiqueta, promediado por imagen, por grupo y por semilla"],
        ]),
    )

    limites = ui.tarjeta(
        "Limitaciones, ética y alcance",
        html.Ul([
            html.Li("El conjunto de prueba ya se observó en el Entregable 1: no es un holdout nunca visto."),
            html.Li("Una sola partición y cinco semillas por red; los IC remuestrean grupos de similitud, que no equivalen "
                    "a escenas independientes."),
            html.Li("Sin imágenes negativas en prueba: no se estima la tasa de falsas alarmas en escenas sin rótulos."),
            html.Li(f"Ruido de etiqueta: {q['ruido_etiqueta']['con_error']} de {q['ruido_etiqueta']['muestra']} imágenes "
                    "revisadas tenían un rótulo sin máscara."),
            html.Li("Fotografías de rótulos, no de personas; no se hizo una revisión sistemática de rostros o matrículas."),
            html.Li("La licencia de las imágenes está sin confirmar: la galería del EDA solo funciona si la carpeta de "
                    "imágenes se configura localmente."),
            html.Li("Las redes se entrenaron desde cero con una receta común; las diferencias no se atribuyen solo a la arquitectura."),
        ], className="mb-0"),
    )

    return html.Div([
        html.H1("Segmentación de rótulos de mercancías peligrosas", className="mt-3"),
        html.P("Benchmark de diez redes sobre HAZMAT-13 y modelo original. Entregable 3, Deep Learning.", className="lead"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Mejor Dice de prueba", className="kpi-etiqueta"),
                                           html.Div(D.fmt(best.dice_test), className="kpi-valor"),
                                           html.Small(best.nombre)])), md=3, sm=6, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Baseline clásico", className="kpi-etiqueta"),
                                           html.Div(D.fmt(base.dice_test), className="kpi-valor"),
                                           html.Small("referente, fuera de la familia")])), md=3, sm=6, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Redes × semillas", className="kpi-etiqueta"),
                                           html.Div(f"{len(D.redes())} × 5", className="kpi-valor"),
                                           html.Small("50 corridas en el E2")])), md=3, sm=6, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Modelo original", className="kpi-etiqueta"),
                                           html.Div("pendiente", className="kpi-valor"),
                                           html.Small("sin cifras hasta congelar el protocolo")])), md=3, sm=6, className="mb-3"),
        ]),
        dbc.Row([dbc.Col(problema, lg=6), dbc.Col(ficha, lg=6)]),
        dbc.Row([dbc.Col(preguntas, lg=12)]),
        ui.seccion("Pipeline", "De la fotografía a esta aplicación. El recuadro punteado indica un paso pendiente."),
        ui.grafico("g-pipeline", plots.fig_pipeline()),
        ui.interpretacion("Los pasos 1 a 6 corresponden a resultados del Entregable 2; la exportación a ONNX habilita la "
                          "predicción en vivo de la página Modelos."),
        dbc.Row([dbc.Col(limites, lg=8),
                 dbc.Col(ui.tarjeta("Autores, versión y fecha",
                                    html.P(f"Versión {D.VERSION}", className="mb-1"),
                                    html.P(f"Fecha: {D.FECHA}", className="mb-1"),
                                    html.P("Autores: por completar antes de la entrega.", className="mb-1"),
                                    html.P("Trazabilidad: las cifras provienen de data/tabla_maestra.csv, generada por "
                                           "experiments/generar_tabla_maestra.py a partir de los resultados del E2.",
                                           className="text-muted small mb-0")), lg=4)]),
    ])

