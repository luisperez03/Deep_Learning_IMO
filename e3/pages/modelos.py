"""Resultados históricos, comparación visual de test e inferencia local verificada."""
import dash
import json
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, callback, dash_table, dcc, html
from dash.dash_table.Format import Format, Group, Scheme

from src import data as D
from src import inference, plots, ui, comparador_test

dash.register_page(__name__, path="/modelos", name="Modelos", order=2, title="Modelos | HAZMAT-13")


def _num(dec, miles=False):
    f = Format(precision=dec, scheme=Scheme.fixed, decimal_delimiter=",")
    return f.group(Group.yes).groups(3).group_delimiter(".") if miles else f


def datos_tabla() -> list[dict]:
    """Filas de la DataTable, leidas de la tabla maestra. Los valores ausentes se rotulan, no se dejan en blanco."""
    filas = []
    for _, r in D.tabla_maestra().iterrows():
        orig = r.tipo == "original"
        base = r.tipo == "baseline"

        def v(x, dec=None, ausente="n/d"):
            if orig:
                return "pendiente"
            if x != x:  # NaN
                return ausente
            return round(float(x), dec) if dec is not None else x

        fila = {
            "Red": r.nombre, "Familia": r.familia,
            "Dice test": v(r.dice_test, 4), "DE semillas": v(r.de_semillas, 4),
            "IC 95 % (grupos)": "pendiente" if orig else f"{D.fmt(r.ic95_inf)} a {D.fmt(r.ic95_sup)}",
            "Dice val.": v(r.dice_val, 4), "IoU test": v(r.iou_test, 4, "n/d (no medido)"),
            "Parámetros": v(r.parametros, 0, "n/d"), "GFLOPs": v(r.gflops, 2, "n/d"),
            "Latencia CPU 1 hilo (ms)": v(r.lat_cpu1_ms, 1, "n/d (requiere CUDA)" if r.modelo == "vmunet" else "n/d"),
            "Latencia CPU 6 hilos (ms)": v(r.lat_cpu6_ms, 1, "n/d (requiere CUDA)" if r.modelo == "vmunet" else "n/d"),
            "Latencia GPU (ms)": v(r.lat_cuda_ms, 2, "n/d"),
            "Calibración (ECE)": r.calibracion_ece,
            "Tipo": "Original" if orig else ("Referente" if base else "Red del E2"),
        }
        filas.append(fila)
    return filas


def columnas_tabla() -> list[dict]:
    num = {"Dice test": 4, "DE semillas": 4, "Dice val.": 4, "IoU test": 4, "GFLOPs": 2, "Latencia CPU 1 hilo (ms)": 1,
           "Latencia CPU 6 hilos (ms)": 1, "Latencia GPU (ms)": 2}
    cols = []
    for c in ["Red", "Familia", "Dice test", "DE semillas", "IC 95 % (grupos)", "Dice val.", "IoU test", "Parámetros", "GFLOPs",
              "Latencia CPU 1 hilo (ms)", "Latencia CPU 6 hilos (ms)", "Latencia GPU (ms)", "Calibración (ECE)", "Tipo"]:
        d = {"name": c, "id": c}
        if c in num:
            d.update(type="numeric", format=_num(num[c]))
        elif c == "Parámetros":
            d.update(type="numeric", format=_num(0, miles=True))
        else:
            d.update(type="text")
        cols.append(d)
    return cols


def layout():
    red = D.redes()
    base = D.fila("baseline_clasico")
    cd = D.meta_tabla()["nemenyi_cd"]
    opciones = [{"label": n, "value": m} for m, n in zip(red.modelo, red.nombre)]
    meta = D.meta_tabla()

    tabla = dash_table.DataTable(
        id="tabla-modelos", data=datos_tabla(), columns=columnas_tabla(), sort_action="native", filter_action="native",
        sort_mode="multi", page_size=12, style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "inherit", "fontSize": "13px", "padding": "6px 10px", "textAlign": "left",
                    "minWidth": "90px", "maxWidth": "240px", "whiteSpace": "normal"},
        style_header={"fontWeight": "700", "backgroundColor": "#EEF2F6", "borderBottom": "2px solid #1F4E79"},
        style_data_conditional=[
            {"if": {"filter_query": '{Tipo} = "Referente"'}, "fontStyle": "italic", "backgroundColor": "#F7F7F7"},
            {"if": {"filter_query": '{Tipo} = "Original"'}, "fontStyle": "italic", "backgroundColor": "#FBEFEF",
             "borderLeft": "4px solid #A4343A"},
        ],
        tooltip_header={"DE semillas": "Desviación estándar entre las cinco semillas", "GFLOPs": "Contador de operadores de PyTorch, entrada 1×3×256×256",
                        "Calibración (ECE)": "Error de calibración esperado; aún no medido"},
    )

    prediccion = ui.tarjeta(
        "Predicción en vivo",
        html.P("Cargue una fotografía PNG o JPG de hasta 5 MB. La imagen se procesa en memoria y no se almacena.", className="small"),
        dcc.Dropdown(id="sel-modelo-vivo", options=opciones, value="fcn", clearable=False, className="mb-2"),
        dcc.Upload(id="upload", children=html.Div(["Arrastre una imagen o ", html.A("seleccione un archivo")]),
                   className="zona-carga", multiple=False, accept="image/png,image/jpeg"),
        dcc.Loading(html.Div(id="salida-prediccion", className="mt-3")),
    )

    diagnostico = ui.tarjeta(
        "Diagnóstico por caso",
        dbc.Row([
            dbc.Col([html.Label("Redes a comparar (hasta 5)", htmlFor="sel-diag"),
                     dcc.Dropdown(id="sel-diag", options=opciones, value=["transunet_2d", "fcn", "segformer_b0"], multi=True)], md=8),
            dbc.Col([html.Label("Partición", htmlFor="sel-split"),
                     dcc.Dropdown(id="sel-split", options=[{"label": "Prueba", "value": "test"}, {"label": "Validación", "value": "val"}],
                                  value="test", clearable=False)], md=4),
        ], className="mb-2"),
        dbc.Row([dbc.Col([ui.grafico("g-dice-caso"), ui.interpretacion("", "t-dice-caso")], lg=6),
                 dbc.Col([ui.grafico("g-dice-area"), ui.interpretacion("", "t-dice-area")], lg=6)]),
        html.H6("Diez imágenes con menor Dice (primera red seleccionada)", className="mt-2"),
        html.Div(id="peores"),
        html.H6("Ejemplos de máscaras", className="mt-3"),
        html.P(["Consulta la imagen, la máscara real y las predicciones en el ",
                html.A("comparador aleatorio de test", href="#comparador-test"), " de esta página."]),
    )

    original = avance_colab_layout()

    return html.Div([
        html.H1("Modelos", className="mt-3"),
        html.P("Diez redes del Entregable 2 evaluadas en prueba con cinco semillas, más el baseline clásico como referente y la fila del modelo original.",
               className="lead"),
        comparador_layout(),
        original,
        ui.seccion("Resultados históricos del E2"),
        tabla,
        html.Small("Dice de etiqueta, promediado por imagen, grupo y semilla. IC95 remuestreando 122 grupos con semillas fijas. "
                   "Latencia: mediana con lote 1 y entrada 256 × 256 en CPU AMD Ryzen 5 PRO 6650U y en GPU; no es un benchmark de FPS. "
                   f"Tabla maestra generada el {meta['generado']}.", className="text-muted d-block mt-1"),
        dbc.Row([
            dbc.Col([ui.grafico("g-dice-ic", plots.fig_dice_ic(red, base)),
                     ui.interpretacion("Los intervalos de las cuatro primeras redes se solapan ampliamente: el orden de sus medias no "
                                       "establece diferencias claras. El baseline clásico queda por debajo de las diez redes.")],
                    lg=12),
        ], className="mt-3"),
        dbc.Row([
            dbc.Col([ui.grafico("g-pareto", plots.fig_pareto(red)),
                     ui.interpretacion("Solo TransUNet y la FCN compacta no están dominadas: la FCN logra casi el mismo Dice con "
                                       "unas 1.460 veces menos parámetros. Es una estimación puntual, sin dominancia estadística.")], lg=6),
            dbc.Col([ui.grafico("g-cd", plots.fig_cd(red, cd)),
                     ui.interpretacion(f"Dos redes con rangos medios a menos de CD = {D.fmt(cd, 2)} no se distinguen según Nemenyi al 5 %, "
                                       "bajo bloques independientes. No equivale al contraste de medias contra FCN.")], lg=6),
        ], className="mt-3"),
        dbc.Row([dbc.Col([ui.grafico("g-limitacion", plots.fig_limitacion(D.limitacion_area())),
                          ui.interpretacion("El gráfico compara el Dice por tamaño en validación. La sensibilidad a los rótulos "
                                            "pequeños motiva el modelo original; su magnitud varía entre redes.")], lg=12)]),
        ui.seccion("Predicción y diagnóstico"),
        dbc.Alert('FCN y U-Net en vivo usan la arquitectura a0498 seleccionada posteriormente; SegNet tiene un respaldo fuera de la aplicación. '
                  'La tabla de prueba anterior corresponde a las arquitecturas históricas. '
                  'La nueva selección se informa abajo por validación, sin atribuirle resultados de test antiguos.', color='info'),
        dash_table.DataTable(data=D.tabla_artefactos(),
                             columns=[{'name': c, 'id': c} for c in ['Modelo', 'Arquitectura', 'Semilla',
                                      'Dice val. media 5 semillas', 'Dice val. artefacto', 'Estado']],
                             style_table={'overflowX': 'auto'}, style_cell={'padding': '6px', 'textAlign': 'left'}),
        dbc.Row([dbc.Col(prediccion, lg=5), dbc.Col(diagnostico, lg=7)]),
    ])


def avance_colab_layout():
    path=D.DATA_DIR/'avance_colab.json'
    if not path.exists():return ui.pendiente('Resultados de Colab aún no integrados.')
    report=json.loads(path.read_text(encoding='utf-8'))
    names={'unetat_dbnet':'Base DBNet / U-Net','unetat_dbnet_at':'UNet-AT (original)',
           'segformer_b0':'SegFormer-B0 corregido'}
    rows=[{'Modelo':names[r['modelo']], 'Corridas':f"{r['n']}/{r['esperadas']}",
           'Dice validación':round(r['media'],4),'DE semillas':round(r['de'],4),
           'Estado':r['estado']} for r in report['resumen']]
    return ui.tarjeta('Avance verificado de Colab',
        html.P('Diez semillas por brazo del original completas; SegFormer parcial. '
               'Valores de mejor validación por corrida, no resultados de test.'),
        dash_table.DataTable(data=rows,columns=[{'name':k,'id':k} for k in rows[0]],
                             style_table={'overflowX':'auto'},style_cell={'padding':'9px','textAlign':'left'}),
        dbc.Alert(f"UNet-AT − base: {report['diferencia_val_at_menos_base']:+.4f} de Dice medio en validación. "
                  'Estos resultados no respaldan una mejora global del original; faltan evaluación final, '
                  'análisis por tamaño, ablaciones y contrastes.',color='info',className='mt-3'),
        html.A('Ficha de modelos y limitaciones',href=D.MODEL_CARD_URL))


def comparador_layout():
    ref=comparador_test.referencia()
    names=D.nombres()
    choices=[{'label':n,'value':m} for m,n in names.items() if m not in [ref['modelo'],'baseline_clasico']]
    return ui.tarjeta('Comparar redes con una imagen de test',
        html.P('Una misma fotografía, su máscara real y las predicciones de cada red. '
               'El caso se elige al azar, sin filtrar por el resultado de los modelos.'),
        dbc.Row([
            dbc.Col(dbc.Button('Elegir imagen aleatoria de test',id='test-aleatoria',color='primary',n_clicks=0),md=5),
            dbc.Col(dcc.Dropdown(id='test-modelos',options=choices,
                       value=[c['value'] for c in choices],multi=True,placeholder='Otras redes para comparar'),md=7),
        ],className='g-2 mb-3'),
        html.Small(f"Referencia: {ref['nombre']}. {ref['criterio']}",className='text-muted'),
        dcc.Store(id='test-caso-id',storage_type='memory'),
        html.Div(id='test-seleccion-estado',className='mt-2'),
        dcc.Loading(html.Div(id='test-comparacion',className='mt-3')),
        html.Small('Verde: máscara real. Naranja: predicción. Dice e IoU corresponden a esta imagen, '
                   'calculados a 256 × 256 sin relleno. Esta exploración no sustituye la evaluación final del E3.',
                   className='text-muted d-block mt-3'), id='comparador-test')


@callback(Output('test-caso-id','data'),Output('test-seleccion-estado','children'),
          Input('test-aleatoria','n_clicks'),State('test-caso-id','data'),prevent_initial_call=True)
def elegir_test(clicks,anterior):
    try:
        return comparador_test.elegir(anterior),''
    except (OSError,ValueError,KeyError):
        return dash.no_update,dbc.Alert('No se pudo abrir el paquete de test. En este PC usa ABRIR_DASHBOARD.cmd; '
                                        'en otra instancia conecta el paquete antes de usar esta opción.',color='warning')


def _tarjeta_prediccion(entry,reference=False):
    title=[entry['nombre']]
    if reference:title.append(dbc.Badge('Ganador de referencia E2',color='primary',className='ms-2'))
    if entry['estado']=='ok':
        body=[html.Img(src=entry['superposicion'],alt=f"Predicción de {entry['nombre']}",className='comparacion-imagen'),
              html.Div([html.Strong(f"Dice {entry['dice']:.3f}"),html.Span(f" · IoU {entry['iou']:.3f}")],className='mt-2'),
              html.Small(f"{entry['arquitectura']} · semilla {entry['semilla']} · precalculada",className='text-muted')]
    else:
        body=[dbc.Alert(entry['mensaje'],color='light' if entry['estado']=='pendiente' else 'warning')]
    return dbc.Card([dbc.CardHeader(title),dbc.CardBody(body)],
                    className='mb-3 comparacion-referencia' if reference else 'mb-3')


@callback(Output('test-comparacion','children'),Input('test-caso-id','data'),Input('test-modelos','value'))
def comparar_test(ident,models):
    if not ident:return dbc.Alert('Pulsa el botón para cargar un caso y comparar las redes.',color='light')
    winner=comparador_test.referencia()['modelo']
    selected=[m for m in (models or []) if m in D.nombres() and m not in [winner,'baseline_clasico']]
    try:
        result=comparador_test.caso(ident,[winner]+selected)
    except (OSError,ValueError,KeyError):
        return dbc.Alert('No se pudo verificar este caso del paquete de test. Selecciona otra imagen o revisa el paquete.',color='warning')
    reference=result['modelos'][0]
    return [html.P([html.Strong(f"Caso {result['id']}"),f" · grupo {result['group_id']} · partición test"]),
        dbc.Row([
            dbc.Col([html.H6('Imagen original'),html.Img(src=result['imagen'],alt='Fotografía original de test',className='comparacion-imagen')],md=6),
            dbc.Col([html.H6('Máscara real'),html.Img(src=result['real'],alt='Máscara real superpuesta en verde',className='comparacion-imagen')],md=6),
        ],className='g-3 mb-4'),
        dbc.Row([
            dbc.Col(html.Div(_tarjeta_prediccion(reference,True),className='comparacion-fija'),lg=5),
            dbc.Col(dbc.Row([dbc.Col(_tarjeta_prediccion(e),md=6) for e in result['modelos'][1:]],className='g-2'),lg=7),
        ],className='g-3')]


@callback(Output("salida-prediccion", "children"), Input("upload", "contents"), Input("sel-modelo-vivo", "value"),
          State("upload", "filename"))
def predecir(contents, modelo, filename):
    if not contents:
        return dbc.Alert("Aún no se ha cargado ninguna imagen.", color="light", className="py-2")
    ok, msg, raw = inference.validar_archivo(contents, filename)
    if not ok:
        return dbc.Alert(msg, color="danger", className="py-2")
    out = [html.Img(src=contents, alt="Imagen cargada para la predicción", className="img-fluid rounded mb-2",
                    style={"maxHeight": "260px"})]
    try:
        for a in inference.avisos_fuera_de_dominio(raw):
            out.append(dbc.Alert(["Posible entrada fuera de dominio. ", a], color="warning", className="py-2 mb-1"))
    except Exception:  # noqa: BLE001
        out.append(dbc.Alert("No se pudo evaluar si la imagen está dentro del dominio.", color="warning", className="py-2 mb-1"))
    try:
        res = inference.predecir(modelo, raw)
    except Exception:
        out.append(dbc.Alert('No se pudo ejecutar el artefacto verificado. Revisa su disponibilidad.', color='danger'))
        return out
    color = "info" if res["estado"] == "pendiente" else "success"
    out.append(dbc.Alert(f"{D.nombres().get(modelo, modelo)}: {res['mensaje']}.", color=color, className="py-2 mb-0"))
    if res['estado'] == 'ok':
        out.append(html.Img(src=res['superposicion'], alt='Máscara predicha superpuesta a la imagen',
                            className='img-fluid rounded mt-2', style={'maxHeight': '360px'}))
        out.append(html.Small('Esta predicción usa el artefacto indicado arriba. Las tablas de prueba siguen mostrando '
                              'el benchmark histórico E2; sus cifras no evalúan la nueva arquitectura a0498.',
                              className='text-muted d-block mt-2'))
    return out


@callback(Output("g-dice-caso", "figure"), Output("t-dice-caso", "children"), Output("g-dice-area", "figure"),
          Output("t-dice-area", "children"), Output("peores", "children"),
          Input("sel-diag", "value"), Input("sel-split", "value"))
def diagnostico(modelos, split):
    modelos = (modelos or [])[:5]
    dc = D.dice_por_caso()
    dc = dc[dc.particion == split]
    est = D.dice_por_area()
    cortes = D.calidad_resumen()["cortes_area_train_frac"]
    f1 = plots.fig_dice_caso(dc, modelos)
    f2 = plots.fig_dice_area(est, modelos, cortes)
    if not modelos:
        return f1, "Seleccione al menos una red.", f2, "", html.Div()
    nom = D.nombres()
    med = {m: float(dc[m].median()) for m in modelos}
    t1 = ("Medianas por imagen: " + "; ".join(f"{nom[m]} {D.fmt(v, 3)}" for m, v in med.items())
          + ". La cola izquierda son imágenes con rótulo casi no detectado.")
    q = est[est.modelo.isin(modelos)]
    t2 = ("Los cuartiles de área provienen de la prueba (E2); la barra es el IC95 por grupos. "
          "En todas las redes seleccionadas el cuartil de menor área tiene el Dice más bajo." if
          all(q[q.modelo == m].sort_values("Dice").iloc[0].cuartil == 1 for m in modelos if (q.modelo == m).any()) else
          "Los cuartiles de área provienen de la prueba (E2); la barra es el IC95 por grupos.")
    m0 = modelos[0]
    p = dc.nsmallest(10, m0)[["id", "clase", "area_pct", m0]].copy()
    p["clase"] = p.clase.map(D.CLASES_ES).fillna(p.clase)
    p = p.rename(columns={"id": "Imagen", "clase": "Categoría", "area_pct": "Área (%)", m0: f"Dice {nom[m0]}"})
    tabla = dash_table.DataTable(data=p.round(3).to_dict("records"), columns=[{"name": c, "id": c} for c in p.columns],
                                 style_cell={"fontSize": "12px", "padding": "4px 8px", "textAlign": "left"},
                                 style_header={"fontWeight": "700"}, style_table={"overflowX": "auto"})
    if split == "val":
        t2 += " El gráfico de área solo existe para prueba."
    return f1, t1, f2, t2, tabla
