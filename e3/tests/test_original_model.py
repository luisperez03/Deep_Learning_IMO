"""Seis pruebas obligatorias del modelo original (UNetAT, e3/src/original_model.py). Ejecutar con:
    pytest e3/tests          (o, sin pytest, python e3/tests/correr_sin_pytest.py)

(i)   sobreajusta un lote diminuto sintetico (rombos y rectangulos sobre fondo, 64 x 64);
(ii)  la misma semilla da las mismas salidas;
(iii) formas de salida y entradas invalidas;
(iv)  gradiente no nulo y finito en todos los parametros;
(v)   la propiedad del teorema (diseno.md 3.5), verificada numericamente, y el inglete propio contra pyclipper;
(vi)  guardar y cargar el state_dict da salidas iguales; exportacion ONNX con prueba de ida y vuelta.
"""
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

E3 = Path(__file__).resolve().parents[1]
if str(E3) not in sys.path:
    sys.path.insert(0, str(E3))
from src import geometria_at as G  # noqa: E402
from src.original_model import UNetAT, perdida_db  # noqa: E402

JSON_GEOMETRIA = E3.parent / "resultados" / "geometria_sintetica.json"
TAM = 256


# ---------------------------------------------------------------------------------------- utilidades
def _poligono(forma, puntos):
    m = np.zeros(forma, np.uint8)
    cv2.fillPoly(m, [(np.asarray(puntos, float) * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m


def _rombo(lado, dx=0.0, forma=(TAM, TAM)):
    h, c = lado / np.sqrt(2), forma[0] / 2 + dx
    return _poligono(forma, [[c, c - h], [c + h, c], [c, c + h], [c - h, c]])


def _rect(alto, aspecto, dx=0.0, forma=(TAM, TAM)):
    w, c = alto * aspecto, forma[0] / 2 + dx
    return _poligono(forma, [[c - w / 2, c - alto / 2], [c + w / 2, c - alto / 2],
                             [c + w / 2, c + alto / 2], [c - w / 2, c + alto / 2]])


def _dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return 2.0 * (a & b).sum() / (a.sum() + b.sum())


def _oraculo(variante, y):
    """Dice de la forma y al pasar el nucleo de su etiqueta por el posproceso de la variante (prediccion perfecta)."""
    v = np.ones(y.shape, bool)
    nucleo = G.objetivos_de(variante)(y, v)["db_gt"][0]
    return _dice(G.post_de(variante)(nucleo, v), y)


def _lote_sintetico(n=4, tam=64):
    """n imagenes tam x tam: un rombo o un rectangulo de color sobre fondo de otro color, con algo de ruido fijo.
    Devuelve x (n, 3, tam, tam) float32, mascaras (n, tam, tam) uint8 y los objetivos de DBNet-AT como tensores."""
    rng = np.random.RandomState(7)
    xs, ys = [], []
    for i in range(n):
        y = np.zeros((tam, tam), np.uint8)
        c = tam / 2 + rng.uniform(-6, 6, 2)
        if i % 2 == 0:  # rombo de lado 18-24
            h = rng.uniform(18, 24) / np.sqrt(2)
            y = _poligono((tam, tam), [[c[0], c[1] - h], [c[0] + h, c[1]], [c[0], c[1] + h], [c[0] - h, c[1]]])
        else:  # rectangulo de 24 a 32 por 12 a 16
            w, a = rng.uniform(24, 32), rng.uniform(12, 16)
            y = _poligono((tam, tam), [[c[0] - w / 2, c[1] - a / 2], [c[0] + w / 2, c[1] - a / 2],
                                       [c[0] + w / 2, c[1] + a / 2], [c[0] - w / 2, c[1] + a / 2]])
        fondo, frente = rng.uniform(-1, 0, 3), rng.uniform(0.3, 1, 3)
        x = np.where(y[None] > 0, frente[:, None, None], fondo[:, None, None]) + 0.05 * rng.randn(3, tam, tam)
        xs.append(x.astype(np.float32))
        ys.append(y)
    obj = [G.at_objetivos(y, np.ones(y.shape, bool)) for y in ys]
    tensores = {k: torch.from_numpy(np.stack([o[k] for o in obj])) for k in obj[0]}
    return torch.from_numpy(np.stack(xs)), np.stack(ys), tensores


# ---------------------------------------------------------------------------------------- (i) sobreajuste
def test_i_sobreajusta_un_lote_diminuto():
    x, ys, obj = _lote_sintetico()
    modelo = UNetAT(semilla=0)
    opt = torch.optim.Adam(modelo.parameters(), lr=3e-3)
    modelo.train()
    inicial = None
    for paso in range(250):
        opt.zero_grad()
        perdida = perdida_db(modelo(x), obj)
        perdida.backward()
        opt.step()
        inicial = float(perdida) if inicial is None else inicial
    final = float(perdida)
    assert final < 0.35 * inicial, f"la perdida no bajo lo suficiente: {inicial:.3f} -> {final:.3f}"
    modelo.eval()
    mascaras = modelo.predict_mask(x, "dbnet_at")
    dice = np.mean([_dice(m, y) for m, y in zip(mascaras, ys)])
    assert dice >= 0.85, f"Dice del lote sobreajustado: {dice:.3f}"


# ---------------------------------------------------------------------------------------- (ii) determinismo
def test_ii_misma_semilla_mismas_salidas():
    x = torch.from_numpy(np.random.RandomState(1).randn(2, 3, 32, 32).astype(np.float32))
    a, b, c = UNetAT(semilla=5).eval(), UNetAT(semilla=5).eval(), UNetAT(semilla=6).eval()
    assert torch.equal(a(x), b(x))
    assert not torch.equal(a(x), c(x))
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    # crear con semilla no toca el generador global
    torch.manual_seed(123)
    antes = torch.rand(1)
    torch.manual_seed(123)
    UNetAT(semilla=99)
    assert torch.equal(antes, torch.rand(1))
    # en entrenamiento tambien (no hay Dropout ni BatchNorm: el modelo es determinista)
    sa, sb = UNetAT(semilla=5).train()(x), UNetAT(semilla=5).train()(x)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


# ---------------------------------------------------------------------------------------- (iii) formas y entradas
def test_iii_formas_y_entradas_invalidas():
    modelo = UNetAT(semilla=0)
    x = torch.zeros(2, 3, 64, 48)
    modelo.train()
    s = modelo(x)
    assert set(s) >= {"prob", "thresh", "binary"}
    for k in ("prob", "thresh", "binary", "logit_prob"):
        assert s[k].shape == (2, 1, 64, 48), k
    for k in ("prob", "thresh", "binary"):
        assert float(s[k].min()) >= 0.0 and float(s[k].max()) <= 1.0
    modelo.eval()
    p = modelo(x)
    assert isinstance(p, torch.Tensor) and p.shape == (2, 1, 64, 48)
    assert torch.allclose(p, modelo.predict_proba(x))
    assert modelo.predict_mask(x).shape == (2, 64, 48) and modelo.predict_mask(x).dtype == bool
    # entradas invalidas: errores claros, no fallos oscuros dentro de una convolucion
    with pytest.raises(ValueError, match="3 canales"):
        modelo(torch.zeros(1, 1, 64, 64))
    with pytest.raises(ValueError, match="3 canales"):
        modelo(torch.zeros(1, 4, 64, 64))
    with pytest.raises(ValueError, match="multiplos de 4"):
        modelo(torch.zeros(1, 3, 66, 64))
    with pytest.raises(ValueError, match="multiplos de 4"):
        modelo.predict_proba(torch.zeros(1, 3, 64, 62))
    with pytest.raises(ValueError, match="B, 3, H, W"):
        modelo(torch.zeros(3, 64, 64))
    with pytest.raises(ValueError, match="tensor"):
        modelo(np.zeros((1, 3, 64, 64), np.float32))
    with pytest.raises(ValueError, match="flotante"):
        modelo(torch.zeros(1, 3, 64, 64, dtype=torch.int64))
    with pytest.raises(ValueError, match="variante"):
        modelo.predict_mask(x, "inexistente")
    with pytest.raises(ValueError):
        UNetAT(cabeza="5x5")


# ---------------------------------------------------------------------------------------- (iv) gradientes
def test_iv_gradiente_no_nulo_y_finito_en_todos_los_parametros():
    x, _, obj = _lote_sintetico(n=2)
    for cabeza in ("1x1", "3x3"):
        modelo = UNetAT(semilla=0, cabeza=cabeza).train()
        perdida = perdida_db(modelo(x), obj)
        assert torch.isfinite(perdida)
        perdida.backward()
        for nombre, p in modelo.named_parameters():
            assert p.grad is not None, f"{cabeza}: {nombre} sin gradiente"
            assert torch.isfinite(p.grad).all(), f"{cabeza}: {nombre} con gradiente no finito"
            assert float(p.grad.abs().sum()) > 0, f"{cabeza}: {nombre} con gradiente nulo"
    # derivada de la binarizacion: dB/dP = k B (1 - B), dB/dT = -k B (1 - B)
    p = torch.tensor([0.4, 0.5, 0.6], requires_grad=True)
    t = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
    b = torch.sigmoid(50.0 * (p - t))
    b.sum().backward()
    esperado = 50.0 * b.detach() * (1 - b.detach())
    assert torch.allclose(p.grad, esperado, atol=1e-5) and torch.allclose(t.grad, -esperado, atol=1e-5)


# ---------------------------------------------------------------------------------------- (v) la propiedad
SUB = [i / 8 for i in range(8)]  # las 8 posiciones subpixel de geometria_sintetica.py


def _medio(variante, formas):
    return float(np.mean([_oraculo(variante, y) for y in formas]))


def test_v_expansion_consistente_reconstruye_rectangulos():
    """Teorema de diseno.md 3.5 con la regla de DBNet (phi = 1, variante 'c2'): la expansion consistente en inglete
    reconstruye rectangulos de aspecto 1, 2 y 4 con Dice >= 0,99; la expansion fija de DBNet no (aspecto 4)."""
    for aspecto in (1, 2, 4):
        formas = [_rect(32, aspecto, d) for d in SUB]
        assert _medio("c2", formas) >= 0.99, f"c2, aspecto {aspecto}"
    dbnet4 = _medio("dbnet", [_rect(32, 4, d) for d in SUB])
    assert dbnet4 < 0.90, f"DBNet en aspecto 4 deberia fallar: {dbnet4:.3f}"
    # el cociente se degrada con el aspecto en DBNet y no en C2: es el corolario de 3.5
    assert _medio("dbnet", [_rect(32, 1, d) for d in SUB]) > dbnet4


def test_v_expansion_consistente_reconstruye_rombos():
    """Rombos de lado >= 10 px. El teorema es exacto en el continuo; en pixeles la inversa no es unica en +-1 px
    (diseno.md 3.3 y 3.5), asi que el Dice >= 0,99 no se cumple en todos los tamanos con ninguna de las dos reglas.
    Se verifica lo que si se sostiene: Dice >= 0,99 en los tamanos donde la discretizacion es exacta (10 y 64, las
    dos reglas) y, en todos los lados de 10 a 96, un promedio >= 0,96 con la mejor de las dos y un minimo >= 0,88."""
    lados = (10, 12, 16, 24, 32, 48, 64, 96)
    dice = {v: {l: _medio(v, [_rombo(l, d) for d in SUB]) for l in lados} for v in ("c2", "dbnet_at")}
    for variante in dice:
        for lado in (10, 64):
            assert dice[variante][lado] >= 0.99, f"{variante}, rombo {lado}: {dice[variante][lado]:.4f}"
        assert min(dice[variante].values()) >= 0.88, (variante, dice[variante])
    mejor = [max(dice["c2"][l], dice["dbnet_at"][l]) for l in lados]
    assert np.mean(mejor) >= 0.96, mejor


def test_v_coincide_con_resultados_previos_con_pyclipper():
    """Las cifras de entregable_3/resultados/geometria_sintetica.json se calcularon con el inglete de pyclipper; el
    inglete propio las reproduce (tolerancia 0,005 sobre el promedio de las 8 posiciones subpixel)."""
    if not JSON_GEOMETRIA.exists():
        pytest.skip("falta resultados/geometria_sintetica.json")
    previo = json.loads(JSON_GEOMETRIA.read_text())
    casos = [("rectangulos_alto32_por_aspecto", "4", [_rect(32, 4, d) for d in SUB]),
             ("rectangulos_alto32_por_aspecto", "2", [_rect(32, 2, d) for d in SUB]),
             ("rombos_por_lado_px", "24", [_rombo(24, d) for d in SUB]),
             ("rombos_por_lado_px", "10", [_rombo(10, d) for d in SUB])]
    for grupo, clave, formas in casos:
        for variante in ("dbnet", "c2", "dbnet_at"):
            esperado = previo[grupo][clave][variante]["dice"]
            obtenido = _medio(variante, formas)
            assert abs(obtenido - esperado) <= 0.005, f"{grupo} {clave} {variante}: {obtenido:.4f} vs {esperado}"


def _inglete_pyclipper(c, d, limite=2.0, escala=16):
    """Referencia: el desplazamiento en inglete de pyclipper (algoritmo de Vatti), sobre el mismo contorno."""
    pyclipper = pytest.importorskip("pyclipper")
    salida = np.zeros(c.shape, np.uint8)
    off = pyclipper.PyclipperOffset(miter_limit=limite)
    for k in G._contornos(c):
        off.AddPath((k * escala).astype(np.int64).tolist(), pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    for q in off.Execute(d * escala):
        cv2.fillPoly(salida, [np.asarray(q, np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return salida > 0


def _iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_distancia_okr_es_medio_lado_corto_en_rectangulos():
    """La Ec. 2 de OKR da S = min(a, b) en un nucleo rectangular a x b; la distancia por lado es S / 2 (el area se
    cuenta en pixeles y el perimetro sobre el contorno de cv2, que pasa por los centros: de ahi la tolerancia)."""
    for a, b in [(10, 40), (16, 16), (24, 60)]:
        m = np.zeros((96, 96), bool)
        m[20:20 + a, 10:10 + b] = True
        assert abs(G.distancia_okr(m) - min(a, b) / 2) <= 1.0, (a, b, G.distancia_okr(m))
    m = np.zeros((96, 96), bool)
    m[30:50, 20:70] = True
    v = np.ones_like(m)
    salida = G.post_de("okr_sobre_at")(m.astype(np.float32), v)
    assert salida.sum() > m.sum()  # la variante OKR corre y expande


def test_v_inglete_propio_coincide_con_pyclipper_en_convexos():
    forma = (160, 160)
    convexos = {
        "cuadrado": [[50, 50], [100, 50], [100, 100], [50, 100]],
        "rectangulo": [[30, 60], [130, 60], [130, 90], [30, 90]],
        "rombo": [[80, 40], [120, 80], [80, 120], [40, 80]],
        "hexagono": [[60, 40], [100, 40], [120, 80], [100, 120], [60, 120], [40, 80]],
        "pentagono": [[80, 30], [125, 65], [110, 120], [50, 120], [35, 65]],
        "triangulo_70": [[40, 120], [120, 120], [80, 50]],
    }
    for nombre, pts in convexos.items():
        c = _poligono(forma, pts) > 0
        for d in (2.0, 5.0, 9.0):
            propio = G.rellenar_inglete(forma, G._contornos(c), d) > 0
            ref = _inglete_pyclipper(c, d)
            assert _iou(propio, ref) >= 0.98, f"{nombre}, d = {d}: IoU {_iou(propio, ref):.4f}"
    # el limite de inglete: una punta de 30 grados se corta a 2 d de su vertice (aqui y en pyclipper)
    agudo = _poligono(forma, [[20, 80], [140, 65], [140, 95]]) > 0
    propio = G.rellenar_inglete(forma, G._contornos(agudo), 6.0) > 0
    assert _iou(propio, _inglete_pyclipper(agudo, 6.0)) >= 0.95
    # no convexo (forma de L): el camino de piezas coincide con pyclipper
    ele = _poligono(forma, [[40, 40], [70, 40], [70, 90], [120, 90], [120, 120], [40, 120]]) > 0
    propio = G.rellenar_inglete(forma, G._contornos(ele), 5.0) > 0
    assert _iou(propio, _inglete_pyclipper(ele, 5.0)) >= 0.95
    # d = 0 no cambia el poligono, y el desplazamiento nunca encoge
    assert _iou(G.rellenar_inglete(forma, G._contornos(ele), 0.0) > 0, ele) >= 0.97


def test_v_variante_dbnet_equivale_a_e2():
    """Con g0 = 0 y r' fijo, la etiqueta y el posproceso son los de E2 (entregable_2/codigo/etiquetas.py)."""
    e2 = E3.parents[1] / "entregable_2" / "codigo"
    if not (e2 / "etiquetas.py").exists():
        pytest.skip("falta entregable_2/codigo/etiquetas.py")
    sys.path.insert(0, str(e2))
    try:
        import etiquetas as E2
    finally:
        sys.path.remove(str(e2))
    y = np.maximum(_rombo(30, 0.3, (TAM, TAM)), _poligono((TAM, TAM), [[20, 20], [70, 20], [70, 40], [20, 40]]))
    v = np.ones(y.shape, bool)
    propio, e2o = G.db_objetivos(y, v), E2.db_objetivos(y, v)
    for k in e2o:
        assert np.allclose(propio[k], e2o[k]), k
    nucleo = propio["db_gt"][0]
    assert np.array_equal(G.db_post(nucleo, v), E2.db_post(nucleo, v))


# ---------------------------------------------------------------------------------------- (vi) guardar y cargar
def test_vi_guardar_y_cargar_state_dict():
    x = torch.from_numpy(np.random.RandomState(3).randn(2, 3, 32, 32).astype(np.float32))
    a = UNetAT(semilla=1).eval()
    buf = io.BytesIO()
    torch.save(a.state_dict(), buf)
    buf.seek(0)
    b = UNetAT(semilla=2).eval()
    assert not torch.equal(a(x), b(x))
    b.load_state_dict(torch.load(buf, map_location="cpu", weights_only=True))
    assert torch.equal(a(x), b(x))
    assert torch.equal(a.predict_proba(x), b.predict_proba(x))


class _RedSinPosproceso(torch.nn.Module):
    """La red tal cual (P, T, B), para exportar a ONNX: sin el posproceso, que es de numpy."""

    def __init__(self, modelo):
        super().__init__()
        self.m = modelo

    def forward(self, x):
        h = self.m._caracteristicas(x)
        p = torch.sigmoid(self.m.cabeza_prob(h))
        t = torch.sigmoid(self.m.cabeza_umbral(h))
        return p, t, torch.sigmoid(self.m.k * (p - t))


def test_vi_exportacion_onnx_ida_y_vuelta(tmp_path=None):
    pytest.importorskip("onnx")  # torch.onnx.export lo exige aun con dynamo=False
    ort = pytest.importorskip("onnxruntime")
    import tempfile
    modelo = UNetAT(semilla=4).eval()
    red = _RedSinPosproceso(modelo).eval()
    x = torch.from_numpy(np.random.RandomState(5).randn(1, 3, 64, 64).astype(np.float32))
    with tempfile.TemporaryDirectory() as d:
        ruta = str(Path(d) / "unetat.onnx")
        torch.onnx.export(red, (x,), ruta, input_names=["imagen"], output_names=["prob", "thresh", "binary"],
                          opset_version=17, dynamo=False)
        sesion = ort.InferenceSession(ruta, providers=["CPUExecutionProvider"])
        salidas = sesion.run(None, {"imagen": x.numpy()})
    with torch.no_grad():
        esperadas = red(x)
    for onnx_, torch_ in zip(salidas, esperadas):
        assert np.allclose(onnx_, torch_.numpy(), atol=1e-4)
