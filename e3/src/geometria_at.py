"""Geometria de DBNet-AT: etiquetas (C1), expansion consistente (C2) y posproceso. Modulo autonomo.

Fuentes:
- DBNet: Liao, Wan, Yao, Chen y Bai (2020), "Real-time Scene Text Detection with Differentiable Binarization",
  AAAI. De ahi salen el encogimiento D = A(1 - r^2)/L con r = 0,4, el mapa de umbral y la expansion D' = A' r'/L'
  con r' = 1,5. Los valores 0,3 (binarizacion) y 0,5 (puntaje minimo de region) son los del codigo oficial.
- Formulacion de DBNet-AT: entregable_3/diseno.md, seccion 3 (C1 en 3.2, C2 en 3.3, teorema en 3.5).

Notacion (la del teorema de 3.5):
- K es una region (poligono convexo si se quiere la garantia exacta). A, L y g son su area, su perimetro y su
  radio inscrito (la mayor distancia de un pixel de K al fondo).
- r = 0,4 es la razon de encogimiento de DBNet; g0 = 4 px es el paso del mapa mas profundo de la U-Net compacta
  (1/4 de 256), fijado por la arquitectura.
- phi(g) = g^2 / (g^2 + g0^2) es la atenuacion por grosor de C1. Con g0 = 0, phi = 1 y se recupera DBNet.
- La regla de encogimiento es rho(K) = phi(g) (1 - r^2) A / L. El nucleo es K_{-D} = {x en K : dist(x, borde) >= D}
  con D = rho(K).
- El desplazamiento en inglete de un poligono P = interseccion de {n_j . x <= c_j} es
  M(P, d) = interseccion de {n_j . x <= c_j + d}: se mueve cada lado d hacia afuera y se cortan los consecutivos.
- C2 busca el d que cumple d = rho(M(K', d)), con K' el nucleo predicho. Teorema: si K es convexo, ningun lado
  colapsa al encogerse (S1) y sus angulos interiores son >= 60 grados (S2), la solucion es unica, d = D, y
  M(K', D) = K.

Cambio respecto a la version de entregable_3/codigo/etiquetas_at.py: el desplazamiento en inglete ya no lo hace
pyclipper (algoritmo de Vatti) sino el codigo de este archivo (inglete_poligono / rellenar_inglete). pyclipper
queda solo como prueba cruzada en tests/test_original_model.py.

Todo lo demas viene de E2: instancias por componente conexo (vecindad 8), perimetro con cv2.arcLength, encoger
y expandir con la transformada de distancia, umbrales del mapa 0,3-0,7, binarizacion 0,3, puntaje minimo 0,5 y
area minima 9 px.
"""
import cv2
import numpy as np
from scipy import ndimage

# ----------------------------------------------------------------------------- constantes
G0 = 4.0            # px: paso del mapa mas profundo de la U-Net compacta (e3, 1/4 de 256)
R = 0.4             # razon de encogimiento de DBNet
R_PRIMA = 1.5       # razon de expansion fija de DBNet
LIMITE_INGLETE = 2.0  # limite de inglete (multiplos de d). Una esquina de 90 grados necesita raiz(2)
ESCALA = 16         # las coordenadas del relleno van en 1/16 de pixel (cv2.fillPoly con shift = 4)
AJUSTE_PX = 0.75    # corrimiento de discretizacion del contorno (calibrar_ajuste.py, 200 imagenes de train)
OCHO = np.ones((3, 3), bool)


# ----------------------------------------------------------------------------- primitivas de E2 (reescritas)
def instancias(mascara):
    """Cada componente conexo (vecindad 8) de la mascara es una instancia. Devuelve (etiquetas, n)."""
    return ndimage.label(mascara > 0, structure=OCHO)


def area_perimetro(inst):
    """A = numero de pixeles; L = longitud del contorno externo (cv2.arcLength). L >= 1 para no dividir por 0."""
    contornos, _ = cv2.findContours(inst.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in contornos)
    return float(inst.sum()), max(float(perim), 1.0)


def encoger(inst, d):
    """Nucleo: los pixeles a distancia > d del fondo (equivale al desplazamiento de Vatti con esquinas vivas
    en poligonos convexos, Lema 1 de diseno.md 3.5)."""
    return ndimage.distance_transform_edt(inst) > d


def expandir(inst, d):
    """Expansion euclidea: se anaden los pixeles de fondo a distancia <= d (esquinas redondeadas, como DBNet)."""
    return ndimage.distance_transform_edt(~inst) <= d


def componentes(binaria, area_min):
    """Etiqueta los componentes de una mascara y devuelve (etiquetas, lista de etiquetas con area >= area_min)."""
    etq, n = instancias(binaria)
    if n == 0:
        return etq, []
    areas = ndimage.sum(np.ones_like(etq), etq, index=np.arange(1, n + 1))
    return etq, [i + 1 for i, a in enumerate(areas) if a >= area_min]


# ----------------------------------------------------------------------------- C1: la regla de encogimiento
def phi(g, g0=G0):
    """Atenuacion por grosor de C1: g^2 / (g^2 + g0^2). Con g0 <= 0 vale 1 (C1 apagado, regla de DBNet)."""
    return 1.0 if g0 <= 0 else g * g / (g * g + g0 * g0)


def regla(region, r=R, g0=G0):
    """rho(K) = phi(g) (1 - r^2) A / L: distancia de encogimiento de una region (bool)."""
    A, L = area_perimetro(region)
    g = float(ndimage.distance_transform_edt(region).max()) if region.any() else 0.0
    return phi(g, g0) * A * (1 - r ** 2) / L


def at_objetivos(y, v, r=R, g0=G0, tmin=0.3, tmax=0.7):
    """Etiquetas de entrenamiento (las de DBNet con D = rho(K) en lugar de A(1 - r^2)/L).

    y: mascara binaria (H, W) de rotulos; v: zona valida (bool). Devuelve, con un eje de canal delante:
    db_gt (nucleo, el blanco de P y de B), db_mask (donde cuenta la perdida), db_umbral (blanco de T, entre tmin y
    tmax) y db_umbral_mask (la banda donde cuenta la perdida de T). La banda de T usa el mismo D que el nucleo.
    Una instancia cuyo nucleo desaparece se ignora (db_mask = 0), como el codigo oficial de DBNet.
    Con g0 = 0 es exactamente la etiqueta de DBNet."""
    etq, n = instancias(y)
    gt = np.zeros(y.shape, np.float32)
    mask = v.astype(np.float32).copy()
    umbral = np.zeros(y.shape, np.float32)
    umbral_mask = np.zeros(y.shape, np.float32)
    for k in range(1, n + 1):
        inst = etq == k
        D = regla(inst, r, g0)
        s = encoger(inst, D)
        if not s.any():
            mask[inst] = 0
            continue
        gt[s] = 1
        gd = expandir(inst, D)
        dist = np.where(inst, ndimage.distance_transform_edt(inst), ndimage.distance_transform_edt(~inst))
        t = np.clip(1 - dist / max(D, 1e-6), 0, 1)  # 1 en el borde de la region, 0 a distancia D
        umbral[gd] = np.maximum(umbral[gd], t[gd])
        umbral_mask[gd] = 1
    umbral = umbral * (tmax - tmin) + tmin
    return {"db_gt": gt[None], "db_mask": mask[None], "db_umbral": umbral[None], "db_umbral_mask": umbral_mask[None]}


# ----------------------------------------------------------------------------- inglete propio (reemplaza a pyclipper)
def _orientar(p):
    """Poligono (n, 2) float sin vertices repetidos y con area con signo positiva (orden antihorario en el
    plano x-y; en coordenadas de imagen, con y hacia abajo, es el horario visual: no importa, la aritmetica es la
    misma). Con esa orientacion la normal exterior del lado (dx, dy) es (dy, -dx)/|e|."""
    p = np.asarray(p, float)
    q = p[np.any(np.abs(p - np.roll(p, 1, axis=0)) > 1e-9, axis=1)]  # quita vertices consecutivos repetidos
    area2 = np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
    return q if area2 >= 0 else q[::-1]


def _geometria_lados(p):
    """Vectores unitarios t_i y normales exteriores n_i de los lados i = p[i] -> p[i+1] de un poligono orientado."""
    e = np.roll(p, -1, axis=0) - p
    t = e / np.linalg.norm(e, axis=1, keepdims=True)
    n = np.stack([t[:, 1], -t[:, 0]], axis=1)
    return t, n


def _esquina(v, t1, n1, t2, n2, d, limite):
    """Puntos de la esquina desplazada en el vertice v, entre el lado entrante (t1, n1) y el saliente (t2, n2).

    Inglete: la interseccion de las dos rectas desplazadas, m = v + d (n1 + n2) / (1 + n1 . n2). Esta a una
    distancia d / sen(theta/2) de v (theta = angulo interior). Si esa distancia pasa de limite * d, la punta se
    corta con una recta perpendicular a la bisectriz, a limite * d de v (Clipper la llama 'squaring'); entonces
    salen dos puntos, uno sobre cada recta desplazada. Con limite = 2 el corte aparece bajo 60 grados (supuesto S2
    del teorema). Devuelve una lista de 1 o 2 puntos."""
    den = 1.0 + float(n1 @ n2)
    if den > 1e-12 and 1.0 / np.sqrt(den / 2.0) <= limite:
        return [v + d * (n1 + n2) / den]
    u = n1 + n2
    nu = np.linalg.norm(u)
    u = u / nu if nu > 1e-12 else t1  # giro de 180 grados: se corta perpendicular al lado entrante
    L = limite * d
    pts = []
    for t, n in ((t1, n1), (t2, n2)):
        ct = float(t @ u)
        if abs(ct) < 1e-12:
            pts.append(v + d * n)
        else:
            pts.append(v + d * n + ((L - d * float(n @ u)) / ct) * t)
    return pts


def inglete_poligono(p, d, limite=LIMITE_INGLETE):
    """Desplazamiento en inglete M(P, d) de un poligono CONVEXO p (n, 2): cada lado d hacia afuera y se cortan los
    consecutivos (diseno.md 3.5). Devuelve el poligono desplazado (m, 2); m >= n si hay esquinas cortadas.

    Solo es correcto para poligonos convexos y d >= 0. Con un poligono no convexo, las esquinas entrantes
    cruzarian los lados: para eso esta rellenar_inglete, que usa piezas convexas."""
    p = _orientar(p)
    t, n = _geometria_lados(p)
    salida = []
    for i in range(len(p)):
        j = i - 1  # lado entrante al vertice i
        salida.extend(_esquina(p[i], t[j], n[j], t[i], n[i], d, limite))
    return np.asarray(salida)


def _piezas_inglete(p, d, limite=LIMITE_INGLETE):
    """Piezas convexas cuya union es M(P, d) para CUALQUIER poligono simple (convexo o no): el poligono mismo, un
    rectangulo sobre cada lado (de p[i], p[i+1] a d hacia afuera) y una cuna en cada vertice convexo (la esquina de
    _esquina). En un vertice entrante no se pone cuna: los rectangulos de sus dos lados ya se solapan alli.
    Caso limite documentado: el desplazamiento por union de piezas no recorta entrantes mas cortos que d; es el
    desplazamiento 'exacto' del conjunto, no el contorno que daria un recorte de bucles como el de Vatti."""
    p = _orientar(p)
    t, n = _geometria_lados(p)
    k = len(p)
    piezas = [p]
    for i in range(k):
        a, b = p[i], p[(i + 1) % k]
        piezas.append(np.array([a, b, b + d * n[i], a + d * n[i]]))
        j = i - 1
        if t[j][0] * t[i][1] - t[j][1] * t[i][0] > 1e-12:  # vertice convexo (giro hacia el interior)
            punta = _esquina(p[i], t[j], n[j], t[i], n[i], d, limite)
            piezas.append(np.array([p[i], p[i] + d * n[j], *punta, p[i] + d * n[i]]))
    return piezas


def _pintar(salida, poligono):
    """Rellena un poligono float en la mascara uint8 (coordenadas en 1/16 de pixel, sin suavizado)."""
    pts = np.round(np.asarray(poligono) * ESCALA).astype(np.int32)
    cv2.fillPoly(salida, [pts], 1, lineType=cv2.LINE_8, shift=4)


def _contornos(c):
    """Contornos externos del nucleo c (bool), simplificados con Douglas-Peucker a 1 px para que la escalera de
    pixeles de un borde inclinado no genere picos al unir en inglete."""
    cont, _ = cv2.findContours(c.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [(cv2.approxPolyDP(k, 1.0, True) if len(k) > 4 else k)[:, 0, :].astype(float) for k in cont]


def rellenar_inglete(forma, contornos, d, limite=LIMITE_INGLETE):
    """Mascara uint8 (forma) con el desplazamiento en inglete, d hacia afuera, de los contornos dados.
    Contorno convexo: un solo poligono (inglete_poligono). No convexo: union de piezas (_piezas_inglete)."""
    salida = np.zeros(forma, np.uint8)
    for p in contornos:
        if len(p) < 3:
            continue
        if cv2.isContourConvex(np.round(p).astype(np.int32)[:, None, :]):
            _pintar(salida, inglete_poligono(p, d, limite))
        else:
            for pieza in _piezas_inglete(p, d, limite):
                _pintar(salida, pieza)
    return salida


def expandir_inglete(c, d, ajuste=None, limite=LIMITE_INGLETE):
    """Desplaza hacia afuera el contorno del nucleo c en d px con uniones en inglete (esquinas vivas) y rellena.

    El contorno de cv2 pasa por los centros de los pixeles del borde y el relleno no coincide exactamente con el
    conjunto de pixeles de la etiqueta: se desplaza d + ajuste (AJUSTE_PX, un corrimiento de discretizacion que no
    depende del tamano y se calibro solo con imagenes de train). Devuelve una mascara bool que incluye a c."""
    desp = d + (AJUSTE_PX if ajuste is None else ajuste)
    if desp <= 0:
        return c.copy()
    salida = rellenar_inglete(c.shape, _contornos(c), desp, limite)
    return (salida > 0) | c


# ----------------------------------------------------------------------------- C2: expansion consistente
def expansion_consistente(comp, v, r=R, g0=G0, iteraciones=16, inglete=True, ajuste=None):
    """d tal que regla(expandir(comp, d) & v) = d, por biseccion; devuelve la region expandida (bool, tamano
    completo). Es la inversa de C1: M(K', d) con d = rho(M(K', d)) (diseno.md 3.3). La region se recorta a su caja
    con un margen suficiente para que los calculos sean locales.
    inglete=True expande con esquinas vivas; False, por distancia euclidea (esquinas redondas)."""
    ys, xs = np.nonzero(comp)
    g_k = float(ndimage.distance_transform_edt(comp).max())
    alto = 6.0 * (g_k + 1.0) + 2.0  # cota: en una franja larga d ~ 5,25 g'; en una compacta d ~ 0,72 g'
    m = int(np.ceil(1.5 * alto)) + 2  # el inglete a 90 grados sale raiz(2) d por la esquina
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, comp.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, comp.shape[1])
    c, vv = comp[y0:y1, x0:x1], v[y0:y1, x0:x1]
    fuera = None if inglete else ndimage.distance_transform_edt(~c)

    def expandida(d):
        return (expandir_inglete(c, d, ajuste) if inglete else fuera <= d) & vv

    def h(d):  # h(d) = rho(M(K', d)) - d es estrictamente decreciente (Lema 2): una sola raiz
        return regla(expandida(d), r, g0) - d

    bajo = 0.0
    if h(alto) > 0:  # no deberia pasar; si pasa, se usa la cota y se sigue
        d = alto
    else:
        for _ in range(iteraciones):
            medio = (bajo + alto) / 2
            if h(medio) > 0:
                bajo = medio
            else:
                alto = medio
        d = (bajo + alto) / 2
    salida = np.zeros(comp.shape, bool)
    salida[y0:y1, x0:x1] = _refinar(c, d, expandida, r, g0)
    return salida


def _refinar(c, d_bis, expandida, r, g0, ventana=(-2.5, 1.5), paso=0.25):
    """Inversa en pixeles. La biseccion da un d continuo, pero la region expandida cambia a saltos (un anillo de
    pixeles cada vez) y la region correcta puede quedar en el escalon de al lado del cruce (en un cuadrado de 32 px
    con C1, la etiqueta usa D = 6,77 y la region original aparece para d en [6,0; 6,75), donde regla - d > 0).
    Entre las regiones distintas que da d en la ventana [d - 2,5; d + 1,5] (pasos de 0,25) se elige la que,
    encogida con la regla de la etiqueta, reproduce mejor el nucleo observado (IoU). Empate: la de d mas cercano
    al cruce de la biseccion."""
    vistas, mejor = set(), (-1.0, 0.0, None)
    for d in np.arange(max(d_bis + ventana[0], 0.0), d_bis + ventana[1] + 1e-9, paso):
        reg = expandida(float(d))
        clave = (int(reg.sum()), hash(np.packbits(reg).tobytes()))
        if clave in vistas or not reg.any():
            continue
        vistas.add(clave)
        nuc = encoger(reg, regla(reg, r, g0))
        inter = int((nuc & c).sum())
        iou = inter / (int(nuc.sum()) + int(c.sum()) - inter)
        if iou > mejor[0] + 1e-12 or (abs(iou - mejor[0]) <= 1e-12 and abs(d - d_bis) < mejor[1]):
            mejor = (iou, abs(d - d_bis), reg)
    return mejor[2] if mejor[2] is not None else expandida(d_bis)


# ----------------------------------------------------------------------------- posproceso
def at_post(prob, v, umbral=0.3, puntaje_min=0.5, r=R, g0=G0, consistente=True, inglete=True, r_prima=R_PRIMA,
            area_min=9, ajuste=None):
    """Inferencia: binarizar P con 0,3, descartar componentes de area < 9 px o P media < 0,5 (como el codigo
    oficial de DBNet) y expandir cada nucleo. consistente=True: con el d de C2. consistente=False: con r' fijo,
    como DBNet (D' = A' r'/L'), por distancia euclidea o, con inglete=True, en inglete (ablacion factorial)."""
    etq, keep = componentes((prob > umbral) & v, area_min)
    salida = np.zeros(prob.shape, bool)
    for k in keep:
        comp = etq == k
        if prob[comp].mean() < puntaje_min:
            continue
        if consistente == "okr":
            salida |= expandir(comp, distancia_okr(comp))
        elif consistente:
            salida |= expansion_consistente(comp, v, r, g0, inglete=inglete, ajuste=ajuste)
        else:
            A, L = area_perimetro(comp)
            salida |= expandir_inglete(comp, A * r_prima / L, ajuste) if inglete else expandir(comp, A * r_prima / L)
    return salida & v


def distancia_okr(comp):
    """Distancia de dilatacion de OKR (Ma et al., 2023, ICASSP, doi 10.1109/ICASSP49357.2023.10094734, Ec. 2), el
    trabajo mas cercano (experiments/novedad/lectura_okr.md): S = (C - sqrt(max(C^2 - 16 A, 0))) / 4, con C y A el
    perimetro y el area de la region predicha. En un rectangulo a x b, C^2 - 16A = 4 (a - b)^2 y S = min(a, b): el
    lado corto del nucleo. S es el TAMANO del elemento estructurante de la dilatacion (un diametro), igual que el
    S de su erosion (Ec. 1, S = min lado / 2), asi que la distancia por lado es S / 2: el nucleo de ancho w/2 vuelve
    a ancho w. Lectura declarada: el articulo redondea S a impar y no precisa la forma del elemento; aqui se usa
    S / 2 sin redondeo y la expansion euclidea (expandir), para comparar solo la regla que elige la distancia contra
    el punto fijo de C2."""
    A, C = area_perimetro(comp)
    return (C - np.sqrt(max(C * C - 16.0 * A, 0.0))) / 8.0


# Variantes de la ablacion factorial 2^3 (diseno.md 5.1): nombre -> (g0, d consistente, inglete).
# Ademas, una variante de comparacion con el trabajo mas cercano (protocolo §8): 'okr_sobre_at' usa los nucleos del
# brazo dbnet_at (C1) y la dilatacion de OKR en lugar de C2. Es exploratoria, fuera de la familia de la ablacion.
# 'dbnet' es el base: regla y posproceso de DBNet (g0 = 0, r' fijo, expansion euclidea).
VARIANTES = {
    "dbnet": (0.0, False, False),
    "c1_rfijo": (G0, False, False),        # solo encogimiento atenuado, con r' = 1,5
    "inglete_rfijo": (0.0, False, True),   # regla de DBNet, r' = 1,5, expansion en inglete (C2b solo)
    "c1_inglete_rfijo": (G0, False, True),  # C1 con r' = 1,5 en inglete (sin C2a)
    "c2_redondo": (0.0, True, False),      # regla de DBNet, d consistente, expansion euclidea
    "c2": (0.0, True, True),               # regla de DBNet, d consistente, expansion en inglete
    "at_redondo": (G0, True, False),       # C1 + d consistente con expansion euclidea
    "dbnet_at": (G0, True, True),          # el modelo propio: C1 + C2
    "okr_sobre_at": (G0, "okr", False),    # nucleos C1 + dilatacion de OKR (comparacion con el trabajo mas cercano)
}


def objetivos_de(variante):
    """Funcion (y, v) -> objetivos de entrenamiento de la variante. Solo importa g0 (C1)."""
    g0 = VARIANTES[variante][0]
    return lambda y, v: at_objetivos(y, v, g0=g0)


def post_de(variante):
    """Funcion (P, v) -> mascara predicha de la variante (posproceso con o sin C2, con o sin inglete)."""
    g0, cons, ing = VARIANTES[variante]
    return lambda p, v: at_post(p, v, g0=g0, consistente=cons, inglete=ing)


# DBNet base, con los nombres de E2
def db_objetivos(y, v, r=R, tmin=0.3, tmax=0.7):
    return at_objetivos(y, v, r=r, g0=0.0, tmin=tmin, tmax=tmax)


def db_post(prob, v, **kw):
    return at_post(prob, v, g0=0.0, consistente=False, inglete=False, **kw)
