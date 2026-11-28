"""Genera las cadenas de Scopus, Web of Science e IEEE Xplore a partir de los mismos
grupos de terminos de cadenas.md, sin comodines (los comodines dentro de frases no
funcionan igual en las tres bases), y escribe cadenas_suscripcion.md, que se pega en
pendiente_bases_suscripcion.md. Sin filtro de ano ni de tipo documental.

Uso: python generar_cadenas_suscripcion.py
"""
from pathlib import Path

H = ['hazmat', '"hazardous material"', '"hazardous materials"', '"dangerous goods"']
P = ['placard', 'placards', 'label', 'labels', 'sign', 'signs']
S = ['segmentation', 'detection', '"deep learning"']
DB = ['"differentiable binarization"', 'DBNet', '"shrink ratio"', '"kernel shrinking"', '"shrunk kernel"', '"shrunk kernels"']
ADAPT = ['adaptive', '"instance-adaptive"', '"area-aware"', '"size-adaptive"', 'inscribed', 'thickness', '"adaptive shrink"']
UNC = ['unclip', '"kernel expansion"', '"expansion ratio"', '"learnable expansion"', '"adaptive expansion"',
       '"adaptive unclip"', '"kernel reconstruction"', '"offset ratio"']
TXT = ['"text detection"', '"scene text"', '"instance segmentation"']
SHR = ['shrink', 'shrunk', 'shrinking', 'kernel', 'DBNet']

# id: (descripcion, [grupos unidos por AND], [grupos con NOT] )
CADENAS = {
    "C1": ("problema, EN", [H, P, S], []),
    "C2": ("problema, ES", [['"mercancías peligrosas"', '"materiales peligrosos"', '"sustancias peligrosas"'],
                           ['rótulo', 'rótulos', 'placa', 'placas', 'etiqueta', 'etiquetas'],
                           ['segmentación', 'detección', '"aprendizaje profundo"']], []),
    "C3": ("mecanismo C1: encogimiento adaptativo", [DB, ADAPT], []),
    "C4": ("mecanismo C2: expansión", [UNC, TXT, SHR], []),
    "C5": ("mecanismo + problema", [['"differentiable binarization"', 'DBNet', '"shrunk kernel"', '"kernel shrinking"', '"progressive scale expansion"'],
                                   ['hazmat', 'placard', 'placards', '"dangerous goods"', 'hazardous', '"warning sign"', '"warning signs"',
                                    '"traffic sign"', '"traffic signs"', '"road sign"', '"road signs"', 'pictogram', 'pictograms']], []),
    "C6": ("origen: otras aplicaciones", [['"differentiable binarization"', '"shrunk kernel"', '"kernel shrinking"'],
                                         ['segmentation', '"instance segmentation"', 'medical', 'crack', 'defect', 'building', 'road',
                                          '"remote sensing"', '"object detection"']], ['"text detection"']),
    "C7": ("mecanismo, ES", [['"binarización diferenciable"', '"binarizacion diferenciable"', '"encogimiento del núcleo"',
                             '"detección de texto en escenas"', '"reducción de núcleo"']], []),
    "C8": ("mecanismo C1: distancia, inscrito", [['"distance transform"', '"inscribed circle"', '"inscribed radius"', '"medial axis"', 'skeleton'],
                                                 ['shrink', 'shrunk', '"kernel map"', '"center region"', '"text region"'],
                                                 ['"instance segmentation"', '"text detection"', '"arbitrary-shaped"']], []),
}


def OR(terms, pref=""):
    return "(" + " OR ".join(pref + t for t in terms) + ")"


def scopus(g, n):
    cuerpo = " AND ".join(OR(x) for x in g) + "".join(f" AND NOT {OR(x)}" for x in n)
    return f"TITLE-ABS-KEY({cuerpo})"


def wos(g, n):
    cuerpo = " AND ".join(OR(x) for x in g) + "".join(f" NOT {OR(x)}" for x in n)
    return f"TS=({cuerpo})"


def ieee(g, n):
    p = '"All Metadata":'
    cuerpo = " AND ".join(OR(x, p) for x in g) + "".join(f" NOT {OR(x, p)}" for x in n)
    return cuerpo


if __name__ == "__main__":
    out = []
    for cid, (desc, g, n) in CADENAS.items():
        out += [f"### {cid}: {desc}", "", "Scopus (Advanced search):", "```text", scopus(g, n), "```", "",
                "Web of Science Core Collection (Advanced search):", "```text", wos(g, n), "```", "",
                "IEEE Xplore (Command Search, pestaña Command Search):", "```text", ieee(g, n), "```", ""]
    (Path(__file__).parent / "cadenas_suscripcion.md").write_text("\n".join(out), encoding="utf-8")
    print("ok")
