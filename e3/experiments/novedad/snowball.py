"""Snowballing hacia atras (referencias) y hacia adelante (citas) con OpenAlex.

Semillas: los trabajos mas cercanos al mecanismo y al problema. Hacia adelante se
filtra en el servidor por palabras del mecanismo en titulo o resumen (shrink, kernel,
offset, unclip, expansion, adaptive, inscribed, thickness) y, para las semillas de
aplicacion, por palabras del problema. Guarda resultados/snowball__<semilla>.json.

Uso: python snowball.py
"""
import json
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

from buscar import get, abstract_oa, OUT

SEMILLAS = {
    "DBNet": "Real-time Scene Text Detection with Differentiable Binarization",
    "DBNetpp": "Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion",
    "Jiang2021": "Arbitrary-Shaped Text Detection with Adaptive Text Region Representation",
    "ASMTD": "Adaptive Shrink-Mask for Text Detection",
    "DeepHAZMAT": "DeepHAZMAT: Hazardous Materials Sign Detection and Segmentation with Restricted Computational Resources",
    "SegFormer": "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers",
}
MECANISMO = ('(shrink OR shrunk OR kernel OR unclip OR "expansion ratio" OR "kernel expansion" OR offset '
             'OR inscribed OR thickness OR "adaptive expansion" OR "learnable expansion")')
PROBLEMA = '(hazmat OR placard OR "dangerous goods" OR "hazardous material" OR "warning sign")'
FILTRO_ADELANTE = {"DBNet": MECANISMO, "DBNetpp": MECANISMO, "Jiang2021": None, "ASMTD": None,
                   "DeepHAZMAT": None, "SegFormer": PROBLEMA}
SEL = "id,doi,title,publication_year,cited_by_count,abstract_inverted_index,referenced_works"
B = "https://api.openalex.org/works"


def q(url):
    cod, txt = get(url)
    return cod, (json.loads(txt) if cod == 200 else None)


def resolver(titulo):
    cod, d = q(f"{B}?filter=title.search:{urllib.parse.quote(titulo)}&per-page=3&sort=cited_by_count:desc&select={SEL}")
    if cod != 200 or not d["results"]:
        return None
    return d["results"][0]


if __name__ == "__main__":
    for clave, titulo in SEMILLAS.items():
        ts = datetime.now().astimezone().isoformat(timespec="seconds")
        w = resolver(titulo)
        if not w:
            (OUT / f"snowball__{clave}.json").write_text(json.dumps(dict(
                semilla=clave, fecha=ts, estado="no accesible o sin resolver")), encoding="utf-8")
            print(clave, "sin resolver")
            continue
        wid = w["id"].rsplit("/", 1)[1]
        refs = [r.rsplit("/", 1)[1] for r in w.get("referenced_works", [])]
        # hacia atras: titulos de las referencias
        atras = []
        for i in range(0, len(refs), 50):
            lote = "|".join(refs[i:i + 50])
            cod, d = q(f"{B}?filter=openalex:{lote}&per-page=50&select=id,doi,title,publication_year")
            if d:
                atras += [dict(titulo=r["title"], anio=r["publication_year"], doi=r.get("doi")) for r in d["results"]]
            time.sleep(1)
        # hacia adelante: citas con filtro de palabras en el servidor
        filtro = f"cites:{wid}"
        extra = FILTRO_ADELANTE[clave]
        url = f"{B}?filter={filtro}"
        if extra:
            url += f",title_and_abstract.search:{urllib.parse.quote(extra)}"
        cod_t, dt = q(url + "&per-page=1&select=id")
        total_citas = None
        cod_c, dc = q(f"{B}?filter={filtro}&per-page=1&select=id")
        if dc:
            total_citas = dc["meta"]["count"]
        adelante, filtradas = [], None
        if dt:
            filtradas = dt["meta"]["count"]
            for pag in range(1, 5):  # hasta 200 registros filtrados
                cod, d = q(url + f"&per-page=50&page={pag}&sort=cited_by_count:desc&select=id,doi,title,publication_year,cited_by_count,abstract_inverted_index")
                if not d or not d["results"]:
                    break
                for r in d["results"]:
                    adelante.append(dict(titulo=r["title"], anio=r["publication_year"], doi=r.get("doi"),
                                         citas=r["cited_by_count"],
                                         resumen=abstract_oa(r.get("abstract_inverted_index"))[:900]))
                if len(adelante) >= filtradas:
                    break
                time.sleep(1)
        (OUT / f"snowball__{clave}.json").write_text(json.dumps(dict(
            semilla=clave, titulo_resuelto=w["title"], openalex=w["id"], doi=w.get("doi"), fecha=ts,
            estado="ok", referencias_declaradas=len(refs), referencias_resueltas=len(atras),
            citas_totales=total_citas, citas_filtradas=filtradas, filtro_adelante=extra,
            atras=atras, adelante=adelante), ensure_ascii=False, indent=1), encoding="utf-8")
        print(clave, w["title"][:60], "refs", len(refs), "citas", total_citas, "filtradas", filtradas, flush=True)
        time.sleep(1.5)
