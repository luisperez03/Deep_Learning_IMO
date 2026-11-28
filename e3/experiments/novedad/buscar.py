"""Ejecuta las cadenas de cadenas.md en fuentes abiertas y guarda el crudo.

Fuentes: OpenAlex, arXiv, Semantic Scholar (bulk), OpenReview, GitHub, Hugging Face Hub.
Salida: resultados/<fuente>__<cadena>.json con fecha, URL consultada, conteo total
declarado por la fuente y los primeros N registros. Si una fuente no responde, el
estado queda como 'no accesible' con el codigo HTTP.

Uso: python buscar.py            (todas las cadenas, todas las fuentes)
     python buscar.py C3 arxiv   (una cadena, una fuente)
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).parent / "resultados"
OUT.mkdir(exist_ok=True)
UA = {"User-Agent": "revision-literatura-academica/1.0"}
N = 50

# Cada cadena: texto para OpenAlex (booleano), arXiv (all:), Semantic Scholar bulk (+ |),
# OpenReview (term plano), GitHub/HF (palabras clave).
CADENAS = {
    "C1": dict(
        desc="Problema: placas/rotulos hazmat, EN",
        oa='(hazmat OR "hazardous material" OR "hazardous materials" OR "dangerous goods") AND (placard OR placards OR label OR labels OR sign OR signs) AND (segmentation OR detection OR "deep learning")',
        arxiv='all:(hazmat OR "hazardous material" OR "dangerous goods") AND all:(placard OR label OR sign) AND all:(segmentation OR detection)',
        ss='(hazmat | "hazardous material" | "dangerous goods") + (placard | placards | label | sign) + (segmentation | detection)',
        orv="hazmat placard segmentation",
        gh="hazmat placard segmentation", hf="hazmat",
    ),
    "C2": dict(
        desc="Problema: rotulos de mercancias peligrosas, ES",
        oa='("mercancías peligrosas" OR "materiales peligrosos" OR "sustancias peligrosas") AND (rótulo OR rótulos OR placa OR etiqueta OR señal) AND (segmentación OR detección OR "aprendizaje profundo")',
        arxiv='all:"mercancías peligrosas"',
        ss='("mercancías peligrosas" | "materiales peligrosos") + (rótulo | placa | etiqueta)',
        orv="mercancias peligrosas rotulo",
        gh="mercancias peligrosas rotulos segmentacion", hf="mercancias peligrosas",
    ),
    "C3": dict(
        desc="Mecanismo: DBNet + razon de encogimiento adaptativa por tamano/grosor/radio inscrito",
        oa='("differentiable binarization" OR DBNet OR "shrunk kernel" OR "kernel shrinking" OR "shrink ratio") AND (adaptive OR "instance-adaptive" OR "area-aware" OR "size-adaptive" OR inscribed OR thickness OR "adaptive shrink")',
        arxiv='all:("differentiable binarization" OR DBNet OR "shrink ratio" OR "kernel shrinking") AND all:(adaptive OR inscribed OR thickness)',
        ss='("differentiable binarization" | DBNet | "shrink ratio" | "kernel shrinking" | "shrunk kernel") + (adaptive | inscribed | thickness | "size-adaptive")',
        orv="differentiable binarization adaptive shrink ratio",
        gh="adaptive shrink ratio DBNet", hf="DBNet",
    ),
    "C4": dict(
        desc="Mecanismo: expansion/unclip/offset del nucleo encogido (adaptativa, aprendida, inversa)",
        oa='(unclip OR "kernel expansion" OR "expansion ratio" OR "learnable expansion" OR "adaptive expansion" OR "adaptive unclip" OR "kernel reconstruction" OR "offset ratio") AND ("text detection" OR "scene text" OR "instance segmentation") AND (shrink OR shrunk OR kernel OR DBNet)',
        arxiv='all:(unclip OR "kernel expansion" OR "expansion ratio" OR "learnable expansion" OR "kernel reconstruction") AND all:("text detection" OR "scene text")',
        ss='(unclip | "kernel expansion" | "expansion ratio" | "learnable expansion" | "adaptive expansion" | "kernel reconstruction") + ("text detection" | "scene text")',
        orv="adaptive unclip kernel expansion text detection",
        gh="unclip ratio adaptive DBNet expansion", hf="unclip",
    ),
    "C5": dict(
        desc="Mecanismo + problema: DBNet / encogimiento de nucleo aplicado a placas hazmat o senales",
        oa='("differentiable binarization" OR DBNet OR "shrunk kernel" OR "kernel shrinking" OR "progressive scale expansion") AND (hazmat OR placard OR "dangerous goods" OR "hazardous" OR "warning sign" OR "traffic sign" OR "road sign" OR pictogram)',
        arxiv='all:("differentiable binarization" OR DBNet OR "progressive scale expansion") AND all:(hazmat OR placard OR "dangerous goods" OR "traffic sign" OR "warning sign")',
        ss='("differentiable binarization" | DBNet | "progressive scale expansion") + (hazmat | placard | "dangerous goods" | "traffic sign" | "warning sign")',
        orv="differentiable binarization sign segmentation",
        gh="DBNet hazmat OR placard OR sign segmentation", hf="dbnet sign",
    ),
    "C6": dict(
        desc="Origen: DBNet y familia (nucleos encogidos) en otras aplicaciones no textuales",
        oa='("differentiable binarization" OR "shrunk kernel" OR "kernel shrinking") AND (segmentation OR "instance segmentation" OR "medical" OR crack OR defect OR building OR road OR "remote sensing" OR "object detection") NOT "text detection"',
        arxiv='all:"differentiable binarization" AND all:(segmentation OR medical OR crack OR defect OR "remote sensing")',
        ss='("differentiable binarization" | "kernel shrinking" | "shrunk kernel") + (segmentation | medical | crack | defect | "remote sensing") - "text detection"',
        orv="differentiable binarization segmentation non-text",
        gh="differentiable binarization segmentation medical OR crack OR defect", hf="differentiable binarization",
    ),
    "C7": dict(
        desc="Mecanismo, ES: binarizacion diferenciable / encogimiento de nucleo",
        oa='("binarización diferenciable" OR "binarizacion diferenciable" OR "encogimiento del núcleo" OR "detección de texto en escenas" OR "reducción de núcleo")',
        arxiv='all:"binarización diferenciable"',
        ss='("binarización diferenciable" | "encogimiento del núcleo" | "detección de texto en escena")',
        orv="binarizacion diferenciable",
        gh="binarizacion diferenciable segmentacion", hf="binarizacion diferenciable",
    ),
    "C8": dict(
        desc="Mecanismo: etiqueta por transformada de distancia / radio inscrito / grosor de instancia, encogimiento",
        oa='("distance transform" OR "inscribed circle" OR "inscribed radius" OR "medial axis" OR "skeleton") AND (shrink OR shrunk OR "kernel map" OR "center region" OR "text region") AND ("instance segmentation" OR "text detection" OR "arbitrary-shaped")',
        arxiv='all:("distance transform" OR "inscribed" OR "medial axis") AND all:(shrink OR "kernel") AND all:("text detection" OR "instance segmentation")',
        ss='("distance transform" | "inscribed" | "medial axis") + (shrink | shrunk | kernel) + ("text detection" | "instance segmentation")',
        orv="inscribed shrink kernel text detection",
        gh="inscribed radius shrink kernel label generation", hf="shrink kernel",
    ),
}


def get(url, tries=4, pausa=5.0):
    """Devuelve (codigo, texto). Reintenta en 429/5xx."""
    ultimo = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            ultimo = (e.code, e.read().decode("utf-8", "replace")[:300])
            if e.code in (429, 500, 502, 503):
                time.sleep(pausa * (i + 1))
                continue
            return ultimo
        except Exception as e:  # red caida, tiempo agotado
            ultimo = (0, str(e)[:300])
            time.sleep(pausa)
    return ultimo


def abstract_oa(inv):
    if not inv:
        return ""
    pos = {}
    for w, ps in inv.items():
        for p in ps:
            pos[p] = w
    return " ".join(pos[k] for k in sorted(pos))


def q_openalex(c):
    sel = "id,doi,title,publication_year,cited_by_count,abstract_inverted_index,primary_location"
    url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(c["oa"]) +
           f"&per-page={N}&select={sel}")
    cod, txt = get(url)
    if cod != 200:
        return url, cod, None, []
    d = json.loads(txt)
    reg = [dict(titulo=w.get("title"), anio=w.get("publication_year"), doi=w.get("doi"),
                id=w.get("id"), citas=w.get("cited_by_count"),
                resumen=abstract_oa(w.get("abstract_inverted_index"))[:900])
           for w in d.get("results", [])]
    return url, cod, d["meta"]["count"], reg


def q_arxiv(c):
    url = ("https://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(c["arxiv"]) +
           f"&max_results={N}&sortBy=relevance")
    cod, txt = get(url)
    if cod != 200:
        return url, cod, None, []
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    root = ET.fromstring(txt)
    total = int(root.find("o:totalResults", ns).text)
    reg = []
    for e in root.findall("a:entry", ns):
        reg.append(dict(titulo=re.sub(r"\s+", " ", e.find("a:title", ns).text.strip()),
                        anio=e.find("a:published", ns).text[:4],
                        doi=e.find("a:id", ns).text,
                        resumen=re.sub(r"\s+", " ", e.find("a:summary", ns).text.strip())[:900]))
    return url, cod, total, reg


def q_ss(c):
    flds = "title,year,externalIds,abstract,citationCount"
    url = ("https://api.semanticscholar.org/graph/v1/paper/search/bulk?query=" +
           urllib.parse.quote(c["ss"]) + f"&fields={flds}&sort=citationCount:desc")
    cod, txt = get(url, tries=5, pausa=8.0)
    if cod != 200:
        return url, cod, None, []
    d = json.loads(txt)
    reg = [dict(titulo=p.get("title"), anio=p.get("year"),
                doi=(p.get("externalIds") or {}).get("DOI") or (p.get("externalIds") or {}).get("ArXiv"),
                id=p.get("paperId"), citas=p.get("citationCount"),
                resumen=(p.get("abstract") or "")[:900]) for p in d.get("data", [])[:N]]
    return url, cod, d.get("total"), reg


def q_orv(c):
    url = ("https://api2.openreview.net/notes/search?term=" + urllib.parse.quote(c["orv"]) +
           f"&limit={N}")
    cod, txt = get(url)
    if cod != 200:
        return url, cod, None, []
    d = json.loads(txt)
    reg = []
    for n in d.get("notes", []):
        co = n.get("content", {})
        t = (co.get("title") or {}).get("value")
        ab = (co.get("abstract") or {}).get("value", "")
        reg.append(dict(titulo=t, anio=None, doi="https://openreview.net/forum?id=" + n.get("forum", ""),
                        resumen=ab[:900]))
    return url, cod, d.get("count"), reg


def q_gh(c):
    url = ("https://api.github.com/search/repositories?q=" + urllib.parse.quote(c["gh"]) +
           f"&per_page={N}&sort=stars")
    cod, txt = get(url, pausa=20.0)
    if cod != 200:
        return url, cod, None, []
    d = json.loads(txt)
    reg = [dict(titulo=r["full_name"], anio=r["created_at"][:4], doi=r["html_url"],
                citas=r["stargazers_count"], resumen=(r.get("description") or "")[:300])
           for r in d.get("items", [])]
    return url, cod, d.get("total_count"), reg


def q_hf(c):
    reg_all = []
    total = 0
    last = None
    for tipo in ("models", "datasets", "spaces"):
        url = (f"https://huggingface.co/api/{tipo}?search=" + urllib.parse.quote(c["hf"]) +
               f"&limit={N}")
        cod, txt = get(url)
        last = (url, cod)
        if cod != 200:
            return url, cod, None, reg_all
        for r in json.loads(txt):
            total += 1
            reg_all.append(dict(titulo=f"[{tipo}] {r.get('id')}", anio=None,
                                doi=f"https://huggingface.co/{'' if tipo=='models' else tipo+'/'}{r.get('id')}",
                                resumen=" ".join(r.get("tags", [])[:8])))
    return last[0], 200, total, reg_all


FUENTES = dict(openalex=q_openalex, arxiv=q_arxiv, semanticscholar=q_ss,
               openreview=q_orv, github=q_gh, huggingface=q_hf)

# GitHub no hace AND de muchas palabras: se usan consultas cortas, una por tema.
GH_CORTAS = ["hazmat segmentation", "hazmat placard", "hazmat detection", "dangerous goods label detection",
             "hazmat-13", "DeepHAZMAT", "adaptive shrink ratio text detection", "unclip_ratio adaptive",
             "differentiable binarization segmentation", "text kernel expansion", "shrink ratio DBNet",
             "IMDG label detection"]


def gh_cortas():
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    salida = []
    for qy in GH_CORTAS:
        url, cod, total, reg = q_gh(dict(gh=qy))
        salida.append(dict(consulta=qy, url=url, estado="ok" if cod == 200 else f"no accesible (HTTP {cod})",
                           total_declarado=total, devueltos=len(reg), registros=reg[:15]))
        print(qy, cod, total, flush=True)
        time.sleep(7)  # limite sin autenticar: 10 consultas por minuto
    (OUT / "github__cortas.json").write_text(json.dumps(dict(fecha=ts, consultas=salida),
                                             ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] == ["gh_cortas"]:
        gh_cortas()
        sys.exit(0)
    args = sys.argv[1:]
    cads = [a for a in args if a in CADENAS] or list(CADENAS)
    fuen = [a for a in args if a in FUENTES] or list(FUENTES)
    for cid in cads:
        for fn in fuen:
            f = OUT / f"{fn}__{cid}.json"
            ts = datetime.now().astimezone().isoformat(timespec="seconds")
            url, cod, total, reg = FUENTES[fn](CADENAS[cid])
            estado = "ok" if cod == 200 else f"no accesible (HTTP {cod})"
            f.write_text(json.dumps(dict(cadena=cid, fuente=fn, fecha=ts, url=url, estado=estado,
                                         total_declarado=total, devueltos=len(reg), registros=reg),
                                    ensure_ascii=False, indent=1), encoding="utf-8")
            print(cid, fn, estado, total, len(reg), flush=True)
            time.sleep(1.5)
