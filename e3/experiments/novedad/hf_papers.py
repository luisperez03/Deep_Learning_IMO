"""Busqueda en Hugging Face Papers (sucesor de Papers with Code) para las consultas de
mecanismo y problema. Guarda resultados/huggingface_papers.json.

Uso: python hf_papers.py
"""
import json
import urllib.parse
from datetime import datetime

from buscar import get, OUT

CONSULTAS = ["hazmat", "differentiable binarization", "text kernel expansion shrink",
             "hazardous material placard segmentation", "adaptive shrink ratio text detection"]

if __name__ == "__main__":
    salida = dict(fecha=datetime.now().astimezone().isoformat(timespec="seconds"), consultas=[])
    for q in CONSULTAS:
        url = "https://huggingface.co/api/papers/search?q=" + urllib.parse.quote(q)
        cod, txt = get(url)
        reg = []
        if cod == 200:
            for r in json.loads(txt):
                p = r["paper"]
                reg.append(dict(titulo=" ".join(p["title"].split()), anio=(p.get("publishedAt") or "")[:4],
                                doi="arXiv:" + p["id"]))
        salida["consultas"].append(dict(consulta=q, url=url,
                                        estado="ok" if cod == 200 else f"no accesible (HTTP {cod})",
                                        devueltos=len(reg), registros=reg))
        print(q, cod, len(reg), flush=True)
    (OUT / "huggingface_papers.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
