"""Snowballing hacia adelante de los trabajos de rotulos hazmat mas cercanos, con
Semantic Scholar (citas por DOI) y OpenAlex (version de revista de DeepHAZMAT).
Guarda resultados/snowball_hazmat.json.

Uso: python snowball_hazmat.py
"""
import json
import time
import urllib.parse
from datetime import datetime

from buscar import get, abstract_oa, OUT

SEMILLAS = {
    "DeepHAZMAT (MLWA 2021)": "10.1016/j.mlwa.2021.100104",
    "Zhang et al. 2024, fusion visual y textual (TAI)": "10.1109/TAI.2023.3298588",
    "Mohamed et al. 2018 (SSRR)": "10.1109/SSRR.2018.8468639",
    "Zhao et al. 2013 (GlobalSIP)": "10.1109/GlobalSIP.2013.6736996",
    "HMD-Net 2022 (Entropy)": "10.3390/e24040466",
}

if __name__ == "__main__":
    salida = dict(fecha=datetime.now().astimezone().isoformat(timespec="seconds"), semillas=[])
    for nombre, doi in SEMILLAS.items():
        url = (f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}/citations"
               "?fields=title,year,externalIds,abstract&limit=100")
        cod, txt = get(url, tries=5, pausa=8.0)
        item = dict(semilla=nombre, doi=doi, fuente="semanticscholar", url=url)
        if cod == 200:
            d = json.loads(txt)
            cit = [dict(titulo=c["citingPaper"].get("title"), anio=c["citingPaper"].get("year"),
                        doi=(c["citingPaper"].get("externalIds") or {}).get("DOI"),
                        resumen=(c["citingPaper"].get("abstract") or "")[:600]) for c in d.get("data", [])]
            item.update(estado="ok", n_citas_devueltas=len(cit), citas=cit)
        else:
            item.update(estado=f"no accesible (HTTP {cod})", citas=[])
        salida["semillas"].append(item)
        print(nombre, item["estado"], item.get("n_citas_devueltas"), flush=True)
        time.sleep(3)
    (OUT / "snowball_hazmat.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
