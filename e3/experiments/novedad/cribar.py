"""Lista compacta de los resultados crudos para el cribado manual (titulo y anio),
marcando con * los que mencionan palabras del mecanismo en titulo o resumen.

Uso: python cribar.py <patron_de_archivo> [maximo_por_archivo]
     python cribar.py "*__C3" 30
"""
import json
import re
import sys
from pathlib import Path

CLAVES = re.compile(r"shrink|shrunk|kernel|unclip|expan|offset|inscribed|thickness|differentiable binar|"
                    r"hazmat|placard|dangerous goods|hazardous", re.I)
sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).parent / "resultados"
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
mx = int(sys.argv[2]) if len(sys.argv) > 2 else 50
for f in sorted(OUT.glob(pat + ".json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    print(f"\n### {f.stem} | {d['fecha']} | {d['estado']} | total={d['total_declarado']} | devueltos={d['devueltos']}")
    for i, r in enumerate(d["registros"][:mx], 1):
        marca = "*" if CLAVES.search((r.get("titulo") or "") + " " + (r.get("resumen") or "")) else " "
        print(f"{i:2d}{marca} {r.get('anio')} {(r.get('titulo') or '')[:110]} | {r.get('doi') or ''}")
