"""Reconstruye el costo de computo de GPU del trabajo a partir de lo registrado, y una cota de energia.

  python scripts/costo_total_e2.py       # escribe resultados/costo_e2/costo_total_e2.json y .md

Solo suma tiempos que quedaron en archivos del repo; no estima lo que no se registro:
  1. entrenamiento de las 50 corridas: `cost.hours_recorded` de resultados/integracion_e2/analisis.json
     (suma de los segundos de epoca de cada corrida; excluye carga de datos fuera de epoca, interrupciones, E/S);
  2. diagnosticos en GPU (limpio, robustez, interpretabilidad): stage_seconds + load_seconds + verification_seconds de
     resultados/diagnosticos_gpu_e2/tiempos/*.json;
  3. complemento de entrenamiento (ablacion, permutacion, HPO): columna `segundos` de
     resultados/complemento_entreno_e2/corridas.csv, mas los segundos de la prueba de sobreajuste a un lote.
Sin tiempo registrado (no se suman): evaluacion en test (T4), replica de SegFormer (T4), tiempo de sesion de Colab,
extraccion de validacion por imagen, e instalacion de dependencias.
Energia: cota superior = TDP publicado de la GPU x horas registradas. Es energia de la GPU a plena potencia, no medida:
no incluye CPU, memoria ni el centro de datos, y la GPU rara vez trabaja a TDP. TDP: NVIDIA L4 = 72 W y Tesla T4 = 70 W
(fichas tecnicas del fabricante). Asignacion de GPU: entrenamiento y diagnosticos en L4 (PROTOCOLO_DIAGNOSTICOS.json,
auditoria_test_e2); el complemento se asume L4 (estimacion_computo.json habla de horas L4; las corridas no guardan GPU).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
R = RAIZ / "resultados"
TDP_W = {"NVIDIA L4": 72, "Tesla T4": 70}


def main():
    analisis = json.loads((R / "integracion_e2/analisis.json").read_text(encoding="utf-8"))
    entren = {c["model"]: c["hours_recorded"] for c in analisis["cost"]}

    diag = {}
    for p in sorted((R / "diagnosticos_gpu_e2/tiempos").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        diag[d["stage"]] = {"gpu": d["device"], "corridas": len(d["runs"]), "horas": (
            sum(r["stage_seconds"] + r.get("load_seconds", 0) for r in d["runs"])
            + d.get("verification_seconds", 0)) / 3600, "archivo": p.name}

    comp = defaultdict(lambda: [0, 0.0])
    with open(R / "complemento_entreno_e2/corridas.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = r["corrida"].split("_")[0]
            comp[k][0] += 1
            comp[k][1] += float(r["segundos"]) / 3600
    resumen = json.loads((R / "complemento_entreno_e2/RESUMEN.json").read_text(encoding="utf-8"))
    sobre_h = sum(v["segundos"] for v in resumen["sobreajuste"].values()) / 3600

    bloques = [("Entrenamiento de las 50 corridas (segundos de época)", "NVIDIA L4", sum(entren.values()), 50)]
    nombres = {"clean": "Diagnóstico: evaluación limpia", "robustness": "Diagnóstico: robustez",
               "interpret": "Diagnóstico: interpretabilidad"}
    for k, v in diag.items():
        bloques.append((nombres.get(k, k), v["gpu"], v["horas"], v["corridas"]))
    nombres_c = {"abl": "Complemento: ablación", "perm": "Complemento: permutación del pipeline",
                 "hpo": "Complemento: búsqueda de hiperparámetros"}
    for k, (n, h) in comp.items():
        bloques.append((nombres_c.get(k, "Complemento: " + k), "NVIDIA L4", h, n))
    bloques.append(("Complemento: sobreajuste a un lote (10 redes)", "NVIDIA L4", sobre_h, 10))

    total = sum(b[2] for b in bloques)
    energia = sum(b[2] * TDP_W[b[1]] for b in bloques) / 1000  # kWh
    out = {"bloques": [{"concepto": c, "gpu": g, "horas": h, "corridas": n, "kwh_cota_tdp": h * TDP_W[g] / 1000}
                       for c, g, h, n in bloques],
           "horas_gpu_registradas": total, "kwh_cota_superior_tdp": energia, "tdp_w": TDP_W,
           "entrenamiento_por_modelo_horas": entren,
           "sin_registro": ["evaluación en test (T4)", "réplica de SegFormer (T4)", "tiempo total de sesión de Colab",
                            "extracción de Dice por imagen de validación", "instalación de dependencias"],
           "advertencias": ["horas = segundos registrados, no tiempo de pared de la sesión",
                            "energía = TDP x horas (cota de la GPU), no medida",
                            "GPU del complemento asumida L4"]}
    (R / "costo_e2").mkdir(exist_ok=True)
    (R / "costo_e2/costo_total_e2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["| Concepto | GPU | Corridas | Horas | kWh (cota TDP) |", "|---|---|---|---|---|"]
    md += [f"| {c} | {g} | {n} | {h:.2f} | {h*TDP_W[g]/1000:.2f} |" for c, g, h, n in bloques]
    md.append(f"| **Total registrado** | | | **{total:.2f}** | **{energia:.2f}** |")
    (R / "costo_e2/costo_total_e2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\nPor modelo (entrenamiento, h):", {k: round(v, 2) for k, v in entren.items()})


if __name__ == "__main__":
    main()
