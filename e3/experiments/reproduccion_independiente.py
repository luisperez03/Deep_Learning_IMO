"""Reproducción independiente de las cifras publicadas del E3 desde un entorno limpio (entregable 13).

  python e3/experiments/reproduccion_independiente.py --e2 e2/tesis_imo_para_codex [--tablero RUTA] [--salida rep.json]

Qué reproduce, sin GPU y sin tocar los archivos del repositorio (todo se regenera en una carpeta temporal):
  1. La tabla maestra del benchmark: corre experiments/generar_tabla_maestra.py contra los resultados por imagen del
     E2 y compara, celda a celda, las filas de las diez redes y del referente con data/tabla_maestra.csv y, si se da
     --tablero, con la del dashboard publicado. El generador ya falla si alguna media no reproduce el informe del E2.
  2. La validación del original: recalcula media y DE por brazo y la diferencia pareada desde
     data/avance_colab_corridas.csv y la compara con data/avance_colab.json.
  3. Los documentos congelados: experiments/verificar_congelamiento.py.
Qué no reproduce: el entrenamiento (requiere GPU; sus identidades y hashes los audita auditar_campana.py) ni el test
del original, que no se ha evaluado. Escribe un reporte JSON y sale con código 1 si algo no coincide.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

E3 = Path(__file__).resolve().parents[1]
CLAVES = ["dice_val", "dice_test", "de_semillas", "ic95_inf", "ic95_sup", "iou_test", "parametros", "gflops",
          "lat_cpu1_ms", "lat_cpu6_ms", "lat_cuda_ms", "horas_entrenamiento", "n_semillas", "n_grupos"]


def comparar_tablas(nueva: pd.DataFrame, ref: pd.DataFrame, nombre: str) -> dict:
    n = nueva[nueva.tipo != "original"].set_index("modelo")
    r = ref[ref.tipo != "original"].set_index("modelo")
    faltan = sorted(set(n.index) ^ set(r.index))
    difs = []
    for m in n.index.intersection(r.index):
        for c in CLAVES:
            a, b = n.at[m, c], r.at[m, c]
            if (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and abs(float(a) - float(b)) <= 5e-5):
                continue
            difs.append({"modelo": m, "columna": c, "regenerada": None if pd.isna(a) else float(a),
                         "publicada": None if pd.isna(b) else float(b)})
    return {"contra": nombre, "filas": int(len(n)), "celdas": int(len(n) * len(CLAVES)), "modelos_distintos": faltan,
            "diferencias": difs, "ok": not faltan and not difs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", required=True, type=Path, help="carpeta tesis_imo_para_codex del E2")
    ap.add_argument("--tablero", type=Path, help="carpeta del repositorio del dashboard (ddcarbono/hazmat13-e3)")
    ap.add_argument("--salida", type=Path)
    a = ap.parse_args()
    rep = {"fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "python": sys.version.split()[0],
           "numpy": np.__version__, "pandas": pd.__version__, "comprobaciones": []}

    # 1. Tabla maestra regenerada en una copia temporal
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "experiments").mkdir()
        (t / "data").mkdir()
        shutil.copy2(E3 / "experiments" / "generar_tabla_maestra.py", t / "experiments")
        r = subprocess.run([sys.executable, str(t / "experiments" / "generar_tabla_maestra.py"), "--e2", str(a.e2.resolve())],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok_gen = r.returncode == 0
        rep["comprobaciones"].append({"nombre": "generar_tabla_maestra.py en entorno limpio", "ok": ok_gen,
                                      "salida": re.sub(r"OK .*?(data[\\/]tabla_maestra\.csv)", r"OK <tmp>/\1",
                                                     r.stdout + r.stderr)[-1500:]})
        if ok_gen:
            nueva = pd.read_csv(t / "data" / "tabla_maestra.csv")
            c = comparar_tablas(nueva, pd.read_csv(E3 / "data" / "tabla_maestra.csv"), "e3/data/tabla_maestra.csv")
            rep["comprobaciones"].append({"nombre": "tabla maestra = repositorio del proyecto", **c})
            if a.tablero:
                c = comparar_tablas(nueva, pd.read_csv(a.tablero / "data" / "tabla_maestra.csv"), "dashboard data/tabla_maestra.csv")
                rep["comprobaciones"].append({"nombre": "tabla maestra = dashboard publicado", **c})

    # 2. Validación del original desde las corridas
    corr = pd.read_csv(E3 / "data" / "avance_colab_corridas.csv")
    av = json.loads((E3 / "data" / "avance_colab.json").read_text(encoding="utf-8"))
    difs = []
    for fila in av["resumen"]:
        x = corr[corr.modelo == fila["modelo"]].dice_val
        for k, v in (("n", len(x)), ("media", x.mean()), ("de", x.std(ddof=1))):
            if abs(float(v) - float(fila[k])) > 1e-9:
                difs.append({"modelo": fila["modelo"], "campo": k, "recalculado": float(v), "publicado": float(fila[k])})
    p = corr[corr.modelo.isin(["unetat_dbnet", "unetat_dbnet_at"])].pivot_table(index="semilla", columns="modelo", values="dice_val")
    d = float((p.unetat_dbnet_at - p.unetat_dbnet).mean())
    if abs(d - av["diferencia_val_at_menos_base"]) > 1e-9:
        difs.append({"campo": "diferencia_val_at_menos_base", "recalculado": d, "publicado": av["diferencia_val_at_menos_base"]})
    rep["comprobaciones"].append({"nombre": "validación del original desde avance_colab_corridas.csv", "ok": not difs,
                                  "diferencia_pareada": d, "diferencias": difs})

    # 3. Documentos congelados
    r = subprocess.run([sys.executable, str(E3 / "experiments" / "verificar_congelamiento.py")], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    rep["comprobaciones"].append({"nombre": "documentos congelados (e3-protocolo-v2)", "ok": r.returncode == 0,
                                  "salida": r.stdout.strip().splitlines()})

    rep["aprobadas"] = sum(c["ok"] for c in rep["comprobaciones"])
    rep["total"] = len(rep["comprobaciones"])
    for c in rep["comprobaciones"]:
        print(("OK   " if c["ok"] else "FALLA"), c["nombre"])
    print(f"{rep['aprobadas']} de {rep['total']}")
    if a.salida:
        a.salida.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if rep["aprobadas"] == rep["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
