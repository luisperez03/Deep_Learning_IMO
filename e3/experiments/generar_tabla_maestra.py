"""Genera data/tabla_maestra.csv: la tabla unica de la que salen las cifras del dashboard.

Fuente (solo lectura): resultados del E2 (repo Deep_Learning_IMO, e2/tesis_imo_para_codex).
Se indica con --e2 o con la variable de entorno E3_E2_RESULTADOS (carpeta tesis_imo_para_codex).

Entradas usadas:
  resultados/errores_atajos_e2/por_imagen_corridas.csv  Dice, TP, FP, FN por imagen, red, semilla y particion
  resultados/costo_e2/costo_e2.csv                      parametros, GFLOPs y latencia por red y dispositivo
  resultados/integracion_e2/analisis.json               rangos por grupos y por semillas, horas, marca de Pareto
  resultados/baseline_clasico_e2/RESUMEN.json           baseline clasico (referente, fuera de la familia)
  libro/resultados_e2.md                                DE entre semillas e IC95 por grupos publicados en el E2

Definicion de la metrica (igual que el E2): Dice por imagen -> promedio dentro de cada grupo de similitud ->
promedio de grupos con igual peso -> media sobre las cinco semillas. El IC del 95 % remuestrea los 122 grupos de
test con las semillas fijas. Se publican la DE y el IC del informe del E2 (libro/resultados_e2.md); el script los
recalcula (percentiles, 10.000 remuestreos, semilla fija) solo para comprobar que coinciden a +-0,003.
Si la media por red no reproduce el informe del E2 (resultados_e2.md) con 4 decimales, el script falla.

Uso:  python experiments/generar_tabla_maestra.py [--e2 RUTA]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SALIDA = BASE / "data" / "tabla_maestra.csv"
META = BASE / "data" / "tabla_maestra_meta.json"
RNG_SEED = 20261123
B = 10000

# Orden del E2 (por Dice de test descendente) y metadatos de presentacion.
MODELOS = {
    "transunet_2d": ("TransUNet", "Híbrido CNN-Transformer"),
    "fcn": ("FCN compacta", "Convolucional"),
    "unet": ("U-Net compacta", "Convolucional codificador-decodificador"),
    "vmunet": ("VM-UNet", "Espacio de estados (Mamba)"),
    "segnet": ("SegNet compacta", "Convolucional codificador-decodificador"),
    "deeplabv3plus": ("DeepLabv3+ R18", "Convolucional con ASPP"),
    "unetplusplus": ("U-Net++ R18", "Convolucional anidada"),
    "pidnet_s": ("PIDNet-S", "Convolucional de tres ramas"),
    "segnext_t": ("SegNeXt-T", "Convolucional con atención"),
    "segformer_b0": ("SegFormer-B0", "Transformer"),
}
# Dice de test publicado en libro/resultados_e2.md (media sobre semillas), para verificar la reproduccion.
REF_TEST = {"transunet_2d": 0.8626, "fcn": 0.8535, "unet": 0.8464, "vmunet": 0.8391, "segnet": 0.8247,
            "deeplabv3plus": 0.8140, "unetplusplus": 0.8076, "pidnet_s": 0.7936, "segnext_t": 0.7792,
            "segformer_b0": 0.7053}


def leer_informe_e2(p: Path) -> dict:
    """Extrae {modelo: (de, ic_inf, ic_sup)} de la tabla principal de resultados_e2.md."""
    import re
    pat = re.compile(r"^\|\s*(\w+)\s*\|\s*[\d.]+\s*\|\s*[\d.]+ ± ([\d.]+)\s*\|\s*\[([\d.]+), ([\d.]+)\]", re.M)
    out = {m: (float(de), float(lo), float(hi)) for m, de, lo, hi in pat.findall(p.read_text(encoding="utf8"))}
    faltan = set(MODELOS) - set(out)
    if faltan:
        raise SystemExit(f"resultados_e2.md: no se pudo leer {sorted(faltan)}")
    return out


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dice_por_grupo(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz grupos x semillas con el Dice medio por grupo, para una red y una particion."""
    return df.groupby(["group_id", "seed"]).Dice.mean().unstack("seed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", default=os.environ.get("E3_E2_RESULTADOS"))
    a = ap.parse_args()
    if not a.e2:
        print("Falta --e2 o la variable E3_E2_RESULTADOS (carpeta tesis_imo_para_codex del E2).", file=sys.stderr)
        return 2
    r = Path(a.e2) / "resultados"
    f_img = r / "errores_atajos_e2" / "por_imagen_corridas.csv"
    f_cost = r / "costo_e2" / "costo_e2.csv"
    f_ana = r / "integracion_e2" / "analisis.json"
    f_base = r / "baseline_clasico_e2" / "RESUMEN.json"
    f_md = Path(a.e2) / "libro" / "resultados_e2.md"
    pub = leer_informe_e2(f_md)

    img = pd.read_csv(f_img)
    img["iou"] = img.TP / (img.TP + img.FP + img.FN).replace(0, np.nan)
    cost = pd.read_csv(f_cost)
    ana = json.loads(f_ana.read_text(encoding="utf8"))
    base = json.loads(f_base.read_text(encoding="utf8"))
    ana_cost = {c["model"]: c for c in ana["cost"]}

    rng = np.random.default_rng(RNG_SEED)
    filas = []
    for m, (nombre, familia) in MODELOS.items():
        t = img[(img.model == m) & (img.split == "test")]
        v = img[(img.model == m) & (img.split == "val")]
        gt = dice_por_grupo(t)  # grupos x semillas
        gv = dice_por_grupo(v)
        por_semilla = gt.mean(axis=0)
        media = float(por_semilla.mean())
        if abs(media - REF_TEST[m]) > 1.1e-4:  # el informe E2 redondea a 4 decimales
            raise SystemExit(f"{m}: Dice test {media:.4f} no reproduce el E2 ({REF_TEST[m]})")
        medias_grupo = gt.to_numpy().mean(axis=1)
        idx = rng.integers(0, medias_grupo.size, size=(B, medias_grupo.size))
        lo, hi = np.percentile(medias_grupo[idx].mean(axis=1), [2.5, 97.5])
        de_p, lo_p, hi_p = pub[m]
        if abs(lo - lo_p) > 3e-3 or abs(hi - hi_p) > 3e-3:
            raise SystemExit(f"{m}: el IC recalculado [{lo:.4f}, {hi:.4f}] no coincide con el del E2")
        iou_t = t.groupby(["group_id", "seed"]).iou.mean().unstack("seed").mean(axis=0).mean()
        c = cost[cost.modelo == m].set_index(["dispositivo", "hilos"])

        def lat(dev, h):
            try:
                x = c.loc[(dev, h), "latencia_ms_mediana"]
                return float(x) if pd.notna(x) else np.nan
            except KeyError:
                return np.nan

        ac = ana_cost[m]
        filas.append(dict(
            modelo=m, nombre=nombre, familia=familia, tipo="red",
            dice_val=round(float(gv.mean(axis=0).mean()), 4),
            dice_test=round(media, 4), de_semillas=de_p,
            ic95_inf=lo_p, ic95_sup=hi_p,
            iou_test=round(float(iou_t), 4),
            parametros=int(ac["parameters"]),
            gflops=float(c.loc[("cuda", 1), "gflops"]),
            lat_cpu1_ms=lat("cpu", 1), lat_cpu6_ms=lat("cpu", 6), lat_cuda_ms=lat("cuda", 1),
            horas_entrenamiento=round(float(ac["hours_recorded"]), 3),
            rango_grupos=round(float(ac["rank_groups"]), 3), rango_semillas=round(float(ac["rank_seeds"]), 2),
            pareto_e2=bool(ac["pareto_point_estimates"]),
            calibracion_ece="pendiente", n_semillas=int(gt.shape[1]), n_grupos=int(gt.shape[0]),
            estado="medido en E2",
        ))

    filas.append(dict(
        modelo="baseline_clasico", nombre="Baseline clásico (HGB, 63 hojas)", familia="Clásico (píxel a píxel)",
        tipo="baseline", dice_val=round(base["selection"]["validation_means"]["63"], 4),
        dice_test=round(base["test_Dice"], 4), de_semillas=round(base["DE_seeds"], 4),
        ic95_inf=round(base["IC95_groups_fixed_seeds"][0], 4), ic95_sup=round(base["IC95_groups_fixed_seeds"][1], 4),
        calibracion_ece="pendiente", n_semillas=5, n_grupos=122, estado="referente fuera de la familia",
    ))
    filas.append(dict(
        modelo="original", nombre="Modelo original", familia="pendiente", tipo="original",
        calibracion_ece="pendiente", estado="pendiente",
    ))

    cols = ["modelo", "nombre", "familia", "tipo", "dice_val", "dice_test", "de_semillas", "ic95_inf", "ic95_sup",
            "iou_test", "parametros", "gflops", "lat_cpu1_ms", "lat_cpu6_ms", "lat_cuda_ms", "horas_entrenamiento",
            "rango_grupos", "rango_semillas", "pareto_e2", "calibracion_ece", "n_semillas", "n_grupos", "estado"]
    out = pd.DataFrame(filas)[cols]
    SALIDA.parent.mkdir(exist_ok=True)
    out.to_csv(SALIDA, index=False, encoding="utf-8")
    META.write_text(json.dumps({
        "generado": date.today().isoformat(),
        "script": "experiments/generar_tabla_maestra.py",
        "semilla_bootstrap": RNG_SEED, "remuestreos": B,
        "nemenyi_cd": ana["nemenyi_CD"],
        "entradas_sha256": {p.name: sha256(p) for p in (f_img, f_cost, f_ana, f_base, f_md)},
        "tabla_maestra_sha256": sha256(SALIDA),  # lo verifica tests/test_consistencia.py
        "nota": "gflops: contador de operadores de PyTorch (entrada 1x3x256x256). Latencia: mediana en CPU AMD Ryzen 5 "
                "PRO 6650U (1 y 6 hilos) y en GPU; lote 1. VM-UNet no corre en CPU (requiere CUDA y mamba-ssm).",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK {SALIDA} ({len(out)} filas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
