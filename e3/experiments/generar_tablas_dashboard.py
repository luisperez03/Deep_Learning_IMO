"""Genera las tablas livianas de data/ que usan las paginas EDA y Modelos (todas derivadas del E2).

  data/eda_imagenes.csv      una fila por imagen elegible (1.451): particion, fraccion de area, regiones, luminancia...
  data/dice_por_caso.csv     Dice por imagen (media de las 5 semillas) y red, en validacion y test
  data/dice_por_area.csv     Dice de test por cuartil de area (cortes de train) y red, con IC95 por grupos
  data/limitacion_area.csv   Dice por tercil de area de validacion y diferencia T3-T1 por red (piloto del E2)
  data/calidad_shift.csv     KS por rasgo entre train y validacion (p de Holm)
  data/calidad_resumen.json  conteos de calidad, ruido de etiqueta y shift

No copia imagenes ni mascaras (la licencia de HAZMAT-13 para las imagenes esta sin confirmar).
Los conteos de similitud y cuarentena que el E2 solo publica en prosa (libro/eda.md, docs/DATA_CARD_HAZMAT13.md)
se transcriben aqui como constantes con su fuente; el resto se calcula de los archivos.

Uso:  python experiments/generar_tablas_dashboard.py [--e2 RUTA]   (o variable E3_E2_RESULTADOS)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

# Transcritos de prosa del E2 (no existen en un JSON). Fuente por entrada.
CONSTANTES_PROSA = {
    "imagenes_originales": (1685, "docs/DATA_CARD_HAZMAT13.md, sec. 1"),
    "sin_mascara": (102, "libro/eda.md, 'Calidad de las anotaciones'"),
    "cuarentena": (132, "docs/DATA_CARD_HAZMAT13.md, sec. 1 (huellas): 132 en cuarentena por discrepancias o revisión manual"),
    "pares_similares_candidatos": (748, "libro/eda.md, 'Similitud, procedencia y fugas': dHash y aHash <= 6 bits de 64"),
    "pares_similares_cruzaban_train_val_heredado": (146, "libro/eda.md, ídem"),
    "enlaces_que_cruzan_particiones_actual": (0, "libro/eda.md, ídem: partición por componentes conexas"),
    "discrepancias_clase_xml_mascara": (69, "libro/eda.md, 'Calidad de las anotaciones'"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", default=os.environ.get("E3_E2_RESULTADOS"))
    a = ap.parse_args()
    if not a.e2:
        print("Falta --e2 o la variable E3_E2_RESULTADOS (carpeta tesis_imo_para_codex del E2).", file=sys.stderr)
        return 2
    r = Path(a.e2) / "resultados"
    DATA.mkdir(exist_ok=True)

    # --- EDA por imagen
    attr = pd.read_csv(r / "errores_atajos_e2" / "atributos_por_imagen.csv")
    attr = attr.rename(columns={"split": "particion"})
    attr["area_pct"] = (attr.area_frac * 100).round(4)
    cols = ["id", "group_id", "particion", "area_frac", "area_pct", "regiones", "borde", "lum_rotulo", "lum_anillo",
            "contraste", "clase", "origen"]
    attr[cols].to_csv(DATA / "eda_imagenes.csv", index=False, encoding="utf-8")

    # --- Dice por caso
    por = pd.read_csv(r / "errores_atajos_e2" / "por_imagen_corridas.csv")
    ancho = por.groupby(["split", "id", "model"]).Dice.mean().unstack("model").round(4).reset_index()
    ancho = ancho.rename(columns={"split": "particion"})
    ancho = ancho.merge(attr[["id", "group_id", "area_pct", "clase"]], on="id", how="left")
    ancho.to_csv(DATA / "dice_por_caso.csv", index=False, encoding="utf-8")

    # --- Dice por cuartil de area
    ana = json.loads((r / "integracion_e2" / "analisis.json").read_text(encoding="utf8"))
    est = pd.DataFrame(ana["strata"])
    est[["ic95_inf", "ic95_sup"]] = pd.DataFrame(est.CI95.tolist(), index=est.index)
    est = est.drop(columns="CI95").rename(columns={"model": "modelo", "quartile_train_area": "cuartil",
                                                   "images": "imagenes", "groups": "grupos"})
    est["Dice"] = est.Dice.round(4)
    est[["ic95_inf", "ic95_sup"]] = est[["ic95_inf", "ic95_sup"]].round(4)
    est.to_csv(DATA / "dice_por_area.csv", index=False, encoding="utf-8")
    cortes = [round(x, 5) for x in ana["strata_cutpoints_train"]]

    # --- Limitacion por area (piloto del E2)
    lim = json.loads((r / "limitacion_segformer_area" / "RESUMEN.json").read_text(encoding="utf8"))
    filas = []
    for m, d in lim["primario"]["redes"].items():
        filas.append(dict(modelo=m, dice_t1=d["Dice_T1"], dice_t2=d["Dice_T2"], dice_t3=d["Dice_T3"],
                          delta_t3_t1=d["Delta_T3_menos_T1"], ic95_delta_inf=d["IC95_Delta"][0],
                          ic95_delta_sup=d["IC95_Delta"][1], de_semillas_delta=d["DE_semillas_Delta"]))
    pd.DataFrame(filas).round(4).to_csv(DATA / "limitacion_area.csv", index=False, encoding="utf-8")

    # --- Calidad y shift
    sh = json.loads((r / "shift_train_val_e2" / "RESUMEN.json").read_text(encoding="utf8"))
    ks = pd.DataFrame(sh["ks_por_rasgo"]).round(4)
    ks.to_csv(DATA / "calidad_shift.csv", index=False, encoding="utf-8")
    ru = json.loads((r / "ruido_etiqueta_e2" / "RESUMEN.json").read_text(encoding="utf8"))
    sd = ru["tasa_imagenes_con_error"]["sin_dudosos"]
    split_n = attr.particion.value_counts().to_dict()
    resumen = {
        "generado": date.today().isoformat(),
        "particiones_imagenes": {k: int(split_n[k]) for k in ("train", "val", "test")},
        "grupos_train_val": sh["grupos"],
        "constantes_prosa": {k: {"valor": v, "fuente": f} for k, (v, f) in CONSTANTES_PROSA.items()},
        "ruido_etiqueta": {"muestra": ru["muestra"]["n"], "con_error": sd["k"], "tasa": round(sd["tasa"], 4),
                           "ic95_wilson": [round(x, 4) for x in sd["ic95_wilson"]], "tipo": "rótulo sin máscara (falta)",
                           "fuente": "resultados/ruido_etiqueta_e2/RESUMEN.json; muestra de train y val, test excluido"},
        "shift": {
            "ks_min_p_holm": float(ks.p_holm.min()),
            "mmd_solo_imagen_p": round(sh["mmd"]["solo_imagen"]["p_permutacion"], 4),
            "adversarial_auc_train_val": round(sh["adversarial_train_val"]["auc_crossfit"], 4),
            "adversarial_p": round(sh["adversarial_train_val"]["p_permutacion"], 4),
            "adversarial_auc_train_test": round(sh["adversarial_train_test_existente"]["auc_crossfit"], 4),
            "lectura": sh["lectura"],
        },
        "cortes_area_train_frac": cortes,
    }
    (DATA / "calidad_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK: eda_imagenes.csv, dice_por_caso.csv, dice_por_area.csv, limitacion_area.csv, calidad_shift.csv, "
          "calidad_resumen.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
