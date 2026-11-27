"""Lectura de la búsqueda de arquitectura de las compactas (sección 14), a partir de lo que dejó el cuaderno.

  python scripts/lectura_busqueda_arquitectura_e2.py                       # tablas y LECTURA.md
  <python con matplotlib> scripts/lectura_busqueda_arquitectura_e2.py --figura   # además la figura del libro

Entrada: resultados/busqueda_arquitectura_e2/ (RESUMEN.json, fase1.csv y los result.json de cada corrida), traídos
con scripts/traer_resultados_busqueda_arquitectura_e2.py. Salida en la misma carpeta: LECTURA.md, TABLA_FASE2.md,
TABLA_RESUMEN.md y COSTO.json; con --figura, libro/figuras/e2/busqueda_arquitectura.png.
Solo validación: el test no se carga. Las reglas de lectura son las del protocolo del 22/11
(docs/PROTOCOLO_BUSQUEDA_ARQUITECTURA_E2.md, secciones 7 a 9); aquí solo se aplican y se redactan.
El Dice de validación del benchmark (las cinco corridas originales) sale de libro/resultados_e2.md y es solo
descriptivo, porque el ajuste previo cambió la tasa en U-Net y SegNet.
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados" / "busqueda_arquitectura_e2"
FIG = RAIZ / "libro" / "figuras" / "e2" / "busqueda_arquitectura.png"
FAMILIAS = ("fcn", "unet", "segnet")
NOMBRE = {"fcn": "FCN", "unet": "U-Net", "segnet": "SegNet"}
DELTA = 0.02
# Dice de validación de las cinco corridas originales del benchmark (libro/resultados_e2.md, tabla principal).
BENCHMARK_VAL = {"fcn": 0.8309, "unet": 0.8320, "segnet": 0.8063}


def n(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def p_texto(x):
    return "< 0,001" if x < 0.001 else "= " + n(x, 3)


def arq(t):
    return f"({', '.join(map(str, t['w']))})/({', '.join(map(str, t['b']))})"


def miles(x):
    return f"{x:,}".replace(",", ".")


def costo():
    horas = defaultdict(float)
    corridas = defaultdict(int)
    for p in glob.glob(str(RES / "*" / "*" / "result.json")):
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        horas[r["bloque"]] += r.get("seconds_total", 0) / 3600
        corridas[r["bloque"]] += 1
    out = {b: {"corridas": corridas[b], "horas_l4": round(horas[b], 1)} for b in sorted(horas)}
    out["total_horas_l4"] = round(sum(horas.values()), 1)
    return out


def tabla_fase2(res):
    filas = ["| Familia | Arquitectura (anchos)/(dropout) | Papel | Parámetros | Dice val fase 2 | IC95 t | DE semillas | "
             "Dice fase 1 | Brecha train−val |", "|---|---|---|---:|---:|---|---:|---:|---:|"]
    for f in FAMILIAS:
        f2 = res["familias"][f]["fase2"]
        gan, mej = f2["ganador"]["arq"], f2["mejor_media"]
        for t in f2["tabla"]:
            marca = " **(ganadora)**" if t["arq"] == gan else (" (mayor media)" if t["arq"] == mej else "")
            filas.append(f"| {NOMBRE[f]} | `{t['arq']}` {arq(t)}{marca} | {t['papel']} | {miles(t['parametros'])} | "
                         f"{n(t['media'], 4)} | [{n(t['ic95'][0])}; {n(t['ic95'][1])}] | {n(t['de'], 4)} | "
                         f"{n(t['dice_fase1'])} | {n(t['brecha_train_val_media'])} |")
    return "\n".join(filas) + "\n"


def tabla_resumen(res):
    filas = ["| | FCN | U-Net | SegNet |", "|---|---|---|---|"]
    fam = {f: res["familias"][f] for f in FAMILIAS}

    def fila(nombre, fn):
        filas.append(f"| {nombre} | " + " | ".join(fn(f, fam[f]) for f in FAMILIAS) + " |")

    def por_arq(f2, a):
        return next(t for t in f2["tabla"] if t["arq"] == a)

    fila("Tasa del ajuste previo", lambda f, x: str(res["ajuste_previo"]["tasa_elegida"][f]).replace(".", ",")
         if "tasa_elegida" in res["ajuste_previo"] else "")
    fila("Fase 1: media (DE entre arquitecturas)", lambda f, x: f"{n(x['fase1']['media'])} ({n(x['fase1']['de_entre_arquitecturas'])})")
    fila("σ semilla / σ arquitectura (r)", lambda f, x: n(x["finalistas"]["r"], 2))
    fila("Mayor media en fase 2", lambda f, x: (lambda t: f"`{t['arq']}` {n(t['media'])}")(por_arq(x["fase2"], x["fase2"]["mejor_media"])))
    fila("Ganadora por la regla (menos parámetros en el empate)", lambda f, x: f"`{x['fase2']['ganador']['arq']}` {n(x['fase2']['ganador']['media'])}")
    fila("Mayor media − ganadora", lambda f, x: n(por_arq(x["fase2"], x["fase2"]["mejor_media"])["media"] - x["fase2"]["ganador"]["media"]))
    fila("Empate técnico / equivalentes TOST", lambda f, x: f"{len(x['fase2']['empate_tecnico'])} / {x['fase2']['tamano_conjunto_equivalente']}")
    fila("Ganadora − heredada (IC95 t)", lambda f, x: (lambda c: f"{n(c['diferencia_media_pareada'])} [{n(c['ic95_t'][0])}; {n(c['ic95_t'][1])}]")(x["fase2"]["comparacion_heredada"]))
    fila("Ganadora − heredada (IC95 bootstrap por grupos)", lambda f, x: (lambda c: f"[{n(c['bootstrap_grupos']['ic95'][0])}; {n(c['bootstrap_grupos']['ic95'][1])}]")(x["fase2"]["comparacion_heredada"]))
    fila("Lectura fijada antes", lambda f, x: x["fase2"]["comparacion_heredada"]["lectura"])
    fila("Heredada en fase 2 / benchmark (val)", lambda f, x: f"{n(por_arq(x['fase2'], 'a0056')['media'])} / {n(BENCHMARK_VAL[f])}")
    fila("Spearman fase 1 − fase 2 (10 finalistas)", lambda f, x: f"{n(x['fase2']['spearman_fase1_fase2']['rho'], 2)} (p {p_texto(x['fase2']['spearman_fase1_fase2']['p'])})")
    fila("Rango de la referencia en fase 1", lambda f, x: str(x["finalistas"]["rango_referencia"]))
    fila("DE arquitecturas / DE semilla", lambda f, x: n(x["sensibilidad"]["cociente"], 2))
    fila("R² del modelo de anchos y dropout", lambda f, x: n(x["fase1"]["modelo_lineal"]["r2"], 2))
    fila("Sin dropout − con dropout (Wilcoxon p)", lambda f, x: f"{n(x['fase1']['wilcoxon_dropout']['mediana_sin_menos_con'], 4)} (p {p_texto(x['fase1']['wilcoxon_dropout']['p'])})")
    return "\n".join(filas) + "\n"


def lectura(res, cst):
    fam = {f: res["familias"][f] for f in FAMILIAS}
    gan = {f: fam[f]["fase2"]["ganador"] for f in FAMILIAS}
    mej = {f: next(t for t in fam[f]["fase2"]["tabla"] if t["arq"] == fam[f]["fase2"]["mejor_media"]) for f in FAMILIAS}
    her = {f: next(t for t in fam[f]["fase2"]["tabla"] if t["arq"] == "a0056") for f in FAMILIAS}
    com = {f: fam[f]["fase2"]["comparacion_heredada"] for f in FAMILIAS}
    sp = {f: fam[f]["fase2"]["spearman_fase1_fase2"] for f in FAMILIAS}
    factores = {f: {x["factor"]: x for x in fam[f]["fase1"]["modelo_lineal"]["factores"]} for f in FAMILIAS}
    unicas = {gan[f]["arq"] for f in FAMILIAS}
    L = ["# Búsqueda de arquitectura de las compactas: lectura", "",
         "Generado por `scripts/lectura_busqueda_arquitectura_e2.py` desde `RESUMEN.json`, `fase1.csv` y los `result.json`. "
         "Solo validación; el test no se cargó (`test_evaluado = false`). Exploratorio: no cambia la selección ni las "
         "conclusiones del benchmark cerrado. Reglas de lectura fijadas antes en `docs/PROTOCOLO_BUSQUEDA_ARQUITECTURA_E2.md`.", "",
         "## Ejecución", "",
         f"- Fase 1 completa: 1.000 arquitecturas por familia, más 40 repeticiones con otra semilla para σ_semilla "
         f"(3.120 corridas), a la fidelidad B de `fidelidad.json` (128 px, todo train, 6 épocas).",
         f"- k = {res['k']} (r = σ_semilla/σ_arquitectura ≤ 1 en las tres: "
         + ", ".join(f"{NOMBRE[f]} {n(fam[f]['finalistas']['r'], 2)}" for f in FAMILIAS)
         + "). Fase 2: 12 arquitecturas por familia (10 finalistas, la referencia (64, 64, 64)/(0, 1, 0) y la heredada "
         "(16, 32, 64)/(0, 0, 0)) con las semillas 11, 23, 37, 53 y 71, a 256 px y 30 épocas con paciencia 5: 180 corridas, "
         "ninguna abierta.",
         f"- Cómputo registrado, suma de los segundos de cada corrida en L4: " + ", ".join(
             f"{b.replace('fase', 'fase ')} {n(v['horas_l4'], 1)} h ({miles(v['corridas'])} corridas)" for b, v in cst.items() if b != "total_horas_l4")
         + f"; total {n(cst['total_horas_l4'], 1)} h. Corrían cuatro procesos a la vez en una sola L4, así que el tiempo de "
         f"reloj fue cerca de la cuarta parte (unas {round(cst['total_horas_l4'] / 4)} h).", "",
         "## 1. Ganadora por la regla fijada antes", ""]
    if len(unicas) == 1:
        L.append(f"La regla elige la misma arquitectura en las tres familias: `{gan['fcn']['arq']}` "
                 f"{arq(gan['fcn'])}, que es la referencia. No es la de mayor media en ninguna. Ocurre porque el empate técnico "
                 "(diferencia con la mejor menor que δ = 0,02 o intervalos al 95 % solapados) abarca a los 11 finalistas no "
                 "heredados de cada familia: con cinco semillas los intervalos miden de 0,01 a 0,05 y todas las medias caen a "
                 "menos de δ de la mejor. Dentro del empate gana la de menos parámetros, y la referencia es la más pequeña "
                 "de ese grupo. La heredada, más pequeña aún, queda fuera del empate en las tres.")
    L.append("")
    for f in FAMILIAS:
        L.append(f"- {NOMBRE[f]}: ganadora {n(gan[f]['media'], 4)} con {miles(gan[f]['parametros'])} parámetros; mayor media "
                 f"`{mej[f]['arq']}` {arq(mej[f])} {n(mej[f]['media'], 4)} con {miles(mej[f]['parametros'])} "
                 f"({n(mej[f]['parametros'] / gan[f]['parametros'], 1)} veces más). Diferencia "
                 f"{n(mej[f]['media'] - gan[f]['media'], 4)}, menor que δ. Equivalentes a la mejor por TOST: "
                 f"{fam[f]['fase2']['tamano_conjunto_equivalente']} de 12"
                 + (" (incluida la ganadora)." if gan[f]["arq"] in fam[f]["fase2"]["conjunto_equivalente_tost"] else " (la ganadora no está entre ellas)."))
    fuera_tost = [NOMBRE[f] for f in FAMILIAS if gan[f]["arq"] not in fam[f]["fase2"]["conjunto_equivalente_tost"]]
    if fuera_tost:
        L += ["", f"En {' y '.join(fuera_tost)} la regla elige una arquitectura que el TOST no certifica equivalente a la "
              "mejor: el criterio de empate (δ o intervalos solapados) es más laxo que la equivalencia. La regla se aplica "
              "tal como se fijó y la diferencia se declara."]
    borde = [NOMBRE[f] for f in FAMILIAS if any(fam[f]["fase2"]["borde_del_espacio"][k] for k in ("algun_ancho_256", "algun_ancho_16"))]
    con256 = [NOMBRE[f] for f in FAMILIAS if 256 in mej[f]["w"]]
    L += ["", ("Ninguna ganadora tiene un ancho en el borde del espacio (16 o 256): el óptimo por la regla no está recortado "
               "por el espacio." if not borde else f"Ganadora con un ancho en el borde en {', '.join(borde)}: limitación del espacio.")
          + (f" La de mayor media sí usa 256 en algún bloque en {', '.join(con256)}." if con256 else ""), "",
          "## 2. Frente a la compacta heredada", "",
          "Ganadora menos heredada, las dos en la fase 2 con la misma receta y las mismas semillas (diferencia pareada por "
          "semilla, IC95 t; bootstrap por los 122 grupos de validación, 2.000 remuestreos; TOST con ±δ):", ""]
    for f in FAMILIAS:
        c = com[f]
        L.append(f"- {NOMBRE[f]}: {n(c['diferencia_media_pareada'], 4)}, IC95 t [{n(c['ic95_t'][0], 4)}; {n(c['ic95_t'][1], 4)}], "
                 f"bootstrap [{n(c['bootstrap_grupos']['ic95'][0], 4)}; {n(c['bootstrap_grupos']['ic95'][1], 4)}]; "
                 f"TOST p = {n(c['tost_delta']['p'], 3)}. Lectura: {c['lectura']}.")
    L += ["", "Según la lectura fijada antes: en U-Net y SegNet la ganadora supera a la heredada por δ o más, así que el "
          "benchmark subestimó esas dos familias en validación. En FCN la mejora es real (el intervalo excluye el cero) pero "
          "su estimación queda justo por debajo de δ y el TOST no acredita equivalencia: ni subestimación declarada ni "
          "herencia justificada. En ninguna familia la heredada cae en el conjunto equivalente a la mejor.", "",
          "Cuánto de esa mejora es tasa y cuánto arquitectura. El ajuste previo cambió la tasa de 0,001 a 0,0003 en U-Net y "
          "SegNet; en FCN la dejó igual. Comparación descriptiva con las cinco corridas originales del benchmark (validación):", ""]
    for f in FAMILIAS:
        L.append(f"- {NOMBRE[f]}: benchmark {n(BENCHMARK_VAL[f], 4)}; heredada en la fase 2 {n(her[f]['media'], 4)} "
                 f"({'+' if her[f]['media'] >= BENCHMARK_VAL[f] else ''}{n(her[f]['media'] - BENCHMARK_VAL[f], 4)}, efecto de la tasa); "
                 f"ganadora {n(gan[f]['media'], 4)} ({'+' if gan[f]['media'] >= her[f]['media'] else ''}{n(gan[f]['media'] - her[f]['media'], 4)} "
                 f"sobre la heredada, efecto de la arquitectura).")
    L += ["", f"En FCN, con la misma tasa y las mismas semillas, la heredada reproduce el Dice de validación del benchmark "
          f"a cuatro decimales ({n(her['fcn']['media'], 4)} frente a {n(BENCHMARK_VAL['fcn'], 4)}): el entrenamiento de la "
          "búsqueda reproduce el del benchmark, y en las otras dos familias la diferencia con el benchmark se explica por la tasa, no por un cambio de código.", "",
          "## 3. La fase 1 no ordenó a los finalistas", "",
          "Correlación de Spearman entre el Dice de la fase 1 y la media de la fase 2 en los 10 finalistas: "
          + "; ".join(f"{NOMBRE[f]} ρ = {n(sp[f]['rho'], 2)} (p {p_texto(sp[f]['p'])})" for f in FAMILIAS) + ". "
          "Entre los 10 mejores de la fase 1 el rango está restringido y se espera una correlación baja, pero en U-Net "
          "el orden se invierte casi por completo. La referencia estaba en los puestos "
          + ", ".join(f"{fam[f]['finalistas']['rango_referencia']} ({NOMBRE[f]})" for f in FAMILIAS)
          + " de 1.000 en la fase 1 y en la fase 2 queda a menos de δ de la mejor. Una explicación posible es que la "
          "fidelidad reducida de la fase 1 (128 px y 6 épocas) premie a las redes que aprenden rápido a baja resolución, "
          "que no son necesariamente las mejores a 256 px y 30 épocas. La fase 1 sirvió de cribado, como estaba previsto, pero no permite afirmar que los finalistas sean "
          "las mejores de las 1.000 a fidelidad completa. Es una limitación de la búsqueda.", "",
          "## 4. El espacio (fase 1)", ""]
    for f in FAMILIAS:
        fa = factores[f]
        L.append(f"- {NOMBRE[f]}: anchos y dropout explican R² = {n(fam[f]['fase1']['modelo_lineal']['r2'], 2)} de la varianza. "
                 f"Proporción por ancho: w1 {n(fa['w1']['proporcion_varianza'], 2)}, w2 {n(fa['w2']['proporcion_varianza'], 2)}, "
                 f"w3 {n(fa['w3']['proporcion_varianza'], 2)}; el dropout suma menos de 0,02. "
                 f"Sin dropout rinde {n(fam[f]['fase1']['wilcoxon_dropout']['mediana_sin_menos_con'], 4)} más que con dropout "
                 f"(mediana por bloque de anchos; Wilcoxon p {p_texto(fam[f]['fase1']['wilcoxon_dropout']['p'])}, "
                 f"r biserial {n(fam[f]['fase1']['wilcoxon_dropout']['rango_biserial'], 2)}).")
    cuad = [(NOMBRE[f], t["termino"].split("_")[1][:2], t["estimado"]) for f in FAMILIAS
            for t in fam[f]["fase1"]["tendencia_log2_ancho"] if "^2" in t["termino"] and t["p"] < 0.05]
    signo = ("todos negativos: rendimientos decrecientes al ensanchar" if all(e < 0 for _, _, e in cuad)
             else "de signo mixto")
    L += ["", "El ancho del primer bloque, el que trabaja a resolución completa, es el factor que más pesa en U-Net y SegNet; "
          "en FCN pesan parecido el primero y el último. Términos cuadráticos en log2 del ancho con p < 0,05: "
          + ", ".join(f"{a} {w}" for a, w, _ in cuad) + f", {signo}. El efecto del dropout es pequeño, del orden de milésimas, y a 6 "
          "épocas puede deberse a que el dropout frena el aprendizaje temprano más que a una peor solución final.", "",
          "Sensibilidad (DE entre arquitecturas en la fase 1 frente a la DE por semilla en la fase 2): "
          + ", ".join(f"{NOMBRE[f]} {n(fam[f]['sensibilidad']['cociente'], 2)}" for f in FAMILIAS)
          + ". La arquitectura importa en el espacio completo, más en U-Net y SegNet que en FCN. Entre los finalistas, en "
          "cambio, las diferencias quedan por debajo de δ.", "",
          "Brecha de Dice train−validación en la fase 2: de "
          + n(min(t["brecha_train_val_media"] for f in FAMILIAS for t in fam[f]["fase2"]["tabla"]), 3) + " a "
          + n(max(t["brecha_train_val_media"] for f in FAMILIAS for t in fam[f]["fase2"]["tabla"]), 3)
          + ("; la menor de cada familia es la de la heredada, la red más pequeña." if all(
              min(fam[f]["fase2"]["tabla"], key=lambda t: t["brecha_train_val_media"])["arq"] == "a0056" for f in FAMILIAS)
             else "."), "",
          "## 5. Qué cambia y qué no", "",
          "- Las conclusiones del benchmark sobre test no cambian: el test no se cargó y la comparación de las diez redes "
          "se hizo con las compactas heredadas.",
          "- Las compactas del benchmark no eran óptimas en su espacio: con la arquitectura de la regla y la tasa del ajuste "
          "previo ganan en validación "
          + ", ".join(f"{n(gan[f]['media'] - BENCHMARK_VAL[f], 3)} ({NOMBRE[f]})" for f in FAMILIAS)
          + " sobre sus cinco corridas originales. Eso no se traslada al test sin evaluarlo, y el test ya fue observado.",
          "- No hay confirmación en test de la ganadora (paso A6 de la guía) ni validación anidada; los dos puntos estaban "
          "declarados como limitación en el protocolo.", ""]
    return "\n".join(L)


def figura(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colores = {"finalista": "#4C72B0", "referencia": "#DD8452", "heredada": "#55A868"}
    fig, ejes = plt.subplots(2, 3, figsize=(13, 7.5))
    for j, f in enumerate(FAMILIAS):
        f2 = res["familias"][f]["fase2"]
        for t in f2["tabla"]:
            c = colores[t["papel"]]
            err = [[t["media"] - t["ic95"][0]], [t["ic95"][1] - t["media"]]]
            ejes[0, j].errorbar(t["dice_fase1"], t["media"], yerr=err, fmt="o", color=c, capsize=2, ms=5)
            ejes[1, j].errorbar(t["parametros"], t["media"], yerr=err, fmt="o", color=c, capsize=2, ms=5)
        s = f2["spearman_fase1_fase2"]
        ejes[0, j].set_title(f"{NOMBRE[f]}: Spearman finalistas ρ = {s['rho']:.2f}".replace(".", ","))
        ejes[0, j].set_xlabel("Dice de validación, fase 1 (128 px, 6 épocas)")
        ejes[1, j].set_xscale("log")
        ejes[1, j].set_xlabel("Parámetros (escala log)")
        mejor = max(t["media"] for t in f2["tabla"])
        for k in (0, 1):
            ejes[k, j].axhline(mejor - DELTA, color="grey", ls=":", lw=1)
            ejes[k, j].set_ylabel("Dice val. fase 2 (media, IC95)" if j == 0 else "")
    for p, c in colores.items():
        ejes[0, 0].plot([], [], "o", color=c, label=p)
    ejes[0, 0].legend(loc="lower right", fontsize=8)
    fig.text(0.5, 0.005, "Línea punteada: mejor media de la fase 2 menos δ = 0,02. Solo validación.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(FIG)


def main():
    res = json.loads((RES / "RESUMEN.json").read_text(encoding="utf-8"))
    if res.get("test_evaluado"):
        raise SystemExit("RESUMEN.json dice que el test se evaluó: no se redacta la lectura")
    aj = json.loads((RES / "ajuste_previo.json").read_text(encoding="utf-8"))
    res["ajuste_previo"]["tasa_elegida"] = aj["lr_elegida"]
    cst = costo()
    (RES / "COSTO.json").write_text(json.dumps(cst, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (RES / "TABLA_FASE2.md").write_text("# Fase 2 por familia\n\n" + tabla_fase2(res), encoding="utf-8")
    (RES / "TABLA_RESUMEN.md").write_text("# Resumen por familia\n\n" + tabla_resumen(res), encoding="utf-8")
    (RES / "LECTURA.md").write_text(lectura(res, cst), encoding="utf-8")
    print("LECTURA.md, TABLA_FASE2.md, TABLA_RESUMEN.md y COSTO.json en", RES)
    if "--figura" in sys.argv:
        figura(res)


if __name__ == "__main__":
    main()
