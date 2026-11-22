"""Actualiza el checklist maestro con la búsqueda de arquitectura de las compactas (fila 38 y casilla B14).

  python scripts/actualizar_checklist_arquitectura.py    # después de los demás actualizar_checklist_*.py

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe; la nota se agrega una sola vez.

La evidencia sale de resultados/busqueda_arquitectura_e2/RESUMEN.json (copia local de lo que deja el cuaderno
colab/Busqueda_Arquitectura_E2.ipynb). Solo marca C si la búsqueda se ejecutó completa: las 1000 arquitecturas
de la fase 1 en las tres familias y la fase 2 con las 5 semillas en todos sus finalistas. Si no, deja P con
el estado real (protocolo y cuaderno listos, o lo que falte) y no afirma una búsqueda realizada.
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ/"docs"/"revision_entregable2"/"CHECKLIST_ENTREGA_E2.md"
RES = RAIZ/"resultados"/"busqueda_arquitectura_e2"
FAMILIAS = ("fcn", "unet", "segnet")
NOMBRE = {"fcn": "FCN", "unet": "U-Net", "segnet": "SegNet"}


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def evidencia():
    """(estado, texto fila 38, texto B14). C solo con la búsqueda completa."""
    base = ("Siete variantes publicadas con arquitectura definida; FCN, U-Net y SegNet compactas (diseño propio) "
            "recorren el espacio de la sección 14: 3 bloques, anchos {16..256}^3, dropout 0,2 en {0,1}^3 = 1000 por "
            "familia. Protocolo a priori del 22/11 (`docs/PROTOCOLO_BUSQUEDA_ARQUITECTURA_E2.md`, "
            "`config/busqueda_arquitectura_e2.json`), solo validación, exploratorio")
    archivo = RES/"RESUMEN.json"
    if not archivo.exists():
        return ("P", base + ". Cuaderno `colab/Busqueda_Arquitectura_E2.ipynb` probado en CPU "
                "(`resultados/busqueda_arquitectura_e2/prueba_cpu.json`); sin ejecutar en GPU. No afirmar búsqueda realizada",
                "ID38; protocolo y cuaderno listos, búsqueda sin ejecutar")
    res = json.loads(archivo.read_text(encoding="utf-8"))
    fid = RES/"fidelidad.json"
    fid = json.loads(fid.read_text(encoding="utf-8")) if fid.exists() else None
    faltas, partes = [], []
    for f in FAMILIAS:
        fam = res["familias"].get(f)
        if not fam or fam["fase1"]["n"] < 1000:
            faltas.append(f"{f}: fase 1 con {fam['fase1']['n'] if fam else 0} de 1000")
            continue
        f2 = fam.get("fase2")
        if not f2 or "ganador" not in f2:
            faltas.append(f"{f}: sin fase 2")
            continue
        incompletos = [t["arq"] for t in f2["tabla"] if t["n_semillas"] < 5]
        if incompletos:
            faltas.append(f"{f}: fase 2 incompleta en {len(incompletos)} arquitecturas")
            continue
        g = f2["ganador"]; c = f2.get("comparacion_heredada") or {}
        rho = f2.get("spearman_fase1_fase2")
        partes.append(f"{NOMBRE[f]}: ganadora {tuple(g['w'])}/{tuple(g['b'])}, Dice val {num(g['media'])} "
                      f"(IC95 {num(g['ic95'][0])}–{num(g['ic95'][1])}); frente a la heredada "
                      f"{num(c.get('diferencia_media_pareada', float('nan')))} ({c.get('lectura', 'sin comparación')})"
                      + (f"; Spearman fase 1-2 {num(rho['rho'], 2)}" if rho else ""))
    if faltas:
        return ("P", base + ". Ejecución parcial: " + "; ".join(faltas) + ". No afirmar búsqueda completa",
                "ID38; búsqueda parcial: " + "; ".join(faltas))
    fidel = ""
    if fid:
        e = fid["elegida"]
        fidel = (f" Fase 1 a fidelidad {e['nombre']} ({e['px']} px, {round(100*e['fraccion_train'])} % de train, "
                 f"6 de 30 épocas){'' if e['nombre'] == 'A' else ', desviación declarada'}.")
    texto = (base + f". Ejecutada: ajuste previo de la tasa con (64,64,64)/(0,1,0); fase 1 con las 1000 × 3; k = {res['k']} "
             "finalistas + referencia + heredada con 5 semillas (fase 2)." + fidel + " " + "; ".join(partes)
             + ". `resultados/busqueda_arquitectura_e2/` (`RESUMEN.json`, `LECTURA.md`)")
    return ("C", texto, "ID38; búsqueda en dos fases ejecutada para las tres compactas")


def main():
    estado, fila38, b14 = evidencia()
    filas = {"38": (estado, fila38)}
    apendice = {"B14": (estado, b14)}
    nota = ("\n**Actualización del 22/11 (arquitectura).** Búsqueda de arquitectura en dos fases para las tres compactas "
            "heredadas (fila 38 y B14): protocolo a priori, cuaderno y, si están, resultados de validación.\n")
    lineas = CHECK.read_text(encoding="utf-8").splitlines()
    pendientes, pend_ap = dict(filas), dict(apendice)
    for k, l in enumerate(lineas):
        celdas = l.split(" | ")
        if l.startswith("| ") and len(celdas) >= 3:
            ident = celdas[0][2:].strip()
            if ident in pendientes and len(celdas) >= 4:
                est, ev = pendientes.pop(ident)
                lineas[k] = f"| {ident} | {celdas[1]} | {est} | {ev} |"
            else:
                clave = ident.split(" ")[0]
                if clave in pend_ap and len(celdas) == 3:
                    est, ev = pend_ap.pop(clave)
                    lineas[k] = f"| {ident} | {est} | {ev} |"
    if pendientes or pend_ap:
        raise SystemExit(f"Filas no encontradas: {sorted(pendientes) + sorted(pend_ap)}")
    texto = "\n".join(lineas) + "\n"
    if nota.strip() not in texto:
        texto = texto.replace("\n## Problema, hipótesis", nota + "\n## Problema, hipótesis", 1)
    CHECK.write_text(texto, encoding="utf-8")
    print(f"Fila 38 y B14 en estado {estado}: {fila38}")


if __name__ == "__main__":
    main()
