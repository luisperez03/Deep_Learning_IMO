"""Actualiza el checklist maestro con la inferencia complementaria del 22/11/2026 (filas 50, 51, 53, 54 y B15).

  python scripts/actualizar_checklist_inferencia.py    # después de los actualizadores anteriores

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe. La nota de actualización se agrega
una sola vez. Evidencia: resultados/inferencia_cierre_e2/ (LECTURA.md, inferencia.json, varianza_mixta.json), de
scripts/inferencia_cierre_e2.py y scripts/varianza_mixta_val_e2.R. Todo exploratorio y posterior al test.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"
EV = "`resultados/inferencia_cierre_e2/`"

FILAS = {
    "50": ("C", "Bootstrap/intercambio, Wilcoxon-Holm, Friedman/Nemenyi en test. 22/11, validación: t, intercambio, "
           "Wilcoxon y signo con Holm(9), y max-T por cambio de signo con IC95 simultáneos (respeta la correlación "
           "entre los nueve contrastes): max-T coincide con t-Holm; Wilcoxon y signo difieren en TransUNet (no "
           "rechazan: ventaja concentrada en pocos grupos) y en DeepLabv3+ y VM-UNet (rechazan: pérdidas pequeñas y "
           "frecuentes). Familia fijada después del test: la desviación temporal queda en B12 (D). "
           f"`scripts/inferencia_cierre_e2.py`, {EV}"),
    "51": ("C", "22/11, validación: simetría de las diferencias por grupo (bootstrap simetrizado, estadístico MGG) "
           "rechazada en 4 de 9 contrastes con FCN; ahí Wilcoxon no es un contraste de media ni de mediana, y la "
           "prueba del signo (sin simetría) da las mismas decisiones. Sensibilidad: t, intercambio, Wilcoxon y signo; "
           "dejar un grupo y una semilla fuera (solo DeepLabv3+ cambia de decisión); autocorrelación entre grupos "
           "vecinos ≈ 0 (p Holm = 1). En test, t, intercambio y Wilcoxon coinciden en 8 de 9. La independencia entre "
           f"grupos sigue siendo aproximada: se comprobó, no se certifica. {EV}"),
    "53": ("C", "22/11, validación (declarada): modelo mixto `lme4` sobre 10.900 Dice imagen x corrida con red fija y "
           "semilla, corrida, grupo, imagen y red x grupo aleatorios. Del total: grupo 64,0 % (IC95 boot "
           "53,0–72,2 %), red x grupo 15,1 %, residuo 15,2 %, red 3,8 %, corrida 0,6 %, semilla común 0,07 % (LRT p = 0,15). "
           "Residuos asimétricos y de colas pesadas: el reparto se mantiene sobre rangos (INT) y sin los 96 Dice = 0. "
           f"`scripts/varianza_mixta_val_e2.R`, {EV}"),
    "54": ("C", "22/11, TOST δ = 0,02: test (t pareado exacto desde media, d_z y n = 122) TransUNet − FCN 0,0091, "
           "IC90 [−0,0061; 0,0242], p = 0,12: no equivalentes (ni superior); U-Net − FCN −0,0072, IC90 [−0,0148; "
           "0,0005], p Holm(9) = 0,028: equivalentes. Validación (t y bootstrap): U-Net ≈ FCN equivalentes; ninguna red "
           "equivalente a TransUNet (RQ3). Sin bootstrap en test: faltan los grupos fuera del repositorio. "
           f"`scripts/inferencia_cierre_e2.py`, {EV}"),
}
APENDICE = {
    "B15": ("C", "Checkpoints por val. 22/11: TOST (ID54); sesgo del ganador en validación: TransUNet gana en el 99,9 % "
            "de los bootstrap, sesgo del nivel 0,0001 y optimismo por mitades ≈ 0; checkpoint por val frente a test: "
            "test − val +0,010 (8 de 10 redes suben), sin optimismo visible; TransUNet − FCN pasa de 0,022 (val) a "
            "0,009 (test)"),
}
NOTA = ("\n**Actualización del 22/11 (inferencia).** Equivalencia TOST en test y validación, descomposición de "
        "varianza con modelo mixto en validación, simetría y sensibilidad, max-T simultáneo y sesgo del ganador "
        "(filas 50, 51, 53, 54 y B15). Exploratorio, posterior al test.\n")


def main():
    lineas = CHECK.read_text(encoding="utf-8").splitlines()
    pendientes, pend_ap = dict(FILAS), dict(APENDICE)
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
    if NOTA.strip() not in texto:
        texto = texto.replace("\n## Problema, hipótesis", NOTA + "\n## Problema, hipótesis", 1)
    CHECK.write_text(texto, encoding="utf-8")
    print(f"{len(FILAS)} filas y {len(APENDICE)} casillas actualizadas en {CHECK}")


if __name__ == "__main__":
    main()
