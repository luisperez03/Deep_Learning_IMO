"""Actualiza las filas 50, 51, 54 y la casilla B15 con la inferencia en test por grupo (22/11/2026).

  python scripts/actualizar_checklist_inferencia2.py    # después de actualizar_checklist_inferencia.py

Mismo mecanismo que scripts/actualizar_checklist_revision.py. Los estados no cambian (C); la evidencia pasa a citar
el test con sus 122 grupos (scripts/inferencia_test_grupos_e2.py, resultados/inferencia_cierre_e2/inferencia_test.json
y LECTURA.md). La fila 53 no cambia: la descomposición de varianza sigue siendo de validación.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"
EV = "`scripts/inferencia_test_grupos_e2.py`, `resultados/inferencia_cierre_e2/`"

FILAS = {
    "50": ("C", "Bootstrap/intercambio, Wilcoxon-Holm, Friedman/Nemenyi en test. 22/11, test y validación: t, "
           "intercambio, Wilcoxon y signo con Holm(9), y max-T por cambio de signo con IC95 simultáneos (respeta la "
           "correlación entre los nueve contrastes). En test max-T rechaza las mismas seis redes que t-Holm "
           "(TransUNet − FCN: p 0,90, IC simultáneo [−0,016; 0,034]). Familia fijada después del test: la desviación "
           f"temporal queda en B12 (D). {EV}"),
    "51": ("C", "22/11, test con 122 grupos y validación: simetría de las diferencias (bootstrap simetrizado, MGG) "
           "rechazada en 6 de 9 contrastes con FCN en test (4 en validación), por cola izquierda; donde se rechaza, "
           "la prueba del signo decide igual que Wilcoxon. Los cuatro procedimientos coinciden en 8 de 9 en test "
           "(U-Net++: la media rechaza, la localización no). Sin un grupo o sin una semilla solo cambian las redes "
           "en el borde sin ajuste (VM-UNet, U-Net); la autocorrelación entre grupos vecinos no es significativa con "
           f"Holm. La independencia entre grupos se comprobó, no se certifica. {EV}"),
    "54": ("C", "22/11, TOST δ = 0,02 en test con 122 grupos, t y bootstrap (mismas decisiones en las 18): "
           "TransUNet − FCN 0,0091, IC90 t [−0,0061; 0,0242], boot [−0,0056; 0,0244], p = 0,12: indeterminado (ni "
           "equivalentes ni superior); U-Net − FCN −0,0072, IC90 [−0,0148; 0,0005], p Holm(9) 0,028 (boot 0,040): "
           "equivalentes; ninguna red equivalente a TransUNet (RQ3). Frágil a las semillas: sin la 53 U-Net − FCN "
           "deja de serlo y sin la 23 TransUNet − FCN lo sería. Validación: mismas conclusiones. "
           f"{EV}"),
}
APENDICE = {
    "B15": ("C", "Checkpoints por val. 22/11: TOST (ID54). Sesgo del ganador: en test TransUNet gana en el 83 % de los "
            "bootstrap de grupos (FCN 16 %), sesgo del nivel 0,0023 (Dice corregido 0,860) y optimismo por mitades "
            "0,005; en validación ≈ 0. Checkpoint por val frente a test: test − val +0,010 (8 de 10 redes suben), sin "
            "optimismo visible; TransUNet − FCN pasa de 0,022 (val) a 0,009 (test)"),
}
NOTA = ("\n**Actualización del 22/11 (inferencia en test por grupo).** Con los 122 grupos del test: TOST con "
        "bootstrap, simetría, max-T, sensibilidad sin grupo o semilla y sesgo del ganador (filas 50, 51, 54 y B15).\n")


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
