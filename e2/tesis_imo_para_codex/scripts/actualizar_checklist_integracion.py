"""Actualiza el checklist maestro con la integración del 22/11/2026: fila 04 (equivalencia ya ejecutada) y fila 73.

  python scripts/actualizar_checklist_integracion.py    # después de actualizar_checklist_inferencia2.py

Mismo mecanismo que scripts/actualizar_checklist_revision.py. Evidencia: TOST pareado por grupos en
`resultados/inferencia_cierre_e2/` (scripts/inferencia_test_grupos_e2.py) y RQ1-4 con su decisión en
`libro/metodos_e2.md`, `libro/cierre_e2.md`; integrar_entregable2.py ya no reescribe libro/ por defecto (commit ea4ee9a).
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"

FILAS = {
    "04": ("C", "RQ1-4 con H0/H1, α = 0,05, efecto (δ = 0,02 Dice) y regla de decisión en `metodos_e2.md`; temporalidad "
           "explícita (ampliación posterior al test). 22/11: equivalencia ejecutada (TOST pareado por grupos, t y "
           "bootstrap, Holm(9)): TransUNet-FCN indeterminado (p = 0,117), U-Net-FCN equivalentes (p Holm 0,028, "
           "condicional a las semillas); RQ3 respondida: ninguna red equivalente a TransUNet (`cierre_e2.md`, ID 54)"),
    "73": ("C", "20/11: `libro/referencias_e2.bib` con 21 entradas (10 arquitecturas en versión editorial si existe, 7 "
           "de aplicación, 4 de mecanismos), DOI verificados con `verificar_doi_registro.py`. 22/11: "
           "`scripts/integrar_entregable2.py` ya no reescribe `libro/` por defecto (escribe en "
           "`resultados/integracion_e2/libro_generado/`; solo con `--escribir-libro`), riesgo resuelto"),
}
APENDICE = {}
NOTA = ("\n**Actualización del 22/11 (integración).** Equivalencia ejecutada en las RQ (fila 04) y BibTeX protegido "
        "de la reintegración (fila 73).\n")


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
