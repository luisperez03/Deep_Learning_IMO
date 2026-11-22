"""Actualiza el checklist maestro con la revisión humana del 22/11/2026 (ruido de etiqueta y techo).

  python scripts/actualizar_checklist_revision.py    # después de actualizar_checklist_cierre.py y actualizar_checklist_entreno.py

Mismo mecanismo que scripts/actualizar_checklist_cierre.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe. La nota de actualización se agrega
una sola vez. Evidencia: resultados/ruido_etiqueta_e2/, resultados/techo_e2/ y el capítulo `complementos_datos_e2`.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"

FILAS = {
    "12": ("C", "22/11: 100 imágenes de train/val; los 12 casos de la primera pasada confirmados a mano: 10 de 100 con "
           "rótulo cortado por el borde sin máscara (IC95 Wilson 5,5–17,4 %), todos tipo falta; el conjunto a veces "
           "los anota (8,5 % de train con región en el borde): inconsistencia. Acuerdo entre anotadores: ID 63. "
           "`resultados/ruido_etiqueta_e2/`, capítulo `complementos_datos_e2`"),
    "63": ("C", "22/11: segunda anotación independiente de 50 imágenes de validación (una por grupo, por tercil de "
           "área): techo Dice 0,941 (IC95 0,928–0,953). TransUNet 0,866 en las mismas 50; γ 0,31 (val) y 0,27 "
           "(test). `scripts/techo_reanotacion_e2.py medir/gamma`, `scripts/tabla_techo_e2.py`, `resultados/techo_e2/`"),
}
APENDICE = {
    "B13": ("C", "ID63: techo 0,941 por reanotación independiente; γ por red"),
}
NOTA = ("\n**Actualización del 22/11.** Revisión humana: los 12 casos de ruido de etiqueta confirmados a mano "
        "(fila 12) y techo por segunda anotación independiente de 50 imágenes de validación (fila 63).\n")


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
