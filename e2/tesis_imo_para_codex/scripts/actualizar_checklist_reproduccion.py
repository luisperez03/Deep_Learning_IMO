"""Actualiza el checklist maestro con la reproducción completa en clon limpio del 22/11/2026 (filas 55 y 68; B20, B21).

  python scripts/actualizar_checklist_reproduccion.py    # después de los demás actualizar_checklist_*.py

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe. La nota de actualización se agrega
una sola vez. Evidencia ejecutada: scripts/reproduccion_completa_e2.py, resultados/reproduccion_completa_e2/
(REPORTE.json y LECTURA.md) y docs/REPRODUCCION_EXTREMO_A_EXTREMO.md.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"

FILAS = {
    "55": ("C", "22/11: clon limpio de GitHub (`cierre-e2`, 7b669a7) con VOC2012 y el test cerrado (2.234 archivos, "
           "huella SHA-256): tablas, contrastes y auditoría con discrepancia 0 (`AUDITORIA.json`, `ANALISIS.md` idénticos "
           "byte a byte; `analisis.json` 1,8e-15). Recálculo independiente desde 2.180 máscaras de semilla 11: Dice por "
           "imagen y grupo, discrepancia 0. Reinferencia en GPU (etapa clean): 1,3e-5. Complementos de EDA hasta 0,0045 "
           "por Pillow 13. No reproducidas: dos figuras de curvas (historias solo en Drive). "
           "`scripts/reproduccion_completa_e2.py`, `resultados/reproduccion_completa_e2/`"),
    "68": ("C", "22/11, clon limpio: `jupyter-book build libro -W` sin avisos (18 páginas); cuadernos de EDA "
           "(`libro/eda_original.ipynb`, `notebooks/01_eda_voc.ipynb`) ejecutados 9 de 9 celdas, texto igual al guardado "
           "salvo versiones; los 9 de Colab piden GPU y Drive, clasificados y documentados. 11 de 12 archivos de tests "
           "pasan (`test_e2.py`: faltan `ml-collections` y `einops` en el entorno local). "
           "`resultados/reproduccion_completa_e2/LECTURA.md`"),
}
APENDICE = {
    "B20": ("C", "ID55: clon limpio, discrepancia 0 en test; reinferencia GPU 1,3e-5"),
    "B21": ("C", "Un comando desde clon y datos externos (`reproduccion_completa_e2.py`); entorno registrado; "
            "`docs/REPRODUCCION_EXTREMO_A_EXTREMO.md`"),
}
NOTA = ("\n**Actualización del 22/11 (reproducción).** Reproducción completa en un clon limpio de GitHub con los datos "
        "externos declarados: tablas y contrastes del test con discrepancia 0, recálculo independiente desde las máscaras "
        "y libro compilado sin avisos (filas 55 y 68; B20 y B21).\n")


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
