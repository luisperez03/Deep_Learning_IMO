"""Cierra la fila 08 del checklist maestro (licencia de HAZMAT-13) como limitación declarada, 24/11/2026.

  python scripts/actualizar_checklist_licencia.py

Mismo mecanismo que scripts/actualizar_checklist_literatura3.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si la fila no existe.
Decisión: no se consulta a los autores; la falta de licencia de datos queda como limitación (D), igual que las
filas 05, 06 y 47. Los hechos verificados el 22/11 se conservan en la evidencia.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"

FILAS = {
    "08": ("D", "24/11: limitación declarada, sin consulta a los autores. Verificado el 22/11: licencia MIT del "
           "repositorio (copyright MRL-AMRL 2021), que nombra «Software»; README y artículo no declaran licencia de "
           "datos, titularidad de las fotos ni cómo se adquirieron (sin EXIF; `database = Unknown` en los XML; parte "
           "de las muestras viene de situaciones de RoboCup). Geografía no documentada. Ética y límites en "
           "`docs/DATA_CARD_HAZMAT13.md`; el libro lo declara (`problema.md`, `implementacion.md`, "
           "`reproducibilidad_e2.md`). Sin redistribución de imágenes ni pesos"),
}


def main():
    lineas = CHECK.read_text(encoding="utf-8").splitlines()
    pendientes = dict(FILAS)
    for k, l in enumerate(lineas):
        celdas = l.split(" | ")
        if l.startswith("| ") and len(celdas) >= 4:
            ident = celdas[0][2:].strip()
            if ident in pendientes:
                est, ev = pendientes.pop(ident)
                lineas[k] = f"| {ident} | {celdas[1]} | {est} | {ev} |"
    if pendientes:
        raise SystemExit(f"Filas no encontradas: {sorted(pendientes)}")
    CHECK.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"{len(FILAS)} filas actualizadas en {CHECK}")


if __name__ == "__main__":
    main()
