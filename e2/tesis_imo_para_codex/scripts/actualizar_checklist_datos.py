"""Actualiza el checklist maestro con la Data Card de HAZMAT-13 (procedencia, licencia, obtención).

  python scripts/actualizar_checklist_datos.py    # después de los demás actualizar_checklist_*.py

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe. La nota de actualización se agrega
una sola vez. Evidencia: docs/DATA_CARD_HAZMAT13.md y scripts/verificar_corpus_hazmat13.py.

La fila 08 queda en P a propósito: la licencia MIT del repositorio nombra «Software» y no se pudo certificar para las
imágenes ni su titularidad; la adquisición no está documentada por los autores.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"

FILAS = {
    "07": ("C", "22/11: Data Card completa, `docs/DATA_CARD_HAZMAT13.md`. Fuente: mrl-amrl/HAZMAT13 (commit `ec85498`, "
           "12/07/2021, sin releases) y artículo doi:10.1016/j.mlwa.2021.100104. Coinciden 1.685 imágenes, 13 clases con "
           "su orden, formato VOC y ruta `/home/amrl/` en los XML; huellas agregadas de imágenes y XML y SHA256 por "
           "archivo (2.902 de 2.902 verificados). Identidad binaria con el Drive de los autores no verificada "
           "(declarado): requiere descargar con permiso. `scripts/verificar_corpus_hazmat13.py`"),
    "08": ("P", "22/11: verificado: licencia MIT del repositorio (copyright MRL-AMRL 2021), que nombra «Software»; "
           "README y artículo no declaran licencia de datos, titularidad de las fotos ni cómo se adquirieron (sin EXIF; "
           "`database = Unknown` en los XML; parte de las muestras viene de situaciones de RoboCup). Geografía no "
           "documentada. Ética y límites declarados en `docs/DATA_CARD_HAZMAT13.md`. Sin redistribución de imágenes. "
           "Sigue P hasta aclaración de los autores (no enviada)"),
    "71": ("C", "22/11: instrucciones de obtención y verificación reproducibles en `docs/DATA_CARD_HAZMAT13.md` §6 "
           "(enlaces, estructura VOC, `scripts/verificar_corpus_hazmat13.py`, particiones con SHA256 en "
           "`resultados/preentrenamiento/*.csv`). La descarga de la carpeta de Drive queda sujeta a autorización; "
           "licencia de imágenes: ver fila 08"),
}
APENDICE = {}
NOTA = ("\n**Actualización del 22/11 (datos).** Data Card de HAZMAT-13 con procedencia, versión, huellas, adquisición, "
        "licencia, ética y obtención (filas 07 y 71); la licencia y adquisición de las imágenes siguen sin certificar "
        "(fila 08).\n")


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
