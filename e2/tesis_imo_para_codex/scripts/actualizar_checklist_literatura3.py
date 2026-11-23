"""Actualiza el checklist maestro con la reapertura de Q5 y Q6 del 22/11/2026 (filas 20, 21, 24 y 30).

  python scripts/actualizar_checklist_literatura3.py    # después de actualizar_checklist_literatura2.py

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe. La nota se agrega una sola vez y
reemplaza la frase de la cabecera del checklist que declara Q5/Q6 omitidas.
Evidencia ejecutada: 17 exportaciones (sha256 en conciliacion.json y BITACORA_BUSQUEDAS.md), procesar_exportaciones.py,
precribado_reconstruido.py y cribado_nucleo.py (ampliar, registrar, cerrar) con decisiones_cribado_transversal.json.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"

FILAS = {
    "20": ("C", "22/11: Q0–Q6, 9.715 → 6.366 grupos; precribado reconstruido con reglas explícitas; núcleo de dominio "
           "de 25 revisado registro por registro (10 incluidos, 15 excluidos con motivo); censo del bloque Q5/Q6 de 52 "
           "(9 incluidos, 43 excluidos con motivo); muestra estratificada de 200 a ciegas: el filtro pierde 37 % "
           "[26–49] de incluibles en su estrato de exclusión; unos 3.100 incluibles estimados, fuera de la síntesis. "
           "`cribado_nucleo.py`, `PRISMA_FINAL.md`, `CRITERIOS_ELEGIBILIDAD.md`; consistencia 20/20"),
    "21": ("C", "22/11: `procesar_exportaciones.py` sobre las 17 exportaciones (sha256 verificados): 9.715 registros, "
           "6.366 grupos, 3.349 duplicados, 0 coincidencias ambiguas; tres alias de DOI aplicados con los ID absorbidos "
           "conservados; los 6.322 grupos previos a Q5/Q6 mantienen su ID (`conciliacion.json`)"),
    "24": ("C", "22/11: Q5 y Q6 reabiertas por instrucción del usuario y ejecutadas y exportadas en Scopus y WoS con las "
           "ecuaciones del protocolo (Q5 38 + 14, iguales al piloto; Q6 11 + 6); con Q4, tres transversales. Causa del "
           "fallo de exportación del 18/11 identificada (host de descargas no reenviado por el proxy). Censo del bloque: "
           "9 transferencias a visión, 8 de espacio de estados. `BITACORA_BUSQUEDAS.md`, `DECISIONES.md`"),
    "30": ("C", "22/11: gap refutable con denominador de 13 estudios de dominio (10 del cribado y 3 anteriores a 2019); "
           "ninguno segmenta el rótulo por píxel con red ni compara arquitecturas con semillas; Q5/Q6 no añaden "
           "registros de dominio; 2 leídos a texto completo, 8 decididos con resúmenes que declaran el método. "
           "Literatura reproducible por scripts (`cribado_nucleo.py`, `cotejar_fichas.py`, `puntuar_candidatos.py`)"),
}
VIEJA = "Q5/Q6 siguen omitidas por instrucción explícita."
NUEVA = "Q5/Q6 se reabrieron y ejecutaron el 22/11 (fila 24)."
NOTA = ("\n**Actualización del 22/11 (Q5 y Q6).** Búsquedas transversales Q5 y Q6 ejecutadas y exportadas en las dos "
        "bases (fila 24); corpus de 6.366 grupos y censo del bloque transversal (filas 20, 21 y 30).\n")


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
    texto = "\n".join(lineas) + "\n"
    texto = texto.replace(VIEJA, NUEVA)
    texto = texto.replace("; Q5/Q6 siguen omitidas.", "; Q5/Q6 ejecutadas el 22/11.")
    if NOTA.strip() not in texto:
        texto = texto.replace("\n## Problema, hipótesis", NOTA + "\n## Problema, hipótesis", 1)
    CHECK.write_text(texto, encoding="utf-8")
    print(f"{len(FILAS)} filas actualizadas en {CHECK}")


if __name__ == "__main__":
    main()
