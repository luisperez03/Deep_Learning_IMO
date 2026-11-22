"""Actualiza el checklist maestro con el cribado de literatura del 22/11/2026 (filas 20, 21, 27 y 30).

  python scripts/actualizar_checklist_literatura2.py    # después de actualizar_checklist_literatura.py

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe. La nota se agrega una sola vez.
Evidencia ejecutada sobre las 13 exportaciones originales (sha256 iguales a conciliacion.json): procesar_exportaciones.py,
precribado_reconstruido.py y cribado_nucleo.py (preparar, registrar y cerrar); fila 27 con los cuartiles SCImago leídos
a mano el 22/11 y registrados en puntuar_candidatos.py.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"

FILAS = {
    "20": ("C", "22/11: 9.646 → 6.322 grupos; precribado reconstruido con reglas explícitas (el original no conservó "
           "reglas ni decisiones); núcleo de dominio de 25 revisado registro por registro (10 incluidos, 15 excluidos con "
           "motivo) y muestra estratificada de 200 a ciegas: el filtro pierde 37 % [26–49] de incluibles en su estrato "
           "de exclusión; unos 3.100 incluibles estimados, fuera de la síntesis. `cribado_nucleo.py`, `PRISMA_FINAL.md`, "
           "`CRITERIOS_ELEGIBILIDAD.md`; consistencia 20/20 (`auditoria_cribado.json`)"),
    "21": ("C", "22/11: `procesar_exportaciones.py` sobre las 13 exportaciones (sha256 verificados): 9.646 registros, "
           "6.322 grupos, 3.324 duplicados, 0 coincidencias ambiguas; tres alias aplicados con los ID absorbidos "
           "conservados (`conciliacion.json`)"),
    "27": ("C", "22/11: rúbrica de seis componentes aplicada a nueve candidatas con evidencia por nivel; citas OpenAlex "
           "(misma fecha), licencias y pesos por la API de GitHub; rango CORE de las conferencias y cuartil SCImago de "
           "las revistas (IEEE TMI Q1 2020, ACM TOMM Q1 2025); óptimo con cuatro familias = selección operativa; "
           "exclusiones razonadas (`puntuar_candidatos.py`, `PUNTUACION_DEFINITIVA.md`)"),
    "30": ("C", "22/11: gap refutable con denominador de 13 estudios de dominio (10 del cribado y 3 anteriores a 2019): "
           "ninguno segmenta el rótulo por píxel con red ni compara arquitecturas con semillas; 2 leídos a texto "
           "completo, 8 decididos con resúmenes que declaran el método (acceso no disponible, declarado). Literatura "
           "reproducible por scripts (`cribado_nucleo.py`, `cotejar_fichas.py`, `puntuar_candidatos.py`)"),
}
NOTA = ("\n**Actualización del 22/11 (cribado).** Exportaciones originales recuperadas: conciliación definitiva "
        "(6.322 grupos, fila 21), PRISMA del núcleo de dominio con error del precribado medido (fila 20) y gap con "
        "denominador cerrado (fila 30).\n")


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
    if NOTA.strip() not in texto:
        texto = texto.replace("\n## Problema, hipótesis", NOTA + "\n## Problema, hipótesis", 1)
    CHECK.write_text(texto, encoding="utf-8")
    print(f"{len(FILAS)} filas actualizadas en {CHECK}")


if __name__ == "__main__":
    main()
