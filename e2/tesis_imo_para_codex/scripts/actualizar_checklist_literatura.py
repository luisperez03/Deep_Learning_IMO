"""Actualiza el checklist maestro con el cierre de literatura del 22/11/2026 (filas 20, 21, 22, 24, 27 y 30).

  python scripts/actualizar_checklist_literatura.py    # después de los demás actualizar_checklist_*.py

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe. La nota se agrega una sola vez.
C solo con evidencia ejecutada: fichas cotejadas (22). Las filas que dependen de las exportaciones originales de
Scopus/WoS (20, 21, 30) quedan en P con el script de cierre probado; 27 queda en P por dos cuartiles de revista.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"

FILAS = {
    "20": ("P", "22/11: alcance fijado (texto completo del núcleo de dominio + muestra estratificada de 200 del "
           "precribado con IC95 de Wilson); `cribado_nucleo.py preparar/cerrar` probado con datos sintéticos. Sin "
           "correr: faltan las exportaciones originales y las decisiones por registro del precribado"),
    "21": ("P", "Pares adjudicados (Crossref y doi.org) y alias en `procesar_exportaciones.py`. Las 13 exportaciones "
           "no están en este equipo (búsqueda por nombre y sha256 el 22/11); 6.322 sigue siendo esperado, no ejecutado"),
    "22": ("C", "22/11: nueve fichas de 13 campos (siete candidatas y dos alternativas) con página y fragmento literal "
           "del PDF; `cotejar_fichas.py` confirma 229 de 229 (`cotejo_fichas.json`, `FICHAS_COTEJADAS.md`). Calidad: "
           "repeticiones, selección por test y dificultad compartida, en el capítulo de literatura"),
    "24": ("P", "Q5 sin exportación (fallo de sesión institucional el 18/11; conteos piloto 38/14) y Q6 sin ejecutar; "
           "no se reabren. Justificación y riesgo de omisión en el capítulo de literatura; requisito no cumplido"),
    "27": ("P", "22/11: rúbrica de seis componentes aplicada a nueve candidatas con evidencia por nivel; citas OpenAlex "
           "(misma fecha), licencias y pesos por la API de GitHub, rango CORE de las conferencias; óptimo con cuatro "
           "familias = selección operativa (`puntuar_candidatos.py`, `PUNTUACION_DEFINITIVA.md`). Falta el cuartil "
           "de IEEE TMI y ACM TOMM (SCImago no automatizable)"),
    "30": ("P", "Scripts de literatura versionados y reproducibles (cotejo, bibliometría, puntuación). Gap refutable "
           "redactado; su denominador sale de `cribado_nucleo.py cerrar`, pendiente de las exportaciones"),
}
NOTA = ("\n**Actualización del 22/11 (literatura).** Fichas de extracción cotejadas con el PDF (fila 22), puntuación "
        "definitiva con citación (27), justificación de Q5/Q6 (24) y cribado del núcleo preparado (20, 21, 30), que "
        "espera las exportaciones originales de Scopus/WoS.\n")


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
