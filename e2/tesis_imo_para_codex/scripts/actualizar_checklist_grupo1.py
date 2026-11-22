"""Actualiza el checklist maestro con las filas que estaban desactualizadas al 22/11/2026 (el trabajo ya existía).

  python scripts/actualizar_checklist_grupo1.py    # después de actualizar_checklist_revision.py

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe. No ejecuta análisis: remite a
evidencia ya registrada (filas 37, 41, 57-62, 64; `resultados/integracion_e2/analisis.json`, `error_correlation`:
correlación de Dice por grupo entre redes, mínimo 0,62 TransUNet-SegFormer, máximo 0,94 FCN-U-Net, mediana 0,80).
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"

FILAS = {
    "66": ("C", "Errores compartidos: correlación del Dice por grupo entre las 10 redes de 0,62 (TransUNet-SegFormer) a "
           "0,94 (FCN-U-Net), mediana 0,80 (`resultados/integracion_e2/analisis.json`, `error_correlation`; capítulo "
           "`resultados_e2`). Ensamble: 5 semillas del top 3, +0,021 a +0,028 de Dice con IC sin 0 (ID 62, capítulo "
           "`diagnosticos_e2`). Ensamble entre redes distintas no ejecutado"),
    "70": ("C", "Código de todos los requisitos ejecutados en el repositorio: diagnósticos GPU (`scripts/"
           "diagnosticos_gpu_e2.py`, filas 57-60, 62, 64), complemento de entrenamiento (`scripts/"
           "complemento_entreno_e2.py`, filas 37, 41, 56, 61), revisión humana (filas 12, 63); los cuadernos de Colab "
           "se generan con `scripts/crear_colab_*.py`"),
    "75": ("P", "Guion de 13 diapositivas y respuestas de defensa en `docs/PRESENTACION_E2.md` y material privado de "
           "preguntas; falta el ensayo"),
}
APENDICE = {
    "B11": ("C", "Tablas con IC, pruebas y efectos (ID 48-49); ajuste y sobreajuste: brecha train-val (ID 57), "
            "sobreajuste a un lote (ID 41) y búsqueda equitativa (ID 37)"),
}
NOTA = ("\n**Actualización del 22/11 (cierre).** Filas desactualizadas puestas al día con evidencia ya registrada "
        "(66, 70, 75; B11).\n")


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
