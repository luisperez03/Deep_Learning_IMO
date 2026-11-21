"""Actualiza el checklist maestro con el complemento de entrenamiento corrido el 21/11/2026 (dos sesiones L4).

  python scripts/actualizar_checklist_entreno.py      # después de scripts/actualizar_checklist_cierre.py, que pisa estas filas

Mismo mecanismo que scripts/actualizar_checklist_cierre.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe. La nota de actualización se agrega
una sola vez. Evidencia: resultados/complemento_entreno_e2/ y el capítulo `entrenamiento_complementario_e2`.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"
EVID = "`resultados/complemento_entreno_e2/`, capítulo `entrenamiento_complementario_e2`"

FILAS = {
    "37": ("C", "21/11, L4, exploratoria: búsqueda aleatoria T = 4 con el mismo espacio relativo (tasa × 10^u común, "
           "weight decay) y el mismo presupuesto (10 épocas, semilla 11) para las 10, elección por validación, 40 "
           "configuraciones registradas. La receta gana solo en VM-UNet; SegFormer-B0 sube de 0,660 a 0,782; las dos "
           "primeras no cambian. Sin la fase de 5 semillas × 30 épocas (diseñada, ≈ 17 h). " + EVID),
    "41": ("C", "Diez preflight GPU y entrenamiento completo; 21/11, sobreajuste a un lote de 4 imágenes (500 pasos): "
           "9 de 10 pasan (Dice ≥ 0,90); VM-UNet 0,750, falla en la imagen de rótulo más pequeño. " + EVID),
    "46": ("C", "Integridad, partición, adversarial, embeddings y casi duplicados (ID 11) y, 21/11, permutación de "
           "máscaras del pipeline (ID 56): p = 0,005"),
    "56": ("C", "Dice vacío = 0. 21/11: permutación de máscaras en train del pipeline completo, FCN, B = 200: Dice val "
           "real 0,823 frente a nula media 0,465 (máx. 0,592), p = 1/201 = 0,005. La de TransUNet (B = 19) no se "
           "corrió. " + EVID),
    "61": ("C", "21/11: TransUNet, 3 componentes × 5 semillas pareadas, referencia del benchmark: sin saltos −0,092 "
           "[−0,105; −0,079], BCE sola −0,037 [−0,043; −0,031] (p Holm < 0,001); sin Transformer +0,007 [−0,001; 0,015], "
           "p Holm 0,065. " + EVID),
}
APENDICE = {
    "B03": ("C", "ID46: fuga por embeddings y permutación de máscaras (p = 0,005) hechas"),
    "B06": ("C", "Recetas fijas en el benchmark; búsqueda equitativa exploratoria T = 4 (ID37)"),
}
NOTA = ("\n**Actualización del 21/11.** Se corrió el complemento de entrenamiento en L4: sobreajuste, búsqueda T = 4, "
        "permutación de FCN con B = 200 y ablación de TransUNet (filas 37, 41, 46, 56, 61). Exploratorio: el test ya "
        "estaba observado y no se cargó.\n")


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
