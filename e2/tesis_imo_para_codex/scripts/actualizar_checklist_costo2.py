"""Actualiza el checklist maestro con el costo computacional (filas 44 y 45).

  python scripts/actualizar_checklist_costo2.py    # despues de los demas actualizar_checklist_*.py

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe. La nota de actualizacion se agrega
una sola vez. Evidencia: resultados/costo_e2/ (medir_costo_e2.py y costo_total_e2.py).
Se ejecuta despues de actualizar_checklist_costo.py (que deja la 45 en C); esta deja la 44 en C con VM-UNet medido en GPU.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"

FILAS = {
    "44": ("C", "22/11: las 10 redes medidas. CPU homogénea (256×256, batch 1, 1 y 6 hilos, 5 de calentamiento y 30 "
           "repeticiones, mediana e IQR) en 9 redes y GPU L4 (misma del entrenamiento) en las 10: parámetros, FLOPs "
           "(FlopCounterMode, cota inferior), memoria y latencia. Los FLOPs no ordenan la latencia ni en CPU ni en GPU "
           "(PIDNet-S y SegNeXt-T: 3,1 G, 11 y 17 ms en L4; U-Net 3,7 G, 1,0 ms); TransUNet 77 G, 23,1 ms; VM-UNet 19,5 G "
           "(cota: no cuenta selective_scan), 32,7 ms, 255 MiB. T4 del test sin medir; CPU de equipo compartido. "
           "`scripts/medir_costo_e2.py`, `colab/Medir_Costo_GPU_E2.ipynb`, `resultados/costo_e2/`"),
}
APENDICE = {}
NOTA = ("\n**Actualización del 22/11 (costo, GPU).** VM-UNet y las demás redes medidas en la L4 (fila 44 cerrada).\n")


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
