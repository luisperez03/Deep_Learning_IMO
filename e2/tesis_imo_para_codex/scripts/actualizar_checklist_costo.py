"""Actualiza el checklist maestro con el costo computacional (filas 44 y 45).

  python scripts/actualizar_checklist_costo.py    # despues de los demas actualizar_checklist_*.py

Mismo mecanismo que scripts/actualizar_checklist_revision.py: reemplaza estado y evidencia por ID en
docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md y falla si una fila no existe. La nota de actualizacion se agrega
una sola vez. Evidencia: resultados/costo_e2/ (medir_costo_e2.py y costo_total_e2.py).
Estados propuestos: 44 queda en P (VM-UNet sin medir: exige CUDA; pasa a C con COLAB_MEDIR_COSTO.md); 45 en C
(reconstruccion declarada, energia solo como cota por TDP).
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"

FILAS = {
    "44": ("P", "22/11: 9 de 10 redes medidas en CPU homogénea (256×256, batch 1, 1 y 6 hilos, 5 de calentamiento y 30 "
           "repeticiones, mediana e IQR): parámetros, FLOPs (FlopCounterMode, cota inferior), memoria (pico de proceso e "
           "incremento en inferencia) y latencia. Los FLOPs no ordenan la latencia (PIDNet-S, SegNeXt-T y SegFormer-B0: "
           "3,1–3,5 G, 90–216 ms con 1 hilo); TransUNet 77 G y 2,1 s. CPU, no L4/T4; equipo compartido. VM-UNet exige "
           "CUDA: solo parámetros; medición en GPU preparada (`COLAB_MEDIR_COSTO.md`). `scripts/medir_costo_e2.py`, "
           "`resultados/costo_e2/`"),
    "45": ("C", "22/11: costo de GPU reconstruido de lo registrado: 33,3 h (entrenamiento 16,4 h; diagnósticos 1,7 h; "
           "complemento 15,2 h), todas en L4. Energía solo como cota superior TDP×horas ≈ 2,4 kWh de GPU (no medida; sin "
           "CPU ni centro de datos). Sin registro: test (T4), réplica, tiempo de sesión: el total es un mínimo. "
           "`scripts/costo_total_e2.py`, `resultados/costo_e2/costo_total_e2.md`, `LECTURA.md`"),
}
APENDICE = {}
NOTA = ("\n**Actualización del 22/11 (costo).** Parámetros, FLOPs, memoria y latencia de 9 de las 10 redes en CPU "
        "homogénea (fila 44; VM-UNet pendiente de GPU) y costo de GPU reconstruido con cota de energía (fila 45).\n")


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
