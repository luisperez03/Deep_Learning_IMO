"""Actualiza filas del checklist maestro con la evidencia de los complementos del 20/11/2026.

  python scripts/actualizar_checklist_cierre.py

Reemplaza el estado y la evidencia de las filas listadas abajo en docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md,
por ID. Un cuaderno o un script preparados NO cambian una F a C: se anotan como "preparado, sin ejecutar" con su
ruta. Falla si una fila no existe o si ya fue cambiada a otro texto (para no pisar ediciones posteriores).
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHECK = RAIZ / "docs" / "revision_entregable2" / "CHECKLIST_ENTREGA_E2.md"

# ID: (estado nuevo, evidencia nueva)
FILAS = {
    "09": ("C", "Conteos/áreas/píxeles y, 20/11, entropía de la distribución de clases y de áreas "
           "(`scripts/eda_clases_formas_e2.py`, capítulo `complementos_datos_e2`)"),
    "11": ("C", "Hashes/dHash/aHash y, 20/11, embeddings ResNet-50 (`scripts/fuga_embeddings_e2.py`): los grupos dHash no "
           "separan sesiones; 9 grupos de test y 8 de val con casi duplicado (coseno > 0,97); sensibilidad sin ellos: "
           "el Dice baja 0,003–0,010 y el orden no cambia (`scripts/sensibilidad_casi_duplicados_e2.py`, semilla 11)"),
    "12": ("P", "20/11: revisión manual de 100 imágenes, primera pasada: 7 errores (IC95 3,4–13,8 %), todos rótulos "
           "cortados por el borde, y 5 dudosas; 12 casos por confirmar a mano (`resultados/ruido_etiqueta_e2/`). "
           "Sin segunda anotación independiente: el acuerdo sigue pendiente"),
    "15": ("C", "Train/test: adversarial con 200 permutaciones (`complementos_locales_e2`); train/val, 20/11: KS con Holm, "
           "MMD y adversarial (`scripts/shift_train_val_e2.py`). Sin cambio de distribución detectado"),
    "16": ("C", "Letterbox, tamaños, normalización train y, 20/11, formas, tono e histogramas por clase "
           "(`scripts/eda_clases_formas_e2.py`, tres figuras)"),
    "32": ("C", "20/11, Colab T4: checkpoint oficial de SegFormer-B0 en ADE20K val (2.000), escala única, mIoU 37,43 % "
           "(IC95 35,47–38,30) frente a 37,4; D = +0,03 (IC95 −1,93 a +0,90); veredicto a priori `compatible`; "
           "desviaciones D1 (−1,49 con el preprocesador de HF) y D2 (media por imagen). Réplica de evaluación, no de "
           "entrenamiento. `resultados/replica_segformer/LECTURA.md`"),
    "33": ("C", "20/11: protocolo a priori `config/limitacion_segformer_area.json`, terciles de área en los 122 grupos "
           "de validación, SegFormer frente a 3 redes piloto (U-Net, DeepLabv3+, SegNeXt-T), 5 semillas. Veredicto "
           "`no_confirmada`: Δ T3−T1 de SegFormer 0,082 [−0,030; 0,194] y sin diferencia con las piloto (p Holm ≥ 0,55); "
           "déficit de nivel, −0,12 frente a U-Net en el tercil pequeño. Dice por imagen extraído de los 50 checkpoints "
           "(`colab/Extraer_Validacion_E2.ipynb`). `resultados/limitacion_segformer_area/RESUMEN.md`"),
    "37": ("F", "Preparado, sin ejecutar: búsqueda con el mismo espacio relativo y el mismo presupuesto para las 10 "
           "(`config/complemento_entreno_e2.json`, `colab/Complemento_Entrenamiento_E2.ipynb`), exploratoria por la "
           "exposición previa del test; T = 4 propuesto (2,3–5,8 h L4 estimadas)"),
    "40": ("C", "`scripts/baseline_clasico_e2.py`: HistGradientBoosting, 13 rasgos, 3 configuraciones × 5 semillas, elección "
           "por validación; Dice test 0,645 (IC95 grupos 0,605–0,682) en `resultados/baseline_clasico_e2/`"),
    "41": ("P", "Diez preflight GPU y entrenamiento completo. Preparado, sin ejecutar: sobreajuste deliberado a un lote "
           "(10 redes × 500 pasos) en `colab/Complemento_Entrenamiento_E2.ipynb`"),
    "46": ("P", "Integridad, partición, adversarial y, 20/11, embeddings y casi duplicados (ID 11). Pendiente: "
           "permutación de etiquetas (ID 56)"),
    "55": ("P", "Reagregación desde registros: 0. 20/11, entorno limpio (`scripts/prueba_entorno_limpio_e2.py`): lo que "
           "corre con el repositorio coincide (máx. 0,0014); la reproducción completa necesita los resultados del test "
           "y las corridas, que están fuera del repositorio (`resultados/entorno_limpio_e2/LECTURA.md`)"),
    "56": ("F", "Dice vacío = 0. Preparado, sin ejecutar: permutación de etiquetas del pipeline (FCN con B = 200; "
           "TransUNet con B = 19, por debajo del mínimo y así declarado) en `colab/Complemento_Entrenamiento_E2.ipynb`"),
    "57": ("C", "20/11, L4, exploratorio (`colab/Complemento_GPU_E2.ipynb`, etapa `clean`): reproduce el Dice de test "
           "cerrado (dif. máx. < 0,0001); brecha train−val positiva en las 10, de 0,026 (SegNet, IC con 0) a 0,213 "
           "(SegFormer) y 0,147 (SegNeXt). `resultados/diagnosticos_gpu_e2/TABLAS.md`, capítulo `diagnosticos_e2`"),
    "58": ("C", "20/11: Brier, NLL, ECE15 fijo y adaptativo, con y sin temperatura ajustada en validación; ECE15 0,008–"
           "0,014; temperatura 1,6–2,6 en todas (sobreconfianza); la temperatura baja el Brier y sube levemente el ECE15"),
    "59": ("C", "20/11: 10 redes × 5 semillas × ruido/desenfoque/brillo × 3 niveles en test. Área normalizada: redes "
           "grandes 0,92–0,96 con ruido y desenfoque; FCN, U-Net y SegNet 0,68–0,77 en las tres perturbaciones"),
    "60": ("C", "20/11: Integrated Gradients en 100 casos de validación del top 3 × 5 semillas; IG supera al borrado "
           "aleatorio en el 89 % (TransUNet) y 100 % (U-Net, FCN) de los casos; Spearman entre semillas 0,73 / 0,92 / 0,92"),
    "61": ("F", "Preparado, sin ejecutar: TransUNet sin Transformer, sin conexiones de salto y con BCE sola, mismas 5 "
           "semillas (≈ 10 h L4 estimadas), en `colab/Complemento_Entrenamiento_E2.ipynb`"),
    "62": ("C", "20/11: ensamble de 5 semillas del top 3: +0,021 a +0,028 de Dice (IC sin 0; TransUNet 0,883); "
           "predicción conforme Mondrian por clase calibrada en validación, cobertura test 0,903–0,912 (nominal 0,90), "
           "con 34–37 % de grupos bajo el nominal (garantía marginal)"),
    "63": ("F", "Sin segunda anotación: no hay techo medido. Kit de reanotación de 50 imágenes de validación listo, con "
           "el cálculo de γ (`scripts/techo_reanotacion_e2.py`); los rótulos cortados por el borde se marcan aparte"),
    "64": ("C", "20/11: HD95/ASSD y precisión/recall de las 5 semillas en test (etapa `clean`); HD95 de 17,1 px "
           "(TransUNet) a 34,3 (SegFormer, recall 0,688)"),
}
APENDICE = {
    "B03": ("P", "ID46; fuga por embeddings hecha; permutación de etiquetas preparada"),
    "B09": ("C", "Máscara vacía, baseline clásico (ID40) y tres E1"),
    "B13": ("F", "ID63; kit listo, sin segunda anotación"),
    "B16": ("C", "ID59–60 ejecutados el 20/11 (exploratorio)"),
    "B17": ("C", "ID62 ejecutado el 20/11 (exploratorio)"),
    "B19": ("C", "ID32 réplica compatible; ID33 limitación por área no confirmada (déficit de nivel)"),
    "B20": ("P", "Agregación 0; entorno limpio parcial (ID55)"),
}


def main():
    lineas = CHECK.read_text(encoding="utf-8").splitlines()
    pendientes = dict(FILAS)
    pend_ap = dict(APENDICE)
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
    nota = ("\n**Actualización del 20/11 (tarde).** Se integraron los complementos de datos, fuga, ruido, bibliografía y "
            "los cuadernos preparados de diagnóstico GPU, entrenamiento y réplica. Lo preparado sin ejecutar sigue en F, "
            "con su ruta. Orden de corridas: `docs/GUIA_CORRIDAS_CIERRE_E2.md`.\n")
    texto = texto.replace("\n## Problema, hipótesis", nota + "\n## Problema, hipótesis", 1)
    CHECK.write_text(texto, encoding="utf-8")
    print(f"{len(FILAS)} filas y {len(APENDICE)} casillas actualizadas en {CHECK}")


if __name__ == "__main__":
    main()
