# Réplica de evaluación de SegFormer-B0 en ADE20K (ítem 32)

> **Trazabilidad.** Ejecutada el 2026-11-20 en Google Colab, GPU Tesla T4, con `colab/Replica_SegFormer_ADE20K.ipynb`
> (generado por `scripts/crear_colab_replica.py`, sin cambios salvo `ACEPTO_TERMINOS_ADE20K = True`). Script
> `scripts/replica_segformer_ade.py` y protocolo `config/replica_segformer.json`, ambos verificados por SHA-256 dentro
> del cuaderno; el protocolo se fijó antes de cualquier evaluación. Salida completa: `resumen_replica.json`. Las áreas
> por imagen y clase (`areas_primaria.npz`, `areas_hf_512.npz`, unos 250 KB cada uno) no entran al repositorio por la
> regla `**/*.npz` del `.gitignore`; con ellas el mIoU y el IC se recalculan sin GPU con las funciones del script y
> dan las mismas cifras (comprobado el mismo día).

## Resultado

| Variante | mIoU (%) | IC95 bootstrap por imagen (%) |
|---|---:|---:|
| Primaria: preprocesamiento de la configuración oficial (AlignedResize 2048 × 512, escala única, `whole`) | **37,43** | 35,47 – 38,30 |
| D1: preprocesador por defecto de Hugging Face (512 × 512 sin proporción) | 35,94 | 34,02 – 36,85 |
| D2: media por imagen en lugar del mIoU acumulado (no es la métrica del artículo) | 33,90 | — |

- Reportado en el artículo (escala única): 37,4 %. Diferencia D = +0,03 puntos; IC95 de D: −1,93 a +0,90.
- **Veredicto según el criterio fijado de antemano: `compatible`.** |D| ≤ 1 punto, pero el IC95 de D sale del
  intervalo [−1; +1] por abajo, que es lo que el protocolo exige para "replicado". El 37,4 cae dentro del IC95.
- Desviaciones: el preprocesamiento por defecto de Hugging Face resta 1,49 puntos (D1); promediar por imagen resta
  3,5 (D2). D3 (pesos convertidos frente al `.pth` oficial) no se evaluó y queda como causa no descartada.
- Clases con menor IoU: shower 0,0 %, tray 1,2 %, bannister 1,6 %, crt screen 1,8 %, bag 2,1 %.
- Datos: `ADEChallengeData2016.zip` de la fuente oficial, 967.382.037 bytes, SHA-256
  `7ff1be44964418441f542a7cc1e1a650e7dc0fc275f5d23252bc9bbdbc977b29`, 2.000 pares de validación. Términos de uso
  de ADE20K aceptados antes de descargar.
- Modelo: `nvidia/segformer-b0-finetuned-ade-512-512`, revisión `489d5cd8`, SHA-256 del `model.safetensors`
  verificado, 3.752.694 parámetros. Versiones: Python 3.13.16, torch 2.14.1+cu130, transformers 5.22.1.

## Qué dice y qué no

Es una réplica de **evaluación** (nivel 1): confirma que el checkpoint publicado, con el preprocesamiento y la métrica
del artículo, da la cifra reportada. **No** replica el entrenamiento (nivel 2, solo diseñado). Tampoco dice nada del
SegFormer-B0 entrenado desde cero en HAZMAT-13: allí la diferencia con el artículo (preentrenamiento, receta, escala
y dominio) es de diseño, no de reproducción.
