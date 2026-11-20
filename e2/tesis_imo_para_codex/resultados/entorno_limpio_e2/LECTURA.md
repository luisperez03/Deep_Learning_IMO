`prueba_entorno_limpio_e2.py` copió el proyecto desde git (commit `a900dec`, solo lo versionado) a una carpeta
nueva, puso `VOC2012` como hermano del proyecto (enlace al conjunto, SHA-256 de las 1.451 imágenes verificado) y
corrió todo con un intérprete aislado (`-E -s`, variables de entorno mínimas): Python 3.12.16, numpy 2.5.5, scipy
1.18.1, scikit-learn 1.9.1, Pillow 13.0.0, Windows 11. El proyecto declara numpy 2.2.6, scipy 1.15.3, scikit-learn
1.6.1 y Pillow 11.3.0: la prueba es también de portabilidad entre versiones. Detalle en
`resultados/entorno_limpio_e2/REPORTE.json`.

**La orden de reproducción no corre completa desde el snapshot.** `scripts/reproducir_entregable2.py` falla en su
segundo paso, `analisis_integracion_e2.py`, porque lee artefactos que el repositorio no lleva. Faltan, con
precisión:

| Falta | Qué es | Quién lo lee |
|---|---|---|
| `../resultados_test_e2_final/` | 50 JSON de test por corrida y `mascaras_seed11/` (2.180 PNG) | `analisis_integracion_e2.py`, `integrar_entregable2.py`, `auditar_test_e2.py`, `completar_analisis_locales_e2.py` (bordes y sumas de cuadrados) |
| `../tesis_imo_e2_arquitecturas_reales_v2/` | 50 carpetas de corrida con `history.json` (y checkpoints) | `analisis_integracion_e2.py` (curvas y costo) |
| `resultados/preentrenamiento/masks_binary/` | 1.451 máscaras binarias | todo lo que carga ejemplos; **reconstruibles**: `reconstruir_mascaras_binarias_e2.py` las regenera idénticas byte a byte (1.451/1.451 por SHA-256) |
| 30 figuras de `libro/figuras/` y `libro/eda_original.ipynb` | imágenes del libro del Entregable 1 y del EDA, y el cuaderno descargable | `jupyter-book build libro -W` |
| paquete `ml-collections` | dependencia de TransUNet | un test de `test_e2.py` (reanudación de entrenamiento) |

**Tests.** 8 de 9 archivos pasan (36 de 37 pruebas). El que falla, `test_e2.py`, falla en una sola prueba por la
falta de `ml-collections` en el entorno, no por un resultado distinto.

**Re-ejecución desde datos.** `completar_analisis_locales_e2.py` corrió en la copia hasta donde pide las máscaras de
test de seed11 (EDA por imagen, descriptores, histogramas y adversarial train/test con 200 permutaciones):

| Salida regenerada | Discrepancia máxima con lo versionado |
|---|---|
| `eda_por_imagen.json` (13.059 valores) | 0 |
| `descriptores_imagen.json` (87.060 valores) | 0,0014 en un bin de histograma de una imagen |
| `histogramas_rgb_por_clase.json` | 337 conteos en un bin de millones; 0,005 % del total, 0,12 % relativo en el peor bin |
| `validacion_adversarial.json` | AUC 0,55981 frente a 0,55972 (Δ 0,0001); IC [0,5062; 0,6144] frente a [0,5061; 0,6143]; p de permutación idéntico, 0,0896 |

Las diferencias vienen del remuestreo bilineal del letterbox (Pillow 13 frente a 11.3), que mueve algunos píxeles de
bin; las máscaras, que usan vecino más cercano, reproducen exacto. Ninguna conclusión cambia.

**Complementos de este capítulo.** Fuga por embeddings, sensibilidad sin casi duplicados, EDA por clase y γ se
reproducen con discrepancia 0; el shift train/val, con 0,0003 (en el IC condicional del adversarial), por la misma
razón de los descriptores. La **discrepancia máxima global** es la de los histogramas (337 conteos, 0,005 % del
total); en estadísticos, 0,0045 en una probabilidad fuera de fold del adversarial.

**Libro.** Con jupyter-book 0.15.1 (el disponible; el proyecto pide 1.0.4) el HTML se construye, pero `-W` lo
detiene por 31 avisos: 30 figuras no versionadas y el cuaderno `eda_original.ipynb`. Ninguno sale del capítulo
nuevo. Para cerrar los controles 55 y 68 hacen falta esos archivos en el repositorio o como hermanos, y repetir con
los 50 JSON de test presentes.
