# Reproducción completa en un clon limpio: lectura

> **Trazabilidad.** Generado el 2026-11-22 con `scripts/reproduccion_completa_e2.py` (detalle en `REPORTE.json`, en
> esta carpeta). Clon de `https://github.com/luisperez03/Deep_Learning_IMO`, rama `cierre-e2`, commit `7b669a7`, con
> `git clone --depth 1`: solo entra lo publicado. Intérprete aislado (`-E -s`, variables de entorno mínimas): Python
> 3.12.16, numpy 2.5.5, scipy 1.18.1, Pillow 13.0.0, matplotlib 3.11.2, Windows 11. Libro con jupyter-book 0.15.1
> (Python 3.9). Duración total: 8 min 19 s, sin GPU.
>
> Orden ejecutada:
>
> ```
> python scripts/reproduccion_completa_e2.py --destino <carpeta nueva> --voc <ruta>/VOC2012
>        --test <ruta>/resultados_test_e2_final --python <python limpio> --python-alt <python con torch>
>        --jupyter-book <ruta>/jupyter-book
> ```

## Qué entró al clon además del repositorio

| Dato externo | Cómo se verificó |
|---|---|
| `VOC2012` (HAZMAT-13, 1.685 imágenes) | SHA-256 de cada imagen y máscara contra los manifiestos `resultados/preentrenamiento/*.csv` (lo hace `auditar_test_e2.py` para test y `reconstruir_mascaras_binarias_e2.py` para las 1.451 máscaras) |
| `resultados_test_e2_final/` (test cerrado) | 2.234 archivos: 50 JSON por corrida, `EVALUACION_CERRADA.json`, `RESUMEN_TEST.json`, `INFORME_TEST.md`, `environment.freeze.txt` y 2.180 máscaras de semilla 11. Huella del conjunto (SHA-256 de la lista ordenada `ruta hash`): `e9d87e516e6909ca5504181809fa8c41f4ee083d42b369a072a97d9d1d78cf82`. La auditoría verifica la firma del protocolo, el `payload_sha256` de cada JSON y el hash de cada checkpoint registrado |
| Máscaras binarias | Regeneradas desde VOC: 1.451 de 1.451 idénticas por SHA-256 |
| Historias de entrenamiento (`tesis_imo_e2_arquitecturas_reales_v2/`) | **No entraron**: solo están en el Drive del proyecto |

## Resultado: cifra a cifra contra lo publicado

**Tablas y contrastes del test: discrepancia 0.** La cadena `auditar_test_e2.py` → `graficar_test_e2.py` →
`analisis_integracion_e2.py` → `integrar_entregable2.py` → `completar_analisis_locales_e2.py` terminó sin error y
reescribió 36 archivos (sin contar las máscaras ni las dos curvas, que se restauran desde git). Contra la versión de GitHub:

| Salida | Discrepancia máxima | Lectura |
|---|---|---|
| `resultados/auditoria_test_e2/AUDITORIA.json`, `CONCLUSIONES.md` | 0: idénticos byte a byte | Tabla de test, IC por grupos, contraste principal; 0 incidencias, 10.900 registros y 2.180 máscaras verificadas |
| `resultados/integracion_e2/ANALISIS.md` | 0: idéntico byte a byte | Tabla de contrastes, Friedman, recursos |
| `resultados/integracion_e2/analisis.json` (611 valores) | 1,8 × 10⁻¹⁵ (en la CD de Nemenyi) | Redondeo de coma flotante; contrastes, p de intercambio y de Wilcoxon, Holm, estratos y Friedman, iguales |
| `complementos_locales_e2/`: bordes de semilla 11 (por imagen y resumen), EDA por imagen | 0: idénticos byte a byte | Ya con las máscaras de test, que en la prueba anterior faltaban |
| `complementos_locales_e2/sensibilidad_estadistica.json` | 6,9 × 10⁻¹⁴ | Coma flotante |
| `complementos_locales_e2/RESUMEN.json` | 8,7 × 10⁻⁵ (AUC del adversarial) | Remuestreo bilineal del letterbox (Pillow 13 frente a 11.3 declarado) |
| `complementos_locales_e2/validacion_adversarial.json` | 0,0045 en una probabilidad fuera de fold | Misma causa; AUC, IC y p de permutación sin cambio de lectura |
| `complementos_locales_e2/descriptores_imagen.json` (87.060 valores) | 0,0014 en un bin de histograma | Misma causa |
| `complementos_locales_e2/histogramas_rgb_por_clase.json` | 337 conteos en un bin de millones (0,005 % del total) | Misma causa; es la discrepancia máxima global en valor absoluto |
| `complementos_locales_e2/PROTOCOLO.json` | 0 numérica; 9 hojas de texto | El archivo de configuración tiene hoy una sección `conformal` que se añadió después de la ejecución publicada; el protocolo regenerado la copia. No es un resultado distinto |

Las diferencias de la última parte de la tabla coinciden, valor por valor, con las que ya midió la prueba de entorno
limpio anterior (`resultados/entorno_limpio_e2/LECTURA.md`): son la misma variación de versión de Pillow y no tocan
ninguna conclusión.

**Figuras.** Se regeneraron 13 figuras del libro (comparación, Pareto, rangos y diferencia crítica, y los 10 diagramas
de redes) y la figura de validación contra test. Las PNG no son idénticas píxel a píxel: difieren entre el 1 % y el 6 %
de los píxeles, en bordes de texto y de marcadores. Comparadas lado a lado, la figura de Pareto muestra los mismos
puntos, rótulos y frontera; la diferencia es de rasterizado (matplotlib 3.11.2 frente a 3.10.3 declarado). Los datos
que grafican están en los JSON de arriba, con discrepancia 0.

**No reproducido: dos figuras.** `libro/figuras/e2/curvas_validacion.png` y `curvas_perdida.png` se dibujan desde
`history.json` de las 50 corridas, que solo están en el Drive del proyecto. El script corre el análisis con esas curvas en blanco,
restaura las dos figuras desde git y las deja marcadas como no reproducidas en `REPORTE.json`. Con la carpeta de
historias, el mismo comando las regenera (`--historias`).

## Recálculo independiente desde las máscaras

`recalculo_independiente()` no importa código del proyecto: solo numpy y Pillow. Parte de las máscaras originales de
VOC (todo color distinto de negro es etiqueta), aplica su propio letterbox a 256 con vecino más cercano y calcula el
Dice dentro de la zona válida.

| Comprobación | Unidades | Discrepancia máxima |
|---|---|---|
| Dice por imagen, 10 redes de semilla 11, desde las máscaras predichas contra las de referencia | 2.180 | 0 |
| Dice por grupo de esas mismas corridas | 1.220 (10 × 122) | 0 |
| Agregación imagen → grupo en las 50 corridas | 6.100 (50 × 122) | 0 |
| Dice de test por red y DE entre semillas, contra `RESUMEN_TEST.json` y la auditoría publicada | 20 | 0 |
| Diferencia de cada red frente a FCN, contra `RESUMEN_TEST.json` | 9 | 2,2 × 10⁻¹⁶ |
| Diferencia de cada red frente a FCN, contra `resultados/integracion_e2/analisis.json` | 9 | 0 |
| Cifras impresas en `INFORME_TEST.md` (4 decimales) | 20 | 0 distintas |

**Repetición de la inferencia.** La única repetición de la inferencia de test es la etapa `clean` de
`diagnosticos_gpu_e2.py` (otra sesión de Colab, los mismos 50 checkpoints, torch 2.5.1+cu121). Su Dice medio por red
difiere del test cerrado en 1,3 × 10⁻⁵ como máximo (PIDNet-S); tres redes dan exactamente lo mismo. Es el orden de
magnitud esperable del no determinismo de cuDNN, y ninguna cifra impresa con cuatro decimales cambia más de una
unidad en la cuarta (PIDNet-S, 0,7937 frente a 0,7936).

## Tests

11 de 12 archivos pasan. `test_e2.py` falla en una prueba (`test_resume_matches_and_rejects_changed_identity`) porque
construye TransUNet, que pide `ml-collections` y `einops`. Los dos están declarados en `requirements-e2.txt` y en el
`environment.freeze.txt` del test cerrado; ninguno de los dos intérpretes locales los tiene a la vez (uno carece de
`ml-collections`, el otro de `einops`). Es una falta del entorno local, no un resultado distinto; instalar paquetes queda
fuera de esta prueba.

## Libro y cuadernos (fila 68)

**Libro.** `jupyter-book build libro -W --keep-going` en el clon: código 0, **0 avisos**, 18 páginas HTML. La falta de
30 figuras y del cuaderno descargable que detenía la prueba anterior ya no existe: están versionados. Se probó con
jupyter-book 0.15.1; el proyecto declara 1.0.4.

**Cuadernos.** El libro tiene `execute_notebooks: "off"`: muestra las salidas guardadas, no ejecuta. Por eso se
ejecutaron aparte los 11 cuadernos del repositorio, clasificados por su código:

| Cuaderno | Pide | Ejecutado | Resultado |
|---|---|---|---|
| `libro/eda_original.ipynb` | nada fuera del repo y VOC | sí | 9 de 9 celdas sin error, 25 figuras; 8 de 9 celdas con texto idéntico al guardado; la otra difiere solo en la línea de versiones de numpy y Pillow. La firma del conjunto de imágenes y máscaras coincide |
| `notebooks/01_eda_voc.ipynb` | nada fuera del repo y VOC | sí | 9 de 9 celdas sin error, 25 figuras (no trae salidas guardadas para comparar) |
| `colab/Benchmark_E2_Colab_v2.ipynb`, `notebooks/benchmark_e2_colab.ipynb` | Colab, Drive y GPU | no | Entrenamiento de las 50 corridas |
| `colab/Evaluacion_Final_E2_Colab.ipynb`, `notebooks/evaluacion_final_e2.ipynb` | Colab, Drive y GPU | no | Test cerrado; sus salidas son `resultados_test_e2_final/`, verificadas arriba |
| `colab/Complemento_GPU_E2.ipynb` | Colab, Drive y GPU | no | Diagnósticos; su etapa `clean` es la repetición de la inferencia de arriba |
| `colab/Complemento_Entrenamiento_E2.ipynb`, `colab/Atajos_Errores_E2.ipynb`, `colab/Replica_SegFormer_ADE20K.ipynb` | Colab, Drive y GPU | no | Ablación y búsqueda, atajos, réplica |
| `colab/Extraer_Validacion_E2.ipynb` | Colab y Drive (los checkpoints) | no | Dice por imagen de validación |

Antes de los cuadernos de EDA se generan `resultados/inicio_voc/` y `resultados/revision_voc/` con
`scripts/inicio_voc.py` y `scripts/revisar_voc.py` (código 0): los cuadernos los leen y no se versionan. Las celdas se
ejecutaron en orden, en un mismo espacio de nombres, con el intérprete limpio: el único kernel de Jupyter disponible
(el del entorno del libro) trae Pillow 10.0, que no acepta `ImageFont.load_default(size=)` y falla en la segunda
celda. Con un kernel con Pillow ≥ 10.1 el cuaderno corre en Jupyter igual.

## Qué cambia en el checklist

- **55 y B20.** Reproducción en clon limpio desde GitHub con los datos externos declarados: tablas y contrastes del
  test con discrepancia 0, recálculo independiente desde las máscaras con discrepancia 0, reinferencia en GPU con
  1,3 × 10⁻⁵. Queda declarado lo no reproducido (dos figuras de curvas, por las historias en Drive).
- **68.** Libro compilado en el clon con `-W` sin avisos; cuadernos sin GPU ejecutados de punta a punta; los GPU,
  clasificados y documentados.
- **B21.** Un comando reproduce todo el informe desde el clon y los datos; el entorno queda registrado y las
  instrucciones extremo a extremo están en `docs/REPRODUCCION_EXTREMO_A_EXTREMO.md`.

## Una propuesta para `reproducir_entregable2.py`

La orden del proyecto llama a `analisis_integracion_e2.py` e `integrar_entregable2.py` sin haber corrido antes
`auditar_test_e2.py` y `graficar_test_e2.py`; en un clon nuevo `integrar_entregable2.py` necesita
`resultados/auditoria_test_e2/comparacion_validacion_test.png`, que no se versiona y sale de `graficar_test_e2.py`. La
propuesta es anteponer esos dos pasos (como hace `reproduccion_completa_e2.py`). No se editó.

## Carpetas que deja la prueba

`C:\Proyectos\estudio\datos\deep_learning\reproduccion_completa_e2_20261122b` (la corrida de esta lectura, 91 MB) y
`..._20261122` (una corrida anterior del mismo día, con el commit `ea4ee9a` y las mismas cifras; los cuadernos se
ejecutaron ahí después). Cada una lleva enlaces de directorio a VOC2012 y al test cerrado en `repo/e2/`: para
borrarlas, quitar antes los enlaces con `rmdir`, que no sigue al destino.
