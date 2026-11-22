# Checklist maestro del Entregable 2

Revisado el 20/11/2026 contra la guía original de 65 páginas, incluida ruta F4b y apéndice B. Entrega prevista: domingo 22/11/2026. Este control no garantiza una nota.

**C:** cubierto con evidencia. **P:** parcial. **F:** falta ejecutar/verificar. **D:** desviación histórica no reparable retroactivamente. **NA:** no aplica con motivo. Un plan o un script no acreditan una ejecución. No convertir el número de filas en porcentaje de calificación.

El usuario confirmó que el profesor acepta **diez modelos totales: tres E1 y siete nuevos**. No se infieren dispensas de otros requisitos. Q5/Q6 siguen omitidas por instrucción explícita.

**Actualización del 20/11 (tarde).** Se integraron los complementos de datos, fuga, ruido, bibliografía y los cuadernos preparados de diagnóstico GPU, entrenamiento y réplica. Lo preparado sin ejecutar sigue en F, con su ruta. Orden de corridas: `docs/GUIA_CORRIDAS_CIERRE_E2.md`.

**Actualización del 21/11.** Se corrió el complemento de entrenamiento en L4: sobreajuste, búsqueda T = 4, permutación de FCN con B = 200 y ablación de TransUNet (filas 37, 41, 46, 56, 61). Exploratorio: el test ya estaba observado y no se cargó.

**Actualización del 22/11.** Revisión humana: los 12 casos de ruido de etiqueta confirmados a mano (fila 12) y techo por segunda anotación independiente de 50 imágenes de validación (fila 63).

**Actualización del 22/11 (cierre).** Filas desactualizadas puestas al día con evidencia ya registrada (66, 70, 75; B11).

**Actualización del 22/11 (datos).** Data Card de HAZMAT-13 con procedencia, versión, huellas, adquisición, licencia, ética y obtención (filas 07 y 71); la licencia y adquisición de las imágenes siguen sin certificar (fila 08).

**Actualización del 22/11 (inferencia).** Equivalencia TOST en test y validación, descomposición de varianza con modelo mixto en validación, simetría y sensibilidad, max-T simultáneo y sesgo del ganador (filas 50, 51, 53, 54 y B15). Exploratorio, posterior al test.

**Actualización del 22/11 (literatura).** Fichas de extracción cotejadas con el PDF (fila 22), puntuación definitiva con citación (27), justificación de Q5/Q6 (24) y cribado del núcleo preparado (20, 21, 30), que espera las exportaciones originales de Scopus/WoS.

**Actualización del 22/11 (costo).** Parámetros, FLOPs, memoria y latencia de 9 de las 10 redes en CPU homogénea (fila 44; VM-UNet pendiente de GPU) y costo de GPU reconstruido con cota de energía (fila 45).

**Actualización del 22/11 (errores y atajos).** Taxonomía automática, subgrupos y daño (local): filas 13, 65 y 67, con protocolo previo y exploratorio.

## Problema, hipótesis, datos y EDA — rúbrica 15 %

| ID | Requisito (página guía) | Estado | Evidencia / cierre necesario |
|---|---|---|---|
| 01 | Formulación entrada/salida/riesgo (2) | C | `metodos_e2.md`: segmentación binaria; no clasificación IMO |
| 02 | Relevancia y ≥3 referencias aplicadas (2) | C | 20/11: seis estudios aplicados a rótulos HAZMAT con DOI verificado en Crossref y doi.org (`evidencia_doi_registro.json`), con tarea y método leídos del resumen en `APLICACION_Y_GAP.md`; en `libro/literatura.md` y en el BibTeX |
| 03 | Métrica, δ y unidad heredados (2) | C | Dice por grupo, δ=0,02; IoU/mIoU/accuracy; independencia aproximada |
| 04 | ≥3 RQ, H0/H1, α, efecto y decisión (2–3) | P | RQ1–4 integradas; temporalidad explícita; equivalencia no ejecutada |
| 05 | Preregistro completo anterior al entrenamiento (42) | D | Configuración/cierre antes de test disponibles; no preregistro integral previo acreditado. No retrofechar |
| 06 | Potencia y número de ejecuciones prospectivos (43) | D | Cinco semillas; MDD/R añadidos solo para planificación futura |
| 07 | Data Card, procedencia, versión/hash (3) | C | 22/11: Data Card completa, `docs/DATA_CARD_HAZMAT13.md`. Fuente: mrl-amrl/HAZMAT13 (commit `ec85498`, 12/07/2021, sin releases) y artículo doi:10.1016/j.mlwa.2021.100104. Coinciden 1.685 imágenes, 13 clases con su orden, formato VOC y ruta `/home/amrl/` en los XML; huellas agregadas de imágenes y XML y SHA256 por archivo (2.902 de 2.902 verificados). Identidad binaria con el Drive de los autores no verificada (declarado): requiere descargar con permiso. `scripts/verificar_corpus_hazmat13.py` |
| 08 | Licencias, adquisición, geografía, ética (3,41) | P | 22/11: verificado: licencia MIT del repositorio (copyright MRL-AMRL 2021), que nombra «Software»; README y artículo no declaran licencia de datos, titularidad de las fotos ni cómo se adquirieron (sin EXIF; `database = Unknown` en los XML; parte de las muestras viene de situaciones de RoboCup). Geografía no documentada. Ética y límites declarados en `docs/DATA_CARD_HAZMAT13.md`. Sin redistribución de imágenes. Sigue P hasta aclaración de los autores (no enviada) |
| 09 | Distribución, desbalance y entropía (3–4) | C | Conteos/áreas/píxeles y, 20/11, entropía de la distribución de clases y de áreas (`scripts/eda_clases_formas_e2.py`, capítulo `complementos_datos_e2`) |
| 10 | Faltantes, corrupción y consistencia (4) | C | 102 máscaras no utilizables, 132 cuarentena, geometría y hashes |
| 11 | Duplicados y similitud residual (4,44) | C | Hashes/dHash/aHash y, 20/11, embeddings ResNet-50 (`scripts/fuga_embeddings_e2.py`): los grupos dHash no separan sesiones; 9 grupos de test y 8 de val con casi duplicado (coseno > 0,97); sensibilidad sin ellos: el Dice baja 0,003–0,010 y el orden no cambia (`scripts/sensibilidad_casi_duplicados_e2.py`, semilla 11) |
| 12 | Ruido de etiqueta y acuerdo independiente (4,48) | C | 22/11: 100 imágenes de train/val; los 12 casos de la primera pasada confirmados a mano: 10 de 100 con rótulo cortado por el borde sin máscara (IC95 Wilson 5,5–17,4 %), todos tipo falta; el conjunto a veces los anota (8,5 % de train con región en el borde): inconsistencia. Acuerdo entre anotadores: ID 63. `resultados/ruido_etiqueta_e2/`, capítulo `complementos_datos_e2` |
| 13 | Sesgos y atajos (4) | P | 22/11, exploratorio: señal del EDA (origen de captura) contrastada por subconjuntos: O1–O6 sin diferencias; O7 (5 imágenes) mejor en las 10 redes. Falta la prueba contrafactual (sin contexto, sin rótulo, gris): `colab/Atajos_Errores_E2.ipynb`, preparado sin correr. `resultados/errores_atajos_e2/` (`LECTURA.md`), protocolo previo del 22/11 (`PROTOCOLO_ERRORES_ATAJOS_20261122.md`) |
| 14 | Particiones y semillas regenerables (4) | C | 1015/218/218 imágenes, 567/122/122 grupos, semilla 20261106 |
| 15 | Shift / validación adversarial (4,45) | C | Train/test: adversarial con 200 permutaciones (`complementos_locales_e2`); train/val, 20/11: KS con Holm, MMD y adversarial (`scripts/shift_train_val_e2.py`). Sin cambio de distribución detectado |
| 16 | Geometría, canales, intensidad y formas (7–8) | C | Letterbox, tamaños, normalización train y, 20/11, formas, tono e histogramas por clase (`scripts/eda_clases_formas_e2.py`, tres figuras) |
| 17 | Tabla hallazgo–decisión–riesgo (9,61) | C | Tabla integrada al EDA |

## Literatura, paper guía y gap — rúbrica 15 %

| ID | Requisito (página guía) | Estado | Evidencia / cierre necesario |
|---|---|---|---|
| 18 | ≥2 bases, ecuaciones, filtros, fechas/conteos (9–10) | C | Scopus/WoS, protocolo, bitácora, 13 exportaciones Q0–Q4 |
| 19 | Inclusión/exclusión y excepciones antiguas (10) | C | Criterios documentados; aplicación final distinta de definición |
| 20 | PRISMA completo y exclusiones razonadas (10) | P | 22/11: alcance fijado (texto completo del núcleo de dominio + muestra estratificada de 200 del precribado con IC95 de Wilson); `cribado_nucleo.py preparar/cerrar` probado con datos sintéticos. Sin correr: faltan las exportaciones originales y las decisiones por registro del precribado |
| 21 | DOI adjudicados y aplicados a registros (10) | P | Pares adjudicados (Crossref y doi.org) y alias en `procesar_exportaciones.py`. Las 13 exportaciones no están en este equipo (búsqueda por nombre y sha256 el 22/11); 6.322 sigue siendo esperado, no ejecutado |
| 22 | Extracción/calidad de cada incluido (10–12) | C | 22/11: nueve fichas de 13 campos (siete candidatas y dos alternativas) con página y fragmento literal del PDF; `cotejar_fichas.py` confirma 229 de 229 (`cotejo_fichas.json`, `FICHAS_COTEJADAS.md`). Calidad: repeticiones, selección por test y dificultad compartida, en el capítulo de literatura |
| 23 | Acuerdo de revisores si hay varios (10) | NA | Sin dos cribados completos independientes; no inventar kappa |
| 24 | Tres búsquedas transversales ejecutadas (10) | P | Q5 sin exportación (fallo de sesión institucional el 18/11; conteos piloto 38/14) y Q6 sin ejecutar; no se reabren. Justificación y riesgo de omisión en el capítulo de literatura; requisito no cumplido |
| 25 | Tabla de mecanismos/fuentes/transferibilidad (10) | C | Literatura: mecanismos e hipótesis con riesgos |
| 26 | Número de redes y cuatro familias (11) | C | Diez totales aceptados por profesor según usuario; CNN/Transformer/híbrido/SSM |
| 27 | Puntuación completa, citas/cuartiles y exclusiones (11) | P | 22/11: rúbrica de seis componentes aplicada a nueve candidatas con evidencia por nivel; citas OpenAlex (misma fecha), licencias y pesos por la API de GitHub, rango CORE de las conferencias; óptimo con cuatro familias = selección operativa (`puntuar_candidatos.py`, `PUNTUACION_DEFINITIVA.md`). Falta el cuartil de IEEE TMI y ACM TOMM (SCImago no automatizable) |
| 28 | Referencias, diagramas, matemáticas y adaptaciones (11) | C | Diez fichas, diez esquemas, DOI/BibTeX; compactas identificadas como adaptaciones |
| 29 | Datasets/resultados/varianza/costo originales (11–12) | C | 20/11, siete redes nuevas: `RESULTADOS_ORIGINALES_7REDES.md`, dataset, métrica, cifra, varianza y costo con tabla y página del PDF de arXiv; `cotejar_resultados_originales.py` confirma 21 de 21 (`cotejo_resultados_originales.json`). Corrige seis cifras de las fichas del 18/11 |
| 30 | Reproducibilidad de literatura y gap refutable (12) | P | Scripts de literatura versionados y reproducibles (cotejo, bibliometría, puntuación). Gap refutable redactado; su denominador sale de `cribado_nucleo.py cerrar`, pendiente de las exportaciones |
| 31 | Paper guía justificado y documentado (16) | C | SegFormer-B0; condiciones originales separadas de HAZMAT |
| 32 | Réplica original y discrepancia/IC (16–17) | C | 20/11, Colab T4: checkpoint oficial de SegFormer-B0 en ADE20K val (2.000), escala única, mIoU 37,43 % (IC95 35,47–38,30) frente a 37,4; D = +0,03 (IC95 −1,93 a +0,90); veredicto a priori `compatible`; desviaciones D1 (−1,49 con el preprocesador de HF) y D2 (media por imagen). Réplica de evaluación, no de entrenamiento. `resultados/replica_segformer/LECTURA.md` |
| 33 | Limitación cuantificada y piloto ≥3 redes (17) | C | 20/11: protocolo a priori `config/limitacion_segformer_area.json`, terciles de área en los 122 grupos de validación, SegFormer frente a 3 redes piloto (U-Net, DeepLabv3+, SegNeXt-T), 5 semillas. Veredicto `no_confirmada`: Δ T3−T1 de SegFormer 0,082 [−0,030; 0,194] y sin diferencia con las piloto (p Holm ≥ 0,55); déficit de nivel, −0,12 frente a U-Net en el tercil pequeño. Dice por imagen extraído de los 50 checkpoints (`colab/Extraer_Validacion_E2.ipynb`). `resultados/limitacion_segformer_area/RESUMEN.md` |

## Implementación y rigor experimental — rúbrica 25 %

| ID | Requisito (página guía) | Estado | Evidencia / cierre necesario |
|---|---|---|---|
| 34 | Diez implementaciones y fuentes fijadas (12–14) | C | Adaptadores reales, SMP fijado, fuentes con commit/hash |
| 35 | ≥5 ejecuciones/red y registros (13) | C | 50 corridas auditadas; pruebas de rangos actuales usan 122 grupos, no 5 pares |
| 36 | Datos/resolución/pérdida base/parada comparables (12–15) | C | Pipeline común; auxiliares PIDNet y recetas distintas explícitos |
| 37 | Búsqueda equitativa de hiperparámetros (13,46–47) | C | 21/11, L4, exploratoria: búsqueda aleatoria T = 4 con el mismo espacio relativo (tasa × 10^u común, weight decay) y el mismo presupuesto (10 épocas, semilla 11) para las 10, elección por validación, 40 configuraciones registradas. La receta gana solo en VM-UNet; SegFormer-B0 sube de 0,660 a 0,782; las dos primeras no cambian. Sin la fase de 5 semillas × 30 épocas (diseñada, ≈ 17 h). `resultados/complemento_entreno_e2/`, capítulo `entrenamiento_complementario_e2` |
| 38 | Arquitectura indefinida: 1000 + finalistas (52–58) | P | Siete variantes publicadas definidas; justificar herencia de compactas locales. No afirmar búsqueda realizada |
| 39 | Tres E1 y baseline trivial adicional (11,22) | C | Tres compactas reentrenadas E2 y máscara vacía evaluada |
| 40 | Clásico fuerte no profundo ajustado (11,46) | C | `scripts/baseline_clasico_e2.py`: HistGradientBoosting, 13 rasgos, 3 configuraciones × 5 semillas, elección por validación; Dice test 0,645 (IC95 grupos 0,605–0,682) en `resultados/baseline_clasico_e2/` |
| 41 | Sanidad de aprendizaje y preflight (46) | C | Diez preflight GPU y entrenamiento completo; 21/11, sobreajuste a un lote de 4 imágenes (500 pasos): 9 de 10 pasan (Dice ≥ 0,90); VM-UNet 0,750, falla en la imagen de rótulo más pequeño. `resultados/complemento_entreno_e2/`, capítulo `entrenamiento_complementario_e2` |
| 42 | Curvas train y validación (13) | C | 50 curvas; pérdida train y Dice val no se restan |
| 43 | Configuración, versiones, semillas/hardware (13,41) | C | Freeze/config; L4 train/T4 test; determinismo warn_only |
| 44 | Parámetros, FLOPs, memoria/latencia homogénea (13,48) | P | 22/11: 9 de 10 redes medidas en CPU homogénea (256×256, batch 1, 1 y 6 hilos, 5 de calentamiento y 30 repeticiones, mediana e IQR): parámetros, FLOPs (FlopCounterMode, cota inferior), memoria (pico de proceso e incremento en inferencia) y latencia. Los FLOPs no ordenan la latencia (PIDNet-S, SegNeXt-T y SegFormer-B0: 3,1–3,5 G, 90–216 ms con 1 hilo); TransUNet 77 G y 2,1 s. CPU, no L4/T4; equipo compartido. VM-UNet exige CUDA: solo parámetros; medición en GPU preparada (`COLAB_MEDIR_COSTO.md`). `scripts/medir_costo_e2.py`, `resultados/costo_e2/` |
| 45 | Costo total/energía cuando sea posible (48) | C | 22/11: costo de GPU reconstruido de lo registrado: 33,3 h (entrenamiento 16,4 h; diagnósticos 1,7 h; complemento 15,2 h), todas en L4. Energía solo como cota superior TDP×horas ≈ 2,4 kWh de GPU (no medida; sin CPU ni centro de datos). Sin registro: test (T4), réplica, tiempo de sesión: el total es un mínimo. `scripts/costo_total_e2.py`, `resultados/costo_e2/costo_total_e2.md`, `LECTURA.md` |
| 46 | Auditoría completa de fuga (44–46) | C | Integridad, partición, adversarial, embeddings y casi duplicados (ID 11) y, 21/11, permutación de máscaras del pipeline (ID 56): p = 0,005 |
| 47 | Test nunca usado para decisiones y una evaluación (4,63) | D | E2 con pesos/umbral cerrados; test ya expuesto en E1 |

## Benchmark e inferencia — rúbrica 20 %

| ID | Requisito (página guía) | Estado | Evidencia / cierre necesario |
|---|---|---|---|
| 48 | Métricas por caso/corrida, media/DE/IC (15,17) | C | 10900 registros; cuatro métricas, IC grupos y DE semillas separados |
| 49 | Jerarquía, pareamiento y efectos (50–51) | C | Grupo como bloque; IC de diferencias y d_z; no pseudorreplicación de píxeles |
| 50 | Tres procedimientos y multiplicidad (23,51) | C | Bootstrap/intercambio, Wilcoxon-Holm, Friedman/Nemenyi en test. 22/11, validación: t, intercambio, Wilcoxon y signo con Holm(9), y max-T por cambio de signo con IC95 simultáneos (respeta la correlación entre los nueve contrastes): max-T coincide con t-Holm; Wilcoxon y signo difieren en TransUNet (no rechazan: ventaja concentrada en pocos grupos) y en DeepLabv3+ y VM-UNet (rechazan: pérdidas pequeñas y frecuentes). Familia fijada después del test: la desviación temporal queda en B12 (D). `scripts/inferencia_cierre_e2.py`, `resultados/inferencia_cierre_e2/` |
| 51 | Supuestos y sensibilidad (18,51) | C | 22/11, validación: simetría de las diferencias por grupo (bootstrap simetrizado, estadístico MGG) rechazada en 4 de 9 contrastes con FCN; ahí Wilcoxon no es un contraste de media ni de mediana, y la prueba del signo (sin simetría) da las mismas decisiones. Sensibilidad: t, intercambio, Wilcoxon y signo; dejar un grupo y una semilla fuera (solo DeepLabv3+ cambia de decisión); autocorrelación entre grupos vecinos ≈ 0 (p Holm = 1). En test, t, intercambio y Wilcoxon coinciden en 8 de 9. La independencia entre grupos sigue siendo aproximada: se comprobó, no se certifica. `resultados/inferencia_cierre_e2/` |
| 52 | Ranking/CD/Pareto (17–18) | C | Rangos grupos/semillas; CD Nemenyi; Pareto puntual Dice/parámetros |
| 53 | Descomposición de varianza/modelo mixto (50) | C | 22/11, validación (declarada): modelo mixto `lme4` sobre 10.900 Dice imagen x corrida con red fija y semilla, corrida, grupo, imagen y red x grupo aleatorios. Del total: grupo 64,0 % (IC95 boot 53,0–72,2 %), red x grupo 15,1 %, residuo 15,2 %, red 3,8 %, corrida 0,6 %, semilla común 0,07 % (LRT p = 0,15). Residuos asimétricos y de colas pesadas: el reparto se mantiene sobre rangos (INT) y sin los 96 Dice = 0. `scripts/varianza_mixta_val_e2.R`, `resultados/inferencia_cierre_e2/` |
| 54 | Equivalencia/relevancia δ (48) | C | 22/11, TOST δ = 0,02: test (t pareado exacto desde media, d_z y n = 122) TransUNet − FCN 0,0091, IC90 [−0,0061; 0,0242], p = 0,12: no equivalentes (ni superior); U-Net − FCN −0,0072, IC90 [−0,0148; 0,0005], p Holm(9) = 0,028: equivalentes. Validación (t y bootstrap): U-Net ≈ FCN equivalentes; ninguna red equivalente a TransUNet (RQ3). Sin bootstrap en test: faltan los grupos fuera del repositorio. `scripts/inferencia_cierre_e2.py`, `resultados/inferencia_cierre_e2/` |
| 55 | Reproducción independiente/discrepancia (41,63) | P | Reagregación desde registros: 0. 20/11, entorno limpio (`scripts/prueba_entorno_limpio_e2.py`): lo que corre con el repositorio coincide (máx. 0,0014); la reproducción completa necesita los resultados del test y las corridas, que están fuera del repositorio (`resultados/entorno_limpio_e2/LECTURA.md`) |

## Ruta F4b y diagnósticos — rúbrica 15 %

| ID | Requisito (página guía) | Estado | Evidencia / cierre necesario |
|---|---|---|---|
| 56 | Adecuación frente a trivial/permutación (23,36) | C | Dice vacío = 0. 21/11: permutación de máscaras en train del pipeline completo, FCN, B = 200: Dice val real 0,823 frente a nula media 0,465 (máx. 0,592), p = 1/201 = 0,005. La de TransUNet (B = 19) no se corrió. `resultados/complemento_entreno_e2/`, capítulo `entrenamiento_complementario_e2` |
| 57 | Brecha train/val con margen de sobreajuste (25,36) | C | 20/11, L4, exploratorio (`colab/Complemento_GPU_E2.ipynb`, etapa `clean`): reproduce el Dice de test cerrado (dif. máx. < 0,0001); brecha train−val positiva en las 10, de 0,026 (SegNet, IC con 0) a 0,213 (SegFormer) y 0,147 (SegNeXt). `resultados/diagnosticos_gpu_e2/TABLAS.md`, capítulo `diagnosticos_e2` |
| 58 | Brier/ECE fijo y adaptativo/calibración (15,49) | C | 20/11: Brier, NLL, ECE15 fijo y adaptativo, con y sin temperatura ajustada en validación; ECE15 0,008–0,014; temperatura 1,6–2,6 en todas (sobreconfianza); la temperatura baja el Brier y sube levemente el ECE15 |
| 59 | Robustez con ≥3 severidades (16) | C | 20/11: 10 redes × 5 semillas × ruido/desenfoque/brillo × 3 niveles en test. Área normalizada: redes grandes 0,92–0,96 con ruido y desenfoque; FCN, U-Net y SegNet 0,68–0,77 en las tres perturbaciones |
| 60 | Explicabilidad top3 + fidelidad/estabilidad (16) | C | 20/11: Integrated Gradients en 100 casos de validación del top 3 × 5 semillas; IG supera al borrado aleatorio en el 89 % (TransUNet) y 100 % (U-Net, FCN) de los casos; Spearman entre semillas 0,73 / 0,92 / 0,92 |
| 61 | Ablación ≥3 componentes, mismas semillas (16) | C | 21/11: TransUNet, 3 componentes × 5 semillas pareadas, referencia del benchmark: sin saltos −0,092 [−0,105; −0,079], BCE sola −0,037 [−0,043; −0,031] (p Holm < 0,001); sin Transformer +0,007 [−0,001; 0,015], p Holm 0,065. `resultados/complemento_entreno_e2/`, capítulo `entrenamiento_complementario_e2` |
| 62 | Incertidumbre predictiva top3/cobertura (49) | C | 20/11: ensamble de 5 semillas del top 3: +0,021 a +0,028 de Dice (IC sin 0; TransUNet 0,883); predicción conforme Mondrian por clase calibrada en validación, cobertura test 0,903–0,912 (nominal 0,90), con 34–37 % de grupos bajo el nominal (garantía marginal) |
| 63 | Techo y γ (48–49) | C | 22/11: segunda anotación independiente de 50 imágenes de validación (una por grupo, por tercil de área): techo Dice 0,941 (IC95 0,928–0,953). TransUNet 0,866 en las mismas 50; γ 0,31 (val) y 0,27 (test). `scripts/techo_reanotacion_e2.py medir/gamma`, `scripts/tabla_techo_e2.py`, `resultados/techo_e2/` |
| 64 | HD95/ASSD y precisión/recall (36) | C | 20/11: HD95/ASSD y precisión/recall de las 5 semillas en test (etapa `clean`); HD95 de 17,1 px (TransUNet) a 34,3 (SegFormer, recall 0,688) |
| 65 | Taxonomía previa, ≥100 casos/modelo e IC (49) | P | 22/11, exploratorio: taxonomía de dos ejes mutuamente excluyentes fijada antes (forma del error A1–A6; condición de la imagen B1–B7); censo de errores de validación (Dice < 0,80), 188–482 casos imagen-semilla por red, frecuencias con IC95 por grupos y sensibilidad 0,70/0,90; errores compartidos: 34/218 imágenes fallan en ≥ 8 redes. Etiquetado automático; falta la revisión humana de 88 casos (`errores_atajos_e2.py revision`). `resultados/errores_atajos_e2/` (`LECTURA.md`), protocolo previo del 22/11 (`PROTOCOLO_ERRORES_ATAJOS_20261122.md`) |
| 66 | Errores compartidos/diversidad/ensemble (49–50) | C | Errores compartidos: correlación del Dice por grupo entre las 10 redes de 0,62 (TransUNet-SegFormer) a 0,94 (FCN-U-Net), mediana 0,80 (`resultados/integracion_e2/analisis.json`, `error_correlation`; capítulo `resultados_e2`). Ensamble: 5 semillas del top 3, +0,021 a +0,028 de Dice con IC sin 0 (ID 62, capítulo `diagnosticos_e2`). Ensamble entre redes distintas no ejecutado |
| 67 | Subgrupos, atajos y daño potencial (41,50) | P | 22/11, exploratorio: Dice por clase, área, borde, número de rótulos, luz, contraste y origen con IC95 por grupos y Holm (val; test descriptivo); caen el cuartil de área menor y el tercil de poca luz; omisión grave (recall < 0,5) 3,8–7,7 % en nueve redes y 20 % en SegFormer, mayor con poca luz. Faltan negativos (sintéticos, C2) y atajos: `colab/Atajos_Errores_E2.ipynb`, preparado sin correr. `resultados/errores_atajos_e2/` (`LECTURA.md`), protocolo previo del 22/11 (`PROTOCOLO_ERRORES_ATAJOS_20261122.md`) |

## Reproducibilidad, redacción y defensa — rúbrica 10 %

| ID | Requisito (página guía) | Estado | Evidencia / cierre necesario |
|---|---|---|---|
| 68 | Jupyter Book y celdas reproducibles (58,62) | P | Libro/scripts integrados; notebooks sin ejecución automática; prueba limpia pendiente |
| 69 | Un comando para tablas/figuras, entorno fijado (40) | C | `reproducir_entregable2.py`, `requirements-libro.txt`; requiere datos/resultados locales |
| 70 | Código completo para todos los requisitos (58) | C | Código de todos los requisitos ejecutados en el repositorio: diagnósticos GPU (`scripts/diagnosticos_gpu_e2.py`, filas 57-60, 62, 64), complemento de entrenamiento (`scripts/complemento_entreno_e2.py`, filas 37, 41, 56, 61), revisión humana (filas 12, 63); los cuadernos de Colab se generan con `scripts/crear_colab_*.py` |
| 71 | Datos/instrucciones/contrato (59) | C | 22/11: instrucciones de obtención y verificación reproducibles en `docs/DATA_CARD_HAZMAT13.md` §6 (enlaces, estructura VOC, `scripts/verificar_corpus_hazmat13.py`, particiones con SHA256 en `resultados/preentrenamiento/*.csv`). La descarga de la carpeta de Drive queda sujeta a autorización; licencia de imágenes: ver fila 08 |
| 72 | Resumen ≤250 palabras, discusión/RQ (62) | C | Índice y cierre E2 |
| 73 | BibTeX completo (41) | C | 20/11: `libro/referencias_e2.bib` con 21 entradas (10 arquitecturas en versión editorial si existe, 7 de aplicación, 4 de mecanismos), DOI verificados con `verificar_doi_registro.py`. Riesgo: `scripts/integrar_entregable2.py` (línea 186) lo reescribe con solo diez entradas |
| 74 | Lista apéndice B completa y honesta (63) | C | 21 filas siguientes, sin convertir P/F/D en aprobados |
| 75 | Presentación y defensa técnica (59) | P | Guion de 13 diapositivas y respuestas de defensa en `docs/PRESENTACION_E2.md` y material privado de preguntas; falta el ensayo |
| 76 | Publicación final | C | 21/11: rama `cierre-e2` integrada en `main` y libro compilado sin advertencias, publicado en GitHub Pages (`docs/`, https://luisperez03.github.io/Deep_Learning_IMO/) |

## Apéndice B: las 21 casillas originales

| Ítem | Estado | Evidencia o límite |
|---|---|---|
| B01 Preregistro anterior al entrenamiento | D | ID05; no retrofechar |
| B02 Hash dataset/test | C | Manifiestos y protocolo cerrado |
| B03 Auditoría de fuga completamente aprobada | C | ID46: fuga por embeddings y permutación de máscaras (p = 0,005) hechas |
| B04 Particiones/semillas regenerables | C | ID14 |
| B05 Pipeline y ajustes solo train | C | Normalización/pesos train; selección val |
| B06 Presupuesto igual/configuraciones | C | Recetas fijas en el benchmark; búsqueda equitativa exploratoria T = 4 (ID37) |
| B07 ≥5 ejecuciones; ≥6 para rangos por semillas | C | Cinco corridas; rangos usan grupos |
| B08 Potencia/R | D | ID06; planificación posterior |
| B09 Trivial/clásico/tres E1 | C | Máscara vacía, baseline clásico (ID40) y tres E1 |
| B10 Software/hardware/determinismo | C | ID43 |
| B11 Tablas A/B/C con pruebas/IC/efectos | C | Tablas con IC, pruebas y efectos (ID 48-49); ajuste y sobreajuste: brecha train-val (ID 57), sobreajuste a un lote (ID 41) y búsqueda equitativa (ID 37) |
| B12 Familias/multiplicidad a priori | D | Ampliación Holm posterior al test |
| B13 Techo/γ | C | ID63: techo 0,941 por reanotación independiente; γ por red |
| B14 Arquitecturas sin diseño y dos fases | P | ID38; compactas heredadas requieren justificación |
| B15 Selección val/equivalencia/sesgo ganador | C | Checkpoints por val. 22/11: TOST (ID54); sesgo del ganador en validación: TransUNet gana en el 99,9 % de los bootstrap, sesgo del nivel 0,0001 y optimismo por mitades ≈ 0; checkpoint por val frente a test: test − val +0,010 (8 de 10 redes suben), sin optimismo visible; TransUNet − FCN pasa de 0,022 (val) a 0,009 (test) |
| B16 Robustez/explicabilidad/fidelidad/estabilidad | C | ID59–60 ejecutados el 20/11 (exploratorio) |
| B17 Incertidumbre top3/cobertura | C | ID62 ejecutado el 20/11 (exploratorio) |
| B18 Test una vez, pesos/umbrales fijos | D | Cierre E2 sí, exposición histórica E1 |
| B19 Réplica/limitación paper | C | ID32 réplica compatible; ID33 limitación por área no confirmada (déficit de nivel) |
| B20 Reproducción independiente/discrepancia | P | Agregación 0; entorno limpio parcial (ID55) |
| B21 Extremo a extremo y entorno | P | Informe con un comando; experimentos faltantes |

## Orden de cierre para el domingo

1. Revisar evidencia integrada y compilar. No llamar definitivo al informe con requisitos centrales F.
2. Ejecutar complemento GPU con pesos congelados: calibración, robustez, explicabilidad, incertidumbre, bordes y costo. Protocolo antes de esa ejecución; carácter exploratorio explícito.
3. Definir presupuesto para réplica original, clásico fuerte y ablaciones. No seleccionar variantes usando test ni prometer cómputo no realizado.
4. Adjudicar bibliografía y anotación humana con trazabilidad; Q5/Q6 siguen omitidas.
5. Preparar defensa, reproducción limpia y revisión final. Solicitar autorización al llegar a publicación.
