# Avance del cierre del Entregable 2 — 20 al 26/11/2026

Trabajo posterior al commit `8395ea5` (20/11), en la rama `cierre-e2`, copiada a `main` en cada publicación. El estado
fila por fila está en `docs/revision_entregable2/CHECKLIST_ENTREGA_E2.md`. Tras la revisión editorial del 26/11 quedan parciales las filas 65
(registro individual de la revisión humana de errores) y 75 (ensayo de sustentación). Los análisis posteriores al test son exploratorios: no cambian la selección original ni
los protocolos cerrados.

## Búsqueda de arquitectura de las compactas (fila 38 y B14), terminada el 26/11

La sección 14 del enunciado pide, para las redes sin arquitectura definida en su artículo (FCN, U-Net y SegNet), 1.000
arquitecturas en la fase 1 y los finalistas con varias semillas en la fase 2. El protocolo se fijó antes de correr
(`docs/PROTOCOLO_BUSQUEDA_ARQUITECTURA_E2.md`) y se ejecutó completo en `colab/Busqueda_Arquitectura_E2.ipynb`, en L4.
Solo validación; el test no se cargó.

| Análisis | Resultado | Evidencia |
|---|---|---|
| Ejecución | Fase 1: 1.000 por familia más 40 repeticiones (3.120), a 128 px y 6 épocas. k = 10 (σ_semilla/σ_arquitectura ≤ 1). Fase 2: 10 finalistas, referencia (64, 64, 64)/(0, 1, 0) y heredada (16, 32, 64) por familia, 5 semillas, 180 corridas. 243,7 h de L4 sumadas por corrida, unas 61 h de reloj con cuatro procesos | `resultados/busqueda_arquitectura_e2/COSTO.json` |
| Ganadora por la regla | La referencia (64, 64, 64)/(0, 1, 0) en las tres: el empate con la mejor abarca a todos los finalistas y gana la de menos parámetros. Dice val FCN 0,851, U-Net 0,873, SegNet 0,861. La de mayor media gana 0,009–0,013 más con 6 a 11 veces los parámetros | `resultados/busqueda_arquitectura_e2/LECTURA.md` |
| Frente a la heredada | FCN +0,020 [0,011; 0,028], U-Net +0,033 [0,026; 0,041], SegNet +0,039 [0,020; 0,059]. Por la lectura fijada antes, el benchmark subestimó U-Net y SegNet. En FCN la heredada reproduce el Dice de validación del benchmark (0,8309); en U-Net y SegNet la tasa menor (0,0003) explica 0,008 y 0,015 | `resultados/busqueda_arquitectura_e2/TABLA_RESUMEN.md` |
| Fase 1 frente a fase 2 | Spearman en los finalistas: FCN −0,07, U-Net −0,96, SegNet 0,39. La fase 1 a fidelidad reducida cribó, pero no ordenó a los finalistas | `libro/busqueda_arquitectura_e2.md` |
| Espacio | Los anchos explican R² 0,39–0,57; pesa sobre todo el primer bloque en U-Net y SegNet; dropout con efecto de milésimas | `scripts/lectura_busqueda_arquitectura_e2.py` |

## Datos

| Análisis | Resultado | Evidencia |
|---|---|---|
| Data Card y procedencia (fila 07) | Fuente mrl-amrl/HAZMAT13 (commit `ec85498`) y artículo doi:10.1016/j.mlwa.2021.100104; coinciden 1.685 imágenes, 13 clases y formato VOC; SHA256 de 2.902 de 2.902 archivos verificados | `docs/DATA_CARD_HAZMAT13.md`, `scripts/verificar_corpus_hazmat13.py` |
| Licencia y adquisición (fila 08) | Licencia MIT del repositorio, que solo nombra «Software»; sin licencia de datos, titularidad de las fotos ni geografía. Limitación declarada; sin redistribución de imágenes ni pesos | `docs/DATA_CARD_HAZMAT13.md` |
| Ruido de etiqueta (fila 12) | 10 de 100 imágenes con rótulo cortado por el borde sin máscara (IC95 Wilson 5,5–17,4 %), todos del tipo falta | `resultados/ruido_etiqueta_e2/` |
| Techo por reanotación (fila 63) | Segunda anotación independiente de 50 imágenes de validación: techo Dice 0,941 (IC95 0,928–0,953); TransUNet 0,866 en las mismas 50; γ 0,31 (val) y 0,27 (test) | `resultados/techo_e2/`, `scripts/techo_reanotacion_e2.py` |
| Fuga por permutación (filas 46 y 56) | Máscaras permutadas en train, pipeline completo, FCN, B = 200: Dice val real 0,823 frente a nula media 0,465 (máx. 0,592), p = 0,005 | `resultados/complemento_entreno_e2/` |

## Inferencia

| Análisis | Resultado | Evidencia |
|---|---|---|
| Equivalencia (filas 04 y 54) | TOST δ = 0,02 en test con 122 grupos, t y bootstrap, Holm(9). TransUNet − FCN 0,0091, IC90 [−0,0061; 0,0242], p = 0,12: indeterminado. U-Net − FCN −0,0072, p Holm 0,028: equivalentes, pero depende de la semilla 53 (sin ella, p = 0,062). No se demostró equivalencia de ninguna red con TransUNet | `libro/resultados_e2.md` |
| Supuestos y sensibilidad (fila 51) | Simetría de diferencias rechazada en 6 de 9 contrastes con FCN en test; donde se rechaza, la prueba del signo decide igual que Wilcoxon; los cuatro procedimientos coinciden en 8 de 9 | `libro/resultados_e2.md` |
| Descomposición de varianza (fila 53) | Modelo mixto (`lme4`) sobre 10.900 Dice de validación: grupo 64,0 % (IC95 53,0–72,2), red × grupo 15,1 %, residuo 15,2 %, red 3,8 %, corrida 0,6 %, semilla común 0,07 % | `libro/resultados_e2.md` |

## Literatura

| Análisis | Resultado | Evidencia |
|---|---|---|
| Cribado con exportaciones reales (filas 20 y 21) | Scopus y WoS, Q0–Q6: 9.715 registros, 6.366 grupos, 3.349 duplicados, 0 coincidencias ambiguas; núcleo de dominio de 25 revisado uno por uno (10 incluidos); muestra a ciegas de 200: el filtro pierde 37 % [26–49] de incluibles en su estrato | `docs/revision_entregable2/PRISMA_FINAL.md`, `docs/revision_entregable2/cribado_nucleo.py` y `conciliacion.json` |
| Búsquedas transversales (fila 24) | Q5 y Q6 reabiertas y corridas en las dos bases (Q5 38 + 14; Q6 11 + 6); con Q4, tres transversales | `docs/revision_entregable2/BITACORA_BUSQUEDAS.md` |
| Fichas y puntuación (filas 22 y 27) | Nueve fichas de 13 campos con página y fragmento literal; cotejo 229 de 229; rúbrica de seis componentes con citas OpenAlex, cuartiles y licencias | `docs/revision_entregable2/FICHAS_COTEJADAS.md` y `puntuar_candidatos.py` |
| Gap refutable (fila 30) | Denominador de 13 estudios de dominio; ninguno segmenta el rótulo por píxel con red ni compara arquitecturas con semillas | `libro/literatura.md` |
| Resultados originales (fila 29) | Dataset, métrica, cifra, varianza y costo de las siete redes nuevas con página del PDF; cotejo 21 de 21 | `docs/revision_entregable2/RESULTADOS_ORIGINALES_7REDES.md` |

## Réplica, entrenamiento y diagnósticos

| Análisis | Resultado | Evidencia |
|---|---|---|
| Réplica de SegFormer-B0 (fila 32) | ADE20K val, checkpoint oficial: mIoU 37,43 % (IC95 35,47–38,30) frente a 37,4 publicado; veredicto a priori `compatible` | `resultados/replica_segformer/` |
| Limitación por área de SegFormer (fila 33) | No confirmada: Δ T3−T1 0,082 [−0,030; 0,194], sin diferencia con tres redes piloto; déficit de nivel, no de área | `resultados/limitacion_segformer_area/` |
| Sanidad (fila 41) | Sobreajuste a un lote de 4 imágenes: 9 de 10 redes pasan; VM-UNet 0,750 | `resultados/complemento_entreno_e2/` |
| Búsqueda equitativa de hiperparámetros (fila 37) | T = 4, mismo espacio y presupuesto para las 10: la receta gana solo en VM-UNet; SegFormer-B0 sube de 0,660 a 0,782; las dos primeras no cambian | `resultados/complemento_entreno_e2/` |
| Ablación de TransUNet (fila 61) | Sin saltos −0,092 [−0,105; −0,079]; BCE sola −0,037; sin Transformer +0,007 (p Holm 0,065) | `libro/entrenamiento_complementario_e2.md` |
| Diagnósticos GPU (filas 57–60, 62, 64) | Brecha train−val de 0,026 a 0,213; ECE15 0,008–0,014; robustez con 3 severidades; Integrated Gradients; ensamble +0,021 a +0,028; conformal con cobertura 0,903–0,912; HD95 de 17,1 a 34,3 px | `resultados/diagnosticos_gpu_e2/TABLAS.md` |

## Errores, atajos y costo

| Análisis | Resultado | Evidencia |
|---|---|---|
| Atajos (fila 13) | Prueba contrafactual con 10 redes × 5 semillas: dependencia del contexto en TransUNet y SegFormer-B0; ninguna infiere el rótulo borrado; en gris todas caen 0,29–0,42 | `resultados/errores_atajos_e2/` |
| Taxonomía de errores (fila 65) | Censo de errores de validación con taxonomía previa de dos ejes; 188–482 casos por red; revisión cualitativa global de 88 paneles, sin registro individual; validación humana pendiente de completar | `scripts/errores_atajos_e2.py` |
| Subgrupos y daño (fila 67) | Caen el cuartil de área menor (6 de 10 redes) y la poca luz (7 de 10); omisión grave 3,8–7,7 % en nueve redes y 20 % en SegFormer | `resultados/errores_atajos_e2/` |
| Costo computacional (filas 44 y 45) | Parámetros, FLOPs, memoria y latencia de las 10 en GPU L4 (VM-UNet 19,5 GFLOPs y 32,7 ms); los FLOPs no ordenan la latencia. Subtotal registrado al 22/11: 33,3 h de L4; no incluye la búsqueda de arquitectura posterior. Esta suma 243,7 h por corrida concurrente, con unas 61 h de reloj estimadas en una sola L4; ambas unidades no deben sumarse como tiempo de ocupación de GPU | `scripts/costo_total_e2.py`, `colab/Medir_Costo_GPU_E2.ipynb` |

## Reproducción y publicación

| Análisis | Resultado | Evidencia |
|---|---|---|
| Reproducción en clon limpio (fila 55) | Tablas, contrastes y auditoría con discrepancia 0; recálculo independiente desde 2.180 máscaras con discrepancia 0 | `resultados/auditoria_test_e2/AUDITORIA.json`, `scripts/reproducir_entregable2.py` |
| Libro (filas 68 y 76) | `jupyter-book build -W` sin avisos; publicado en GitHub Pages el 21/11 y republicado el 24 y el 26/11 | `scripts/publicar_libro_docs.py` |

## Declarado como limitación

No hubo preregistro previo (05), el número de semillas no salió de un cálculo de potencia (06), el test ya se había
visto en el E1 (47), no hubo dos revisores en el cribado (23) y la licencia de las imágenes no está declarada (08).
Nada se retrofechó.
