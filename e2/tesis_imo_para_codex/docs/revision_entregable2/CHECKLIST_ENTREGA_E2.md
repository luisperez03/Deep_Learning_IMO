# Checklist maestro del Entregable 2

Revisado el 20/11/2026 contra la guía original de 65 páginas, incluida ruta F4b y apéndice B. Entrega prevista: domingo 22/11/2026. Este control no garantiza una nota.

**C:** cubierto con evidencia. **P:** parcial. **F:** falta ejecutar/verificar. **D:** desviación histórica no reparable retroactivamente. **NA:** no aplica con motivo. Un plan o un script no acreditan una ejecución. No convertir el número de filas en porcentaje de calificación.

El usuario confirmó que el profesor acepta **diez modelos totales: tres E1 y siete nuevos**. No se infieren dispensas de otros requisitos. Q5/Q6 siguen omitidas por instrucción explícita.

## Problema, hipótesis, datos y EDA — rúbrica 15 %

| ID | Requisito (página guía) | Estado | Evidencia / cierre necesario |
|---|---|---|---|
| 01 | Formulación entrada/salida/riesgo (2) | C | `metodos_e2.md`: segmentación binaria; no clasificación IMO |
| 02 | Relevancia y ≥3 referencias aplicadas (2) | P | Fuentes de procedencia disponibles; verificar y completar BibTeX de aplicación |
| 03 | Métrica, δ y unidad heredados (2) | C | Dice por grupo, δ=0,02; IoU/mIoU/accuracy; independencia aproximada |
| 04 | ≥3 RQ, H0/H1, α, efecto y decisión (2–3) | P | RQ1–4 integradas; temporalidad explícita; equivalencia no ejecutada |
| 05 | Preregistro completo anterior al entrenamiento (42) | D | Configuración/cierre antes de test disponibles; no preregistro integral previo acreditado. No retrofechar |
| 06 | Potencia y número de ejecuciones prospectivos (43) | D | Cinco semillas; MDD/R añadidos solo para planificación futura |
| 07 | Data Card, procedencia, versión/hash (3) | P | Tamaños, objetivo y manifiestos; origen HAZMAT13 probable, sin identidad externa certificada |
| 08 | Licencias, adquisición, geografía, ética (3,41) | P | Límites declarados; falta verificar licencia de imágenes externas y metadatos de adquisición |
| 09 | Distribución, desbalance y entropía (3–4) | P | Conteos/áreas/píxeles disponibles; completar entropía y análisis específicos restantes |
| 10 | Faltantes, corrupción y consistencia (4) | C | 102 máscaras no utilizables, 132 cuarentena, geometría y hashes |
| 11 | Duplicados y similitud residual (4,44) | P | Hashes/dHash/aHash; cero enlaces conocidos cruzados; falta sensibilidad por embeddings/vecinos |
| 12 | Ruido de etiqueta y acuerdo independiente (4,48) | F | Casos dirigidos no estiman prevalencia; requiere segunda anotación |
| 13 | Sesgos y atajos (4) | P | Origen XML y limitaciones documentados; falta prueba de atajos |
| 14 | Particiones y semillas regenerables (4) | C | 1015/218/218 imágenes, 567/122/122 grupos, semilla 20261106 |
| 15 | Shift / validación adversarial (4,45) | F | Clasificador train–test o alternativa, con inferencia por grupos |
| 16 | Geometría, canales, intensidad y formas (7–8) | P | Letterbox, tamaños, normalización train; completar histogramas por clase y formas |
| 17 | Tabla hallazgo–decisión–riesgo (9,61) | C | Tabla integrada al EDA |

## Literatura, paper guía y gap — rúbrica 15 %

| ID | Requisito (página guía) | Estado | Evidencia / cierre necesario |
|---|---|---|---|
| 18 | ≥2 bases, ecuaciones, filtros, fechas/conteos (9–10) | C | Scopus/WoS, protocolo, bitácora, 13 exportaciones Q0–Q4 |
| 19 | Inclusión/exclusión y excepciones antiguas (10) | C | Criterios documentados; aplicación final distinta de definición |
| 20 | PRISMA completo y exclusiones razonadas (10) | P | 9646 registros, 6325 grupos provisionales; precribado automático, sin elegibilidad final |
| 21 | DOI adjudicados y aplicados a registros (10) | F | Tres pares ambiguos; falta correspondencia y conteo definitivo |
| 22 | Extracción/calidad de cada incluido (10–12) | P | Siete fichas preliminares; falta cotejo completo de tablas originales |
| 23 | Acuerdo de revisores si hay varios (10) | NA | Sin dos cribados completos independientes; no inventar kappa |
| 24 | Tres búsquedas transversales ejecutadas (10) | P | Q5/Q6 omitidas; fuentes suplementarias no sustituyen consultas |
| 25 | Tabla de mecanismos/fuentes/transferibilidad (10) | C | Literatura: mecanismos e hipótesis con riesgos |
| 26 | Número de redes y cuatro familias (11) | C | Diez totales aceptados por profesor según usuario; CNN/Transformer/híbrido/SSM |
| 27 | Puntuación completa, citas/cuartiles y exclusiones (11) | P | Lista operativa; bibliometría y rúbrica definitiva sin cerrar |
| 28 | Referencias, diagramas, matemáticas y adaptaciones (11) | C | Diez fichas, diez esquemas, DOI/BibTeX; compactas identificadas como adaptaciones |
| 29 | Datasets/resultados/varianza/costo originales (11–12) | P | Fichas preliminares; faltan verificaciones numéricas completas |
| 30 | Reproducibilidad de literatura y gap refutable (12) | P | Gap candidato; denominador de estudios incluidos no cerrado |
| 31 | Paper guía justificado y documentado (16) | C | SegFormer-B0; condiciones originales separadas de HAZMAT |
| 32 | Réplica original y discrepancia/IC (16–17) | F | ADE20K/escala única/checkpoint original no ejecutados |
| 33 | Limitación cuantificada y piloto ≥3 redes (17) | F | Estratos exploratorios no sustituyen piloto específico preregistrado |

## Implementación y rigor experimental — rúbrica 25 %

| ID | Requisito (página guía) | Estado | Evidencia / cierre necesario |
|---|---|---|---|
| 34 | Diez implementaciones y fuentes fijadas (12–14) | C | Adaptadores reales, SMP fijado, fuentes con commit/hash |
| 35 | ≥5 ejecuciones/red y registros (13) | C | 50 corridas auditadas; pruebas de rangos actuales usan 122 grupos, no 5 pares |
| 36 | Datos/resolución/pérdida base/parada comparables (12–15) | C | Pipeline común; auxiliares PIDNet y recetas distintas explícitos |
| 37 | Búsqueda equitativa de hiperparámetros (13,46–47) | F | Una receta fija por modelo; no búsqueda sistemática ni exención E2 acreditada |
| 38 | Arquitectura indefinida: 1000 + finalistas (52–58) | P | Siete variantes publicadas definidas; justificar herencia de compactas locales. No afirmar búsqueda realizada |
| 39 | Tres E1 y baseline trivial adicional (11,22) | C | Tres compactas reentrenadas E2 y máscara vacía evaluada |
| 40 | Clásico fuerte no profundo ajustado (11,46) | F | FCN es profundo. Falta clásico train/val con presupuesto explícito |
| 41 | Sanidad de aprendizaje y preflight (46) | P | Diez preflight GPU y entrenamiento completo; falta sobreajuste deliberado a lote/permutación |
| 42 | Curvas train y validación (13) | C | 50 curvas; pérdida train y Dice val no se restan |
| 43 | Configuración, versiones, semillas/hardware (13,41) | C | Freeze/config; L4 train/T4 test; determinismo warn_only |
| 44 | Parámetros, FLOPs, memoria/latencia homogénea (13,48) | P | Parámetros y horas; FLOPs, pico y latencia pura pendientes |
| 45 | Costo total/energía cuando sea posible (48) | P | Horas de épocas; excluyen interrupciones/E/S; no inventar energía |
| 46 | Auditoría completa de fuga (44–46) | P | Integridad/partición sí; adversarial y etiquetas aleatorias pendientes |
| 47 | Test nunca usado para decisiones y una evaluación (4,63) | D | E2 con pesos/umbral cerrados; test ya expuesto en E1 |

## Benchmark e inferencia — rúbrica 20 %

| ID | Requisito (página guía) | Estado | Evidencia / cierre necesario |
|---|---|---|---|
| 48 | Métricas por caso/corrida, media/DE/IC (15,17) | C | 10900 registros; cuatro métricas, IC grupos y DE semillas separados |
| 49 | Jerarquía, pareamiento y efectos (50–51) | C | Grupo como bloque; IC de diferencias y d_z; no pseudorreplicación de píxeles |
| 50 | Tres procedimientos y multiplicidad (23,51) | P | Bootstrap/intercambio, Wilcoxon-Holm, Friedman/Nemenyi; ampliación posterior explícita |
| 51 | Supuestos y sensibilidad (18,51) | P | Shapiro/estimandos documentados; independencia aproximada y simetría no certificadas |
| 52 | Ranking/CD/Pareto (17–18) | C | Rangos grupos/semillas; CD Nemenyi; Pareto puntual Dice/parámetros |
| 53 | Descomposición de varianza/modelo mixto (50) | P | DE semillas y correlación disponibles; falta descomposición formal con diagnóstico |
| 54 | Equivalencia/relevancia δ (48) | P | δ interpretado; no equivalencia demostrada ni TOST ejecutado |
| 55 | Reproducción independiente/discrepancia (41,63) | P | Reagregación desde registros: 0; no nueva inferencia independiente |

## Ruta F4b y diagnósticos — rúbrica 15 %

| ID | Requisito (página guía) | Estado | Evidencia / cierre necesario |
|---|---|---|---|
| 56 | Adecuación frente a trivial/permutación (23,36) | P | Dice vacío=0; falta sanidad y prueba completa de adecuación |
| 57 | Brecha train/val con margen de sobreajuste (25,36) | F | No Dice train en eval ni δg previo; curvas no certifican ausencia de sobreajuste |
| 58 | Brier/ECE fijo y adaptativo/calibración (15,49) | F | Faltan probabilidades; máscaras umbralizadas no bastan |
| 59 | Robustez con ≥3 severidades (16) | F | Corrupciones y curvas no ejecutadas |
| 60 | Explicabilidad top3 + fidelidad/estabilidad (16) | F | Paneles de máscaras no son atribuciones; falta técnica y validación |
| 61 | Ablación ≥3 componentes, mismas semillas (16) | F | Requiere entrenamientos separados; no renombrar redes como ablaciones |
| 62 | Incertidumbre predictiva top3/cobertura (49) | F | IC del score no es cobertura predictiva; falta calibración y conjuntos/intervalos |
| 63 | Techo y γ (48–49) | F | Sin doble anotación ni cota empírica. Dice 1 no es techo estimado |
| 64 | HD95/ASSD y precisión/recall (36) | F | Guardadas máscaras de una semilla; falta análisis completo de cinco semillas |
| 65 | Taxonomía previa, ≥100 casos/modelo e IC (49) | P | Panel dirigido y estratos; sin etiquetado sistemático de 100 errores/modelo |
| 66 | Errores compartidos/diversidad/ensemble (49–50) | P | Matriz de correlación disponible; ensemble y ganancia pendientes |
| 67 | Subgrupos, atajos y daño potencial (41,50) | P | Cuartiles de área con IC descriptivos; faltan negativos/validación operacional |

## Reproducibilidad, redacción y defensa — rúbrica 10 %

| ID | Requisito (página guía) | Estado | Evidencia / cierre necesario |
|---|---|---|---|
| 68 | Jupyter Book y celdas reproducibles (58,62) | P | Libro/scripts integrados; notebooks sin ejecución automática; prueba limpia pendiente |
| 69 | Un comando para tablas/figuras, entorno fijado (40) | C | `reproducir_entregable2.py`, `requirements-libro.txt`; requiere datos/resultados locales |
| 70 | Código completo para todos los requisitos (58) | P | Benchmark disponible; diagnósticos F siguen sin evidencia ejecutada |
| 71 | Datos/instrucciones/contrato (59) | P | Manifiestos y fuente probable; verificar obtención/licencia externa |
| 72 | Resumen ≤250 palabras, discusión/RQ (62) | C | Índice y cierre E2 |
| 73 | BibTeX completo (41) | P | Diez arquitecturas; completar aplicación/mecanismos |
| 74 | Lista apéndice B completa y honesta (63) | C | 21 filas siguientes, sin convertir P/F/D en aprobados |
| 75 | Presentación y defensa técnica (59) | F | Preparar diapositivas/guion y ensayar |
| 76 | Publicación final | F | Solicitar autorización antes de GitHub/despliegue; Daniel puede tener cambios |

## Apéndice B: las 21 casillas originales

| Ítem | Estado | Evidencia o límite |
|---|---|---|
| B01 Preregistro anterior al entrenamiento | D | ID05; no retrofechar |
| B02 Hash dataset/test | C | Manifiestos y protocolo cerrado |
| B03 Auditoría de fuga completamente aprobada | P | ID46; integridad no equivale a todas las pruebas |
| B04 Particiones/semillas regenerables | C | ID14 |
| B05 Pipeline y ajustes solo train | C | Normalización/pesos train; selección val |
| B06 Presupuesto igual/configuraciones | P | Recetas fijas, no búsqueda ID37 |
| B07 ≥5 ejecuciones; ≥6 para rangos por semillas | C | Cinco corridas; rangos usan grupos |
| B08 Potencia/R | D | ID06; planificación posterior |
| B09 Trivial/clásico/tres E1 | P | Falta clásico fuerte |
| B10 Software/hardware/determinismo | C | ID43 |
| B11 Tablas A/B/C con pruebas/IC/efectos | P | Comparación avanzada; ajuste/sobreajuste pendientes |
| B12 Familias/multiplicidad a priori | D | Ampliación Holm posterior al test |
| B13 Techo/γ | F | ID63 |
| B14 Arquitecturas sin diseño y dos fases | P | ID38; compactas heredadas requieren justificación |
| B15 Selección val/equivalencia/sesgo ganador | P | Checkpoints por val; equivalencia no demostrada |
| B16 Robustez/explicabilidad/fidelidad/estabilidad | F | ID59–60 |
| B17 Incertidumbre top3/cobertura | F | ID62 |
| B18 Test una vez, pesos/umbrales fijos | D | Cierre E2 sí, exposición histórica E1 |
| B19 Réplica/limitación paper | F | ID32–33 |
| B20 Reproducción independiente/discrepancia | P | Agregación 0, no inferencia nueva |
| B21 Extremo a extremo y entorno | P | Informe con un comando; experimentos faltantes |

## Orden de cierre para el domingo

1. Revisar evidencia integrada y compilar. No llamar definitivo al informe con requisitos centrales F.
2. Ejecutar complemento GPU con pesos congelados: calibración, robustez, explicabilidad, incertidumbre, bordes y costo. Protocolo antes de esa ejecución; carácter exploratorio explícito.
3. Definir presupuesto para réplica original, clásico fuerte y ablaciones. No seleccionar variantes usando test ni prometer cómputo no realizado.
4. Adjudicar bibliografía y anotación humana con trazabilidad; Q5/Q6 siguen omitidas.
5. Preparar defensa, reproducción limpia y revisión final. Solicitar autorización al llegar a publicación.
