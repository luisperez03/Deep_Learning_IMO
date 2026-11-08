# Alineación con la metodología del profesor

Documento de contraste: *Metodología para la obtención de máscaras y benchmarks de segmentación y clasificación. Reconocimiento de etiquetas IMO en contenedores*, archivo `MaskForIMOImagesLPerezThesis.pdf`, 20 páginas, SHA-256 `2342c4242b9bad1a6c9ce423df389a80f81cc48229058d6f8070e8cde387cad0`. Las páginas citadas son las numeradas en el PDF. Las valoraciones por estrellas de su tabla 1 son juicios del autor. No son observaciones del experimento.

## Alcance de las tres tareas

**Segmentación semántica binaria.** Cada píxel recibe etiqueta o fondo. Una imagen puede contener varios rótulos; si se tocan, la predicción puede fusionarlos. Los resultados actuales evalúan exclusivamente esta tarea.

**Segmentación de instancias.** La salida distinguiría una máscara por objeto, incluso con objetos contiguos. La propuesta principal del profesor añade una categoría IMO por instancia (pp. 4–5). Nuestra salida binaria no contiene identificadores, clases ni puntuaciones por objeto para calcular AP de máscaras por instancia.

**Clasificación.** El PDF propone identificar clase, división o variante a partir de un recorte rectificado, con opciones de rechazo y evaluación de propagación del error (pp. 6–7 y 15–16). Un rótulo bien delimitado puede recibir una clase incorrecta. La métrica de este libro no permite estimar ese riesgo.

## Contraste con evidencia

| Recomendación y páginas | Situación en Entregable 1 | Alcance o corrección |
|---|---|---|
| Comparación semántica sobre datos y protocolo comunes (4–5) | FCN, SegNet y U-Net pequeñas desde cero, misma resolución, pérdida, partición y regla de parada. Cinco semillas comunes frente a las tres sugeridas. | Cumplido para la referencia semántica binaria. U-Net pertenece a la lista del profesor. FCN y SegNet se justifican por su función comparativa en el curso. |
| Agrupar por adquisición y controlar duplicados (5–6, 16) | 748 enlaces de similitud se conservan dentro de grupos; ninguno cruza las particiones. | Cumplimiento parcial. Falta un identificador fiable de contenedor, puerto o sesión. La similitud protege frente a los enlaces detectados y no certifica independencia. |
| Máscaras revisadas y prueba humana separada (2–3, 5, 16) | Se usaron máscaras semánticas existentes. Se excluyeron 102 fotografías sin máscara y 132 pares en cuarentena. El EDA incluye inspección dirigida. | La inspección y los hashes no acreditan procedencia humana individual ni acuerdo. La sugerencia de 300–500 imágenes anotadas con cuatro esquinas pertenece a otra colección. |
| Calidad, eficiencia y robustez (5, 16) | Dice e IoU de etiqueta, mIoU, exactitud, parámetros, tiempos por época y equipo; ahora también mapas de errores. | Los tiempos son de entrenamiento/validación. Las figuras de borde son descriptivas; no se calculó Boundary IoU. Quedan sin medir latencia de inferencia comparable, memoria, FLOPs ni rendimiento por clima, oclusión o puerto. |
| Incertidumbre, comparación pareada y reserva del test (5, 13–16) | Selección por Dice de validación, test único por corrida, IC con grupos completos y Holm para los tres pares. El libro separa variación por semilla y por grupo. | Las conclusiones dependen de cinco semillas y de la independencia aproximada de grupos. La ventaja de FCN frente a U-Net queda por debajo de δ; no se declara equivalencia formal. |
| Rectificación, clases, rechazo y cadena completa (6–16) | Sin recortes de referencia/predichos ni taxonomía IMO verificada. | Corresponde a la tesis. Macro-F1 por clase, calibración, costos, McNemar sobre aciertos de clase y matrices de confusión entre divisiones requieren otra variable objetivo. |

## Justificaciones que necesita la defensa

Las máscaras binarias provienen de los archivos de segmentación, no de cajas convertidas a rectángulos ni de pseudoetiquetas. El artículo asociado al corpus hace probable la procedencia HAZMAT-13; la auditoría local acredita integridad de esta copia. La literatura abierta documentada no establece un orden de frecuencia de las tres redes en HAZMAT. El margen de 0,02 Dice fue una decisión metodológica previa; no se calibró contra daño portuario o desacuerdo humano. La prueba interna reserva grupos de este mismo corpus, que ya había sido explorado antes del reparto.

## Qué se traslada y qué se difiere

Para el **Entregable 2** se propone comparar modelos adicionales de segmentación semántica con FCN bajo un protocolo escrito y presupuesto documentado, ampliar evaluación de bordes y eficiencia, y diseñar comprobaciones de robustez que dispongan de metadatos suficientes. El listado operativo está en `PENDIENTES_ETAPA_2.md`.

Para el **trabajo de grado completo** quedan la colección manual por esquinas, adjudicación de máscaras, anotación de instancias, uso de cajas o modelos fundacionales para preetiquetar, datos sintéticos y aprendizaje activo (pp. 1–5). También quedan la rectificación, identificación de negativos difíciles e ilegibles, clasificación jerárquica, rechazo y evaluación de extremo a extremo (pp. 6–16). La combinación de fuentes requiere auditoría formal de compatibilidad, procedencia y licencia antes de cualquier mezcla.
