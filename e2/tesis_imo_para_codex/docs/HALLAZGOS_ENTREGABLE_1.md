# Registro de auditoría y cierre del Entregable 1

## Auditoría previa a las correcciones

Se revisaron la consigna, la propuesta del profesor, las fuentes de los cinco capítulos, el EDA inicial, los manifiestos de preparación, el cargador, las arquitecturas, el ciclo de entrenamiento, la evaluación cerrada, el análisis estadístico y el paquete cualitativo. El inventario histórico de propuestas de ajuste tiene fechas anteriores a la evaluación final.

| Comprobación de solo lectura | Evidencia y resultado |
|---|---|
| Procedencia experimental | ZIP final con SHA-256 `d17caeefccf7f15f4865db64cc8858a440dc9963cca6700031698838ae2b3954`. Coinciden el SHA-256 del TAR final y las huellas de los 15 checkpoints con `procedencia.json`. El origen institucional del corpus sigue siendo probable. |
| Integridad de las particiones | Coinciden las huellas de los manifiestos congelados. Se cotejaron los JPG, máscaras originales y máscaras binarias de los 1.451 pares elegibles con sus huellas. Cada fila coincide con el manifiesto maestro. No se repite ID ni grupo entre particiones. Train tiene 1.015 imágenes y 567 grupos; validación y test tienen cada uno 218 imágenes y 122 grupos. |
| Similitud | Se cotejaron los dos extremos de los 748 enlaces del CSV de candidatos con el manifiesto final. Cada par pertenece a un único grupo. El control cubre enlaces conocidos; no hay identificadores certificados de contenedor, puerto o sesión. |
| Selección | El máximo de cada uno de los 15 historiales concuerda con el Dice de validación registrado en el resumen de entrenamiento. El código guarda el primer máximo estricto. La tasa 0,001, paciencia 5, máximo 30 épocas, cinco semillas y umbral 0,5 preceden a la evaluación final. |
| Métricas | El producto cartesiano arquitectura × semilla × ID de test coincide exactamente con las 3.270 filas. Todas las métricas están entre cero y uno. En cada imagen se cumple Dice = 2 IoU/(1 + IoU) con tolerancia absoluta de 10⁻¹². La identidad no se aplica a las medias agregadas. |
| Inferencia | Se examinó el orden imagen → grupo → semilla, las mismas remuestras para los tres modelos, el IC percentil, las familias de Holm y la enumeración de Friedman. El IC describe diferencias medias entre grupos con semillas fijas. El contraste de rangos sobre cinco semillas carece de resolución para rechazar al 5 %. Un intervalo por contraste no da cobertura simultánea para los tres pares. |
| Presentación y visualización | El libro ofrece un procedimiento Colab pero no muestra las predicciones ni separa falsos positivos, falsos negativos y discrepancias de borde. Ya estaban fijados la semilla 11 y los tres identificadores, seleccionados según la distribución de FCN. |
| Fundamentos y alcance | La documentación describe bloques y pérdida, pero faltan una derivación del gradiente, la actualización de Adam y el vínculo entre IoU y Dice. Falta distinguir las recomendaciones del profesor para segmentación por instancia y clasificación de los resultados binarios actuales. |

## Cambios necesarios registrados antes de intervenir

1. Conservar los cinco capítulos y los artefactos experimentales. Incorporar al índice un resumen ejecutivo y enlazar alineación, hallazgos y pendientes.
2. Ampliar el capítulo de implementación con operaciones, dimensiones y ecuaciones del entrenamiento efectivamente utilizado.
3. Obtener con los tres checkpoints de semilla 11 únicamente las nueve máscaras cualitativas previstas. Cotejar con las métricas cerradas y documentar cualquier discrepancia CPU/T4. Representar contornos en coordenadas de evaluación y excluir el relleno.
4. Separar en la discusión resultados observados, interpretación bajo supuestos y propuestas futuras. Corregir el texto que aún presenta el panel como pendiente y verificar el HTML reconstruido.

## Resolución y controles

- **Integridad.** `scripts/auditar_cierre.py` reproduce la verificación de hashes y recalcula las componentes del grafo perceptual; la asignación coincide con el manifiesto en los 1.685 identificadores. Las seis pruebas de preparación pasaron, incluidas la geometría compartida, el manejo de máscaras vacías y la exclusión del relleno.
- **Estadística.** Se regeneraron las tablas desde los CSV y se cotejaron los resúmenes y los IC guardados. Una comprobación separada con SciPy reprodujo el estadístico de Friedman (1,6) y los p exactos de Wilcoxon. Para FCN–U-Net, FCN–SegNet y U-Net–SegNet, respectivamente: por semillas 0,4375, 0,8125 y 0,4375; por grupos 0,461631, 0,000621 y 0,0000635 antes de Holm. No hubo ceros ni empates en los rangos absolutos de los 122 grupos para estos pares.
- **Explicación matemática.** Se enlazaron las operaciones de las tres clases, la pérdida, el gradiente por logit y Adam con el código efectivo. Se contrastó la derivada escrita con autograd en logits artificiales, con relleno excluido y sin entrenar ninguna red; la discrepancia máxima fue inferior a 2 × 10⁻¹⁷.
- **Figuras.** Se usaron las tres fotos preseleccionadas y los pesos de semilla 11. La máxima discrepancia entre las cuatro métricas de los nueve paneles en CPU y sus registros en T4 fue cero. Solo se dispone de métricas guardadas en el ZIP, por lo que este cotejo no prueba igualdad de las máscaras píxel a píxel. Se incorporaron FP, FN y contornos, con leyendas y conteos.
- **Alcance documental.** La alineación identifica las secciones y páginas del PDF relevantes para cada tarea. El índice integra el resumen ejecutivo y enlaces a los documentos de cierre. La lista de pendientes distingue la siguiente comparación semántica de la eventual tesis de instancias y clasificación.

Los productos modificados son el libro, sus figuras y scripts de análisis/documentación. Ninguna corrección requirió alterar datos, checkpoints, particiones, hiperparámetros o registros originales. El archivo `libro.zip` de la versión anterior es histórico; la edición vigente se construye desde `libro/`.
