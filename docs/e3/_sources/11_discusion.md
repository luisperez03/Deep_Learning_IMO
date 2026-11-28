(sec-discusion)=
# Discusión

## Lectura del benchmark histórico

TransUNet presenta el mayor Dice medio de validación del E2 (0,8532) y un Dice de test de 0,8626. La FCN histórica alcanza 0,8535 en test con 72.145 parámetros, frente a 105.322.001 de TransUNet. La diferencia de medias es 0,0091. Esto ilustra el compromiso entre desempeño y tamaño, pero no demuestra equivalencia ni superioridad estadística. Los intervalos por grupos y la dispersión entre semillas se informan en {ref}`sec-resultados`.

La selección posterior de a0498 cambia la configuración de las compactas. La eficiencia y las métricas históricas no se trasladan automáticamente a esos artefactos. La media de una configuración, el resultado de una semilla representativa y el Dice de una imagen cumplen funciones distintas.

## Tamaño del objeto y motivación del mecanismo

En `data/limitacion_area.csv`, las diez redes tienen una diferencia puntual positiva entre el tercil grande y el pequeño en validación, de 0,0528 a 0,1198 de Dice. Sin embargo, el intervalo del 95 % de esa diferencia excluye cero solo en cinco redes. El patrón motiva examinar el tamaño del objeto, sin convertirlo en una pérdida estadísticamente establecida para todas las arquitecturas ni atribuirlo exclusivamente al borde.

UNet-AT propone conservar núcleos de objetos delgados y reconstruir las regiones mediante expansión consistente. Las pruebas geométricas verifican casos sintéticos y la implementación bajo una versión fijada de OpenCV. No permiten concluir que la red entrenada mejore en fotografías ni que supere a los rivales.

## Lecciones de integración

La discrepancia entre OpenCV 4.10 y 4.11 mostró que la rasterización afecta la comparación aun sin cambiar pesos ni geometría. Registrar la versión en la identidad del entrenamiento evita mezclar entornos al reanudar. SegNet muestra además que coincidencia de máscaras y paridad de probabilidades no son equivalentes: la exportación ONNX falló el segundo criterio.

El tablero permite inspeccionar fallos sobre una misma fotografía para todas las redes disponibles. Su selección aleatoria no elimina la exposición al test ni convierte ejemplos en evidencia de superioridad.

## Discusión pendiente del original

La hipótesis requiere contrastar `dbnet_at` con `dbnet` en el tercil pequeño y en el conjunto completo, con diferencias e intervalos por la unidad de análisis especificada. Las ablaciones deben separar C1 y C2. Deben revisarse potencia, familia de comparaciones y multiplicidad antes de concluir. Un resultado no significativo no probaría equivalencia; una media mayor tampoco bastaría para acreditar R3. La evaluación adicional y la revisión de novedad siguen abiertas.
