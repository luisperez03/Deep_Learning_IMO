# Discusión, limitaciones y próximos pasos

## Qué permite concluir el experimento

TransUNet presenta el mayor Dice medio de test en esta comparación, seguido de FCN y U-Net compactas. Su mejora frente a FCN es menor que el margen práctico heredado de 0,02 y el IC pareado incluye cero. Esto no demuestra equivalencia ni descarta una mejora relevante: el extremo superior supera 0,02, y el TOST con δ = 0,02 queda indeterminado (IC90 [−0,0061; 0,0242], p = 0,117). FCN constituye una referencia de menor tamaño. La comparación de velocidad se basa en las mediciones homogéneas de latencia, no en el número de parámetros.

Los resultados no sostienen que una familia sea universalmente superior. Solo hay un representante Transformer, uno híbrido y uno SSM, con capacidad y optimizadores diferentes. En especial, la puntuación de SegFormer entrenado desde cero a 256 píxeles no es una réplica de su artículo ni demuestra que la autoatención sea inadecuada para HAZMAT.

## Respuesta a las preguntas

- **RQ1:** mayor estimación puntual para TransUNet, sin mejora clara frente a FCN con el IC cerrado. La ampliación estadística se presenta separada y como exploratoria.
- **RQ2:** el contraste global y los contrastes corregidos permiten describir heterogeneidad entre estas diez adaptaciones; ver los valores p en resultados. Una diferencia global no prueba todas las diferencias pareadas.
- **RQ3:** existen alternativas compactas competitivas, pero no se demostró equivalencia con TransUNet mediante TOST con δ = 0,02 (la más cercana, FCN, p = 0,117; luego U-Net, p = 0,36). La evidencia es insuficiente para afirmar equivalencia; tampoco demuestra ausencia de equivalencia. La inferencia es condicional a las cinco semillas.
- **RQ4:** los estratos de área y la correlación de errores permiten localizar dificultades; no prueban que el tamaño sea su causa ni que las anotaciones sean correctas.

## Amenazas a la validez

| Dimensión | Amenaza | Consecuencia y tratamiento |
|---|---|---|
| Interna | Recetas fijas distintas, auxiliares PIDNet y capacidades muy diferentes | No atribuir toda diferencia a arquitectura; la búsqueda equitativa (T = 4, exploratoria) muestra que SegFormer-B0 depende de la receta y que el resto se mueve poco |
| Interna | Compactas heredadas del E1, sin búsqueda de arquitectura en el benchmark | La búsqueda en dos fases ({doc}`busqueda_arquitectura_e2`, solo validación) muestra que no eran óptimas: con la arquitectura elegida por la regla y la tasa del ajuste previo ganan 0,020 (FCN), 0,041 (U-Net) y 0,054 (SegNet) de Dice de validación. El benchmark subestimó U-Net y SegNet; el test no se reevalúa |
| Selección | Test ya observado en E1; cierre E2 después de validación | Declarar la cronología; no afinar usando los resultados actuales |
| Estadística | Cinco semillas, una partición, grupos de similitud aproximados | IC condicionales; no usar píxeles como réplicas ni afirmar reproducibilidad entre datasets |
| Constructo | Máscaras con omisiones y objetivo binario | Dice evalúa coincidencia con anotación disponible, no cumplimiento normativo IMO |
| Externa | Sin imágenes completamente negativas ni conjunto portuario independiente | No estimar tasa operacional de falsas alarmas ni seguridad industrial |
| Medición de costo | L4 en entrenamiento, T4 en test; tiempos con E/S | No interpretar segundos registrados como latencia pura; no inferir energía sin medición |
| Literatura | Cribado y bibliometría incompletos | No llamar a la lista Top 10 exhaustivo ni afirmar un gap universal |

## Techo y margen restante

Una segunda anotación independiente de 50 imágenes de validación da un techo de Dice 0,941 (IC95 0,928–0,953), no 1 ({doc}`complementos_datos_e2`). Sobre esas mismas imágenes TransUNet llega a 0,866: le falta cerca de 0,075 y cubre el 70 % del margen sobre el clásico (γ = 0,31 en validación, 0,27 en test). El margen restante es mayor que las diferencias entre las mejores redes y se concentra en los rótulos pequeños. Es una sola segunda persona y una muestra de 50; el techo mide acuerdo con las etiquetas del conjunto, que omiten rótulos cortados por el borde en un 10 % de las imágenes (IC95 5,5–17,4 %).

## Gap e hipótesis para una siguiente fase

La evidencia local motiva investigar si mecanismos que preservan bordes de rótulos pequeños mejoran la relación desempeño/tamaño bajo una comparación controlada. Es una **hipótesis candidata**, no una ausencia demostrada de estudios en toda la literatura. El siguiente experimento debe separar capacidad, preentrenamiento y pérdidas; fijar α, δ, presupuesto y evaluación independiente antes de ejecutarse. No se seleccionará una arquitectura E3 mediante optimización sobre este test.

## Estado del cierre y trabajo pendiente

El cierre del 26 de noviembre incluye los diagnósticos sobre los 50 checkpoints, el clásico no profundo, las ablaciones, la prueba de máscaras permutadas, la réplica de evaluación de SegFormer y las búsquedas de hiperparámetros y arquitectura. Q5 y Q6 se ejecutaron el 22 de noviembre. Los complementos posteriores al test se mantienen como análisis exploratorios, separados del benchmark principal.

Estos resultados matizan la comparación inicial. Las compactas conservan entre el 68 y el 77 % del Dice limpio en las curvas de degradación, frente a cerca del 92 % de TransUNet con ruido y desenfoque. La búsqueda de arquitectura mejora las compactas en validación, pero no se evaluaron esas variantes sobre un test independiente. La ablación sin Transformer presenta una diferencia de +0,007, con un intervalo que incluye cero: no demuestra una mejora ni una pérdida de desempeño.

Quedan pendientes el ensayo de sustentación y el registro individual de la revisión humana de errores. La revisión disponible es una confirmación global de 88 paneles; no permite estimar un acuerdo independiente caso por caso. El checklist conserva las limitaciones históricas de preregistro, potencia, exposición previa al test y licencia de las imágenes.

Para el entregable 3, la evidencia orienta una comparación controlada de mecanismos de borde y resolución en redes compactas. Se requiere concretar la propuesta, fijar presupuesto y criterios de éxito, y disponer de una evaluación independiente del test ya observado.
