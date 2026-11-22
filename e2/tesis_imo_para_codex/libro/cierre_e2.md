# Discusión, limitaciones y cierre pendiente

## Qué permite concluir el experimento

TransUNet presenta el mayor Dice medio de test en esta comparación, seguido de FCN y U-Net compactas. Su mejora frente a FCN es menor que el margen práctico heredado de 0,02 y el IC pareado incluye cero. Esto no demuestra equivalencia ni descarta una mejora relevante: el extremo superior supera 0,02. FCN constituye una referencia de menor tamaño; no debe presentarse como ganadora en Dice ni como más rápida sin medir latencia comparable.

Los resultados no sostienen que una familia sea universalmente superior. Solo hay un representante Transformer, uno híbrido y uno SSM, con capacidad y optimizadores diferentes. En especial, la puntuación de SegFormer entrenado desde cero a 256 píxeles no es una réplica de su artículo ni demuestra que la autoatención sea inadecuada para HAZMAT.

## Respuesta a las preguntas

- **RQ1:** mayor estimación puntual para TransUNet, sin mejora clara frente a FCN con el IC cerrado. La ampliación estadística se presenta separada y como exploratoria.
- **RQ2:** el contraste global y los contrastes corregidos permiten describir heterogeneidad entre estas diez adaptaciones; ver los valores p en resultados. Una diferencia global no prueba todas las diferencias pareadas.
- **RQ3:** existen alternativas compactas competitivas; la equivalencia práctica con TransUNet no está demostrada.
- **RQ4:** los estratos de área y la correlación de errores permiten localizar dificultades; no prueban que el tamaño sea su causa ni que las anotaciones sean correctas.

## Amenazas a la validez

| Dimensión | Amenaza | Consecuencia y tratamiento |
|---|---|---|
| Interna | Recetas fijas distintas, auxiliares PIDNet y capacidades muy diferentes | No atribuir toda diferencia a arquitectura; la búsqueda equitativa (T = 4, exploratoria) muestra que SegFormer-B0 depende de la receta y que el resto se mueve poco |
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

## Plan de cierre para el domingo

1. **Informe y evidencia existente:** conservar esta integración, tablas, curvas, análisis estadístico, checklist y archivos reproducibles. Revisar juntos la redacción y la defensa técnica.
2. **Complemento GPU sobre pesos congelados:** medir Dice de train para diagnóstico, probabilidades y calibración, robustez con tres severidades, incertidumbre predictiva en los tres mejores por validación, explicabilidad con fidelidad/estabilidad, bordes y costo comparable. Ejecutado en una L4 sobre los 50 checkpoints, exploratorio; resultados en el capítulo de diagnósticos. Lo más relevante para la discusión: las redes compactas (FCN, U-Net, SegNet) conservan entre el 68 y el 77 % del Dice limpio a lo largo de las severidades de ruido, desenfoque y brillo (área normalizada de la curva), frente a cerca del 92 % de TransUNet con ruido y desenfoque, lo que matiza el argumento de parsimonia.
3. **Experimentos separados:** baseline clásico fuerte, tres ablaciones con cinco semillas, sanidad de etiquetas permutadas y réplica SegFormer en condiciones originales. No hacerlos pasar por parte del benchmark cerrado. Todos se ejecutaron: baseline clásico (Dice 0,645), réplica de evaluación compatible (37,43 % frente a 37,4 % mIoU) y, en {doc}`entrenamiento_complementario_e2`, la ablación de TransUNet (sin saltos −0,092, BCE sola −0,037, sin Transformer +0,007 con IC que incluye el cero), la permutación de máscaras con FCN (p = 0,005) y una búsqueda de hiperparámetros con el mismo presupuesto para las diez redes, exploratoria.
4. **Trabajo documental y humano:** adjudicar cribado/DOI, verificar bibliometría y cuartiles cuando correspondan, completar fichas de resultados de artículos, estimar acuerdo de anotación y preparar defensa. Q5/Q6 siguen omitidas por instrucción del usuario; no se reabren automáticamente.
5. **Control de entrega:** compilar localmente, revisar enlaces y cifras, ejecutar reproducción desde resultados y dejar registro de discrepancias. Solicitar autorización antes de publicar/subir a GitHub.

No se puede garantizar una calificación de 5 ni marcar como ejecutado lo pendiente. El checklist distingue requisitos cubiertos, parciales, pendientes, no aplicables y desviaciones históricas que no pueden corregirse retroactivamente.
