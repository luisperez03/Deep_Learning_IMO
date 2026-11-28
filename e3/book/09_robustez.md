(sec-robustez)=
# Robustez, eficiencia y calidad (Cuadro 5)

Estudios declarados en `experiments/protocolo.md`, §10, y su estado.

```{list-table} Cuadro 5: estudios de robustez y su estado
:header-rows: 1
:name: tab-cuadro5

* - Estudio
  - Qué se hace
  - Estado
  - Fuente de los resultados
* - Estabilidad por semilla
  - Cajas, IC, coeficiente de variación y Kruskal-Wallis entre semillas
  - Parcial: DE entre semillas de los rivales en `data/tabla_maestra.csv`
  - Pendiente: script sobre el Dice por imagen de las 10 semillas
* - Perturbaciones
  - Ruido, desenfoque, brillo y oclusión en 3 niveles, por inferencia
  - Pendiente
  - Pendiente: requiere checkpoints
* - Eficiencia y Pareto
  - Parámetros, FLOPs, latencia en CPU y GPU, memoria pico, horas de entrenamiento
  - Rivales medidos en el E2 (tabla maestra); original pendiente
  - `data/tabla_maestra.csv`
* - Calibración
  - ECE, Brier, pendiente e intercepto, antes y después de temperature scaling
  - Pendiente en todas las filas (columna `calibracion_ece`)
  - Pendiente: no se midió en el E2
* - Sensibilidad a $g_0$
  - $g_0\in\{2, 8\}$ px, 3 semillas, en validación, exploratorio
  - Pendiente
  - Pendiente
* - Curva de aprendizaje
  - Original y su base, 25 % y 50 % de los grupos, 3 semillas
  - Pendiente
  - Pendiente
* - Cambio de distribución
  - Sin conjunto externo
  - No se hace (declarado)
  - `experiments/estandar.md`
* - Análisis de errores
  - Por área de la región y número de rótulos
  - Parcial: `data/dice_por_area.csv`, `data/limitacion_area.csv` (E2)
  - Pendiente para el original
* - Interpretabilidad
  - Grad-CAM y prueba de borrado, original frente al mejor rival
  - Pendiente
  - Pendiente
```

El original no entrega probabilidad por píxel de la región completa ($P$ es del núcleo): se reportará la calibración
de $P$ frente al núcleo y se declara la limitación. La energía solo se estima (horas por potencia máxima del
acelerador; {ref}`sec-reproducibilidad`).
