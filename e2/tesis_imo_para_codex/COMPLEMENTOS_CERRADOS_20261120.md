# Complementos locales terminados — 20/11/2026

Resultados exploratorios posteriores al test original. No alteran selección original ni protocolos cerrados.

| Análisis | Resultado | Evidencia |
|---|---|---|
| Clásico HistGradientBoosting | Dice 0,6446858; DE 5 semillas 0,0022226; IC95 grupos [0,6047324;0,6823762] | `resultados/baseline_clasico_e2/RESUMEN.json` |
| Selección clásico | 63 hojas; Dice val 0,6533073; candidatos 15/31/63, cinco semillas cada uno | `resultados/baseline_clasico_e2/SELECCION_ANTES_TEST.json` |
| Clasificación adversarial train/test | AUC crossfit 0,559719; IC95 condicional [0,506096;0,614331]; p permutación 0,089552 (200 permutaciones) | `resultados/complementos_locales_e2/validacion_adversarial.json` |
| Equivalencia TransUNet–FCN | TOST p=0,117387, margen ±0,02: no se establece equivalencia | `resultados/complementos_locales_e2/sensibilidad_estadistica.json` |
| Bordes TransUNet, seed11 | HD95 19,2505 px; ASSD 4,01491 px; precisión 0,859192; recall 0,903679 | `resultados/complementos_locales_e2/bordes_seed11_resumen.json` |

La falta de rechazo adversarial no demuestra ausencia de shift. Estas permutaciones prueban dominio, no etiquetas de segmentación. Las distancias de borde son a 256×256 con padding excluido y penalización diagonal para predicciones vacías; solo seed11. La descomposición de sumas de cuadrados es descriptiva, no estimación poblacional de efectos aleatorios. El clásico tiene tres candidatos de ajuste: no hubo igualdad retrospectiva de HPO con redes.

Este apartado conserva los resultados locales del 20/11. Los diagnósticos GPU y los experimentos complementarios se ejecutaron posteriormente y se presentan en sus capítulos; el avance de cierre recoge el estado al 26/11.
