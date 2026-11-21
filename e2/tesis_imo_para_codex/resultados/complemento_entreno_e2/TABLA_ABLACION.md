# Ablación de TransUNet: Dice de validación por grupo

Generado por `scripts/tablas_complemento_entreno_e2.py` desde `RESUMEN.json`. Cinco semillas pareadas (11, 23, 37, 53, 71). Diferencia = variante − completa. Con 5 pares, el p mínimo del Wilcoxon bilateral es 0,0625. El bootstrap remuestrea los 122 grupos de validación.

| Variante | Parámetros | Dice val medio | Diferencia | IC95 t (semillas) | p t, Holm | p Wilcoxon | IC95 bootstrap (grupos) | Épocas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TransUNet completa (benchmark) | 105,3 M | 0,853 | — | — | — | — | — | 30 máx. |
| Sin Transformer | 20,3 M | 0,861 | 0,007 | [−0,001; 0,015] | 0,065 | 0,1250 | [0,001; 0,014] | 27–30 |
| Sin conexiones de salto | 103,8 M | 0,761 | −0,092 | [−0,105; −0,079] | < 0,001 | 0,0625 | [−0,109; −0,076] | 12–28 |
| BCE sola (sin Dice) | 105,3 M | 0,816 | −0,037 | [−0,043; −0,031] | < 0,001 | 0,0625 | [−0,049; −0,026] | 18–30 |
