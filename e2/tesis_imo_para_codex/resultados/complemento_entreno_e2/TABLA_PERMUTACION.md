# Permutación de máscaras en train

Generado por `scripts/tablas_complemento_entreno_e2.py` desde `RESUMEN.json` y `corridas.csv`. p = (1 + #{nula ≥ real}) / (B + 1); con B = 200 el mínimo posible es 1/201 = 0,0050.

| Pipeline | B | Dice val real | Nula: mín. | media | mediana | máx. | p | Épocas por réplica |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FCN compacta | 200 | 0,823 | 0,045 | 0,465 | 0,490 | 0,592 | 0,0050 | 6–24 (media 11,8) |
