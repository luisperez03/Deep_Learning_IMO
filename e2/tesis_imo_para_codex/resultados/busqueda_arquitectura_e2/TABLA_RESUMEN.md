# Resumen por familia

| | FCN | U-Net | SegNet |
|---|---|---|---|
| Tasa del ajuste previo | 0,001 | 0,0003 | 0,0003 |
| Fase 1: media (DE entre arquitecturas) | 0,723 (0,016) | 0,729 (0,020) | 0,721 (0,026) |
| σ semilla / σ arquitectura (r) | 0,90 | 0,89 | 0,67 |
| Mayor media en fase 2 | `a0982` 0,860 | `a0593` 0,882 | `a0756` 0,873 |
| Ganadora por la regla (menos parámetros en el empate) | `a0498` 0,851 | `a0498` 0,873 | `a0498` 0,861 |
| Mayor media − ganadora | 0,010 | 0,009 | 0,013 |
| Empate técnico / equivalentes TOST | 11 / 4 | 11 / 7 | 11 / 5 |
| Ganadora − heredada (IC95 t) | 0,020 [0,011; 0,028] | 0,033 [0,026; 0,041] | 0,039 [0,020; 0,059] |
| Ganadora − heredada (IC95 bootstrap por grupos) | [0,010; 0,030] | [0,025; 0,041] | [0,028; 0,051] |
| Lectura fijada antes | el ganador supera a la heredada por menos de δ | el ganador supera a la heredada por δ o más | el ganador supera a la heredada por δ o más |
| Heredada en fase 2 / benchmark (val) | 0,831 / 0,831 | 0,840 / 0,832 | 0,821 / 0,806 |
| Spearman fase 1 − fase 2 (10 finalistas) | -0,07 (p = 0,855) | -0,96 (p = 0,000) | 0,39 (p = 0,260) |
| Rango de la referencia en fase 1 | 593 | 440 | 563 |
| DE arquitecturas / DE semilla | 1,36 | 2,67 | 2,37 |
| R² del modelo de anchos y dropout | 0,39 | 0,39 | 0,57 |
| Sin dropout − con dropout (Wilcoxon p) | 0,0022 (p = 0,015) | 0,0021 (p = 0,232) | 0,0045 (p = 0,000) |
