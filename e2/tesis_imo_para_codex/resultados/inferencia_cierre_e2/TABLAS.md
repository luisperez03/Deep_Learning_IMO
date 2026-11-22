# Tablas de inferencia complementaria

Generado por `scripts/inferencia_cierre_e2.py`. Exploratorio, posterior al test. δ = 0,02 Dice; TOST con α = 0,05 por lado (IC90 dentro de ±δ).

## TOST en test, red − FCN (t pareado desde media, d_z y n = 122)

| Red | Diferencia | DE | IC90 t | p TOST | p TOST Holm(9) | Equivalente (Holm) |
|---|---:|---:|---:|---:|---:|---|
| TransUNet | 0,0091 | 0,1012 | [-0,0061; 0,0242] | 0,117 | 0,939 | No |
| U-Net compacta | -0,0072 | 0,0510 | [-0,0148; 0,0005] | 0,00315 | 0,0284 | Sí |
| VM-UNet | -0,0145 | 0,0866 | [-0,0275; -0,0015] | 0,242 | 1 | No |
| SegNet compacta | -0,0289 | 0,0558 | [-0,0372; -0,0205] | 0,959 | 1 | No |
| DeepLabv3+ R18 | -0,0395 | 0,1116 | [-0,0563; -0,0228] | 0,972 | 1 | No |
| U-Net++ R18 | -0,0459 | 0,1441 | [-0,0676; -0,0243] | 0,975 | 1 | No |
| PIDNet-S | -0,0599 | 0,1460 | [-0,0818; -0,0380] | 0,998 | 1 | No |
| SegNeXt-T | -0,0744 | 0,1188 | [-0,0922; -0,0565] | 1 | 1 | No |
| SegFormer-B0 | -0,1483 | 0,1790 | [-0,1751; -0,1214] | 1 | 1 | No |

## TOST en validación, red − FCN

| Red | Diferencia | IC90 t | IC90 bootstrap | p TOST t Holm(9) | p TOST boot Holm(9) | Equivalente t / boot |
|---|---:|---:|---:|---:|---:|---|
| TransUNet | 0,0223 | [0,0094; 0,0352] | [0,0097; 0,0355] | 1 | 1 | No / No |
| U-Net compacta | 0,0011 | [-0,0079; 0,0100] | [-0,0075; 0,0100] | 0,00285 | 0,0045 | Sí / Sí |
| VM-UNet | -0,0102 | [-0,0266; 0,0062] | [-0,0267; 0,0060] | 1 | 1 | No / No |
| DeepLabv3+ R18 | -0,0176 | [-0,0353; 0,0001] | [-0,0352; -0,0005] | 1 | 1 | No / No |
| U-Net++ R18 | -0,0210 | [-0,0424; 0,0003] | [-0,0426; -0,0004] | 1 | 1 | No / No |
| SegNet compacta | -0,0247 | [-0,0358; -0,0135] | [-0,0357; -0,0137] | 1 | 1 | No / No |
| PIDNet-S | -0,0346 | [-0,0538; -0,0153] | [-0,0539; -0,0156] | 1 | 1 | No / No |
| SegNeXt-T | -0,0715 | [-0,0916; -0,0514] | [-0,0917; -0,0521] | 1 | 1 | No / No |
| SegFormer-B0 | -0,1284 | [-0,1595; -0,0973] | [-0,1596; -0,0980] | 1 | 1 | No / No |

## TOST en validación, red − TransUNet

| Red | Diferencia | IC90 t | IC90 bootstrap | p TOST t Holm(9) | p TOST boot Holm(9) | Equivalente t / boot |
|---|---:|---:|---:|---:|---:|---|
| U-Net compacta | -0,0212 | [-0,0325; -0,0100] | [-0,0326; -0,0105] | 1 | 1 | No / No |
| FCN compacta | -0,0223 | [-0,0352; -0,0094] | [-0,0355; -0,0097] | 1 | 1 | No / No |
| VM-UNet | -0,0325 | [-0,0435; -0,0215] | [-0,0437; -0,0218] | 1 | 1 | No / No |
| DeepLabv3+ R18 | -0,0399 | [-0,0523; -0,0275] | [-0,0525; -0,0280] | 1 | 1 | No / No |
| U-Net++ R18 | -0,0434 | [-0,0585; -0,0282] | [-0,0588; -0,0292] | 1 | 1 | No / No |
| SegNet compacta | -0,0470 | [-0,0591; -0,0349] | [-0,0593; -0,0351] | 1 | 1 | No / No |
| PIDNet-S | -0,0569 | [-0,0722; -0,0416] | [-0,0722; -0,0421] | 1 | 1 | No / No |
| SegNeXt-T | -0,0938 | [-0,1135; -0,0742] | [-0,1138; -0,0749] | 1 | 1 | No / No |
| SegFormer-B0 | -0,1507 | [-0,1798; -0,1216] | [-0,1802; -0,1225] | 1 | 1 | No / No |

## Sensibilidad en validación, red − FCN (p Holm(9) por procedimiento)

| Red | Media | Mediana | HL | t | Intercambio | Wilcoxon | Signo | Concuerdan | Simetría p | Asimetría |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| TransUNet | 0,0223 | 0,0042 | 0,0077 | 0,0248 | 0,0205 | 0,0664 | 0,255 | No | 0,0082 | 1,34 |
| U-Net compacta | 0,0011 | 0,0003 | -0,0016 | 0,842 | 0,846 | 0,536 | 0,928 | Sí | 0,876 | 1,70 |
| VM-UNet | -0,0102 | -0,0049 | -0,0083 | 0,609 | 0,611 | 0,0827 | 0,0106 | No | 0,543 | -0,33 |
| DeepLabv3+ R18 | -0,0176 | -0,0098 | -0,0129 | 0,406 | 0,408 | 0,0295 | 0,00186 | No | 0,402 | -0,52 |
| U-Net++ R18 | -0,0210 | -0,0054 | -0,0139 | 0,406 | 0,408 | 0,0664 | 0,255 | Sí | 0,157 | -0,72 |
| SegNet compacta | -0,0247 | -0,0089 | -0,0149 | 0,00247 | 0,0021 | 1,5e-05 | 3,5e-07 | Sí | 0,0052 | -0,39 |
| PIDNet-S | -0,0346 | -0,0186 | -0,0261 | 0,021 | 0,0205 | 2,7e-04 | 1,7e-05 | Sí | 0,0976 | -0,50 |
| SegNeXt-T | -0,0715 | -0,0376 | -0,0524 | 2,8e-07 | 1,8e-04 | 1,0e-08 | 3,7e-09 | Sí | 4,0e-04 | -1,78 |
| SegFormer-B0 | -0,1284 | -0,0594 | -0,0977 | 3,2e-09 | 1,8e-04 | 4,9e-10 | 7,5e-11 | Sí | 2,0e-04 | -1,02 |

## Multiplicidad simultánea en validación (max-T por cambio de signo; IC95 simultáneos bootstrap)

Cuantil 95 % del máximo |t| bootstrap: 2,752.

| Red | t | p max-T | IC95 simultáneo |
|---|---:|---:|---:|
| TransUNet | 2,86 | 0,0269 | [0,0009; 0,0438] |
| U-Net compacta | 0,20 | 1 | [-0,0138; 0,0159] |
| VM-UNet | -1,03 | 0,851 | [-0,0374; 0,0170] |
| DeepLabv3+ R18 | -1,65 | 0,42 | [-0,0469; 0,0118] |
| U-Net++ R18 | -1,64 | 0,43 | [-0,0565; 0,0144] |
| SegNet compacta | -3,68 | 0,00158 | [-0,0431; -0,0062] |
| PIDNet-S | -2,98 | 0,0188 | [-0,0665; -0,0026] |
| SegNeXt-T | -5,89 | 2,0e-05 | [-0,1049; -0,0381] |
| SegFormer-B0 | -6,83 | 2,0e-05 | [-0,1801; -0,0767] |

## Sesgo del ganador en validación

Ganador: TransUNet, Dice 0,8532. Bootstrap: sesgo del nivel 0,0001, nivel corregido 0,8532; frecuencia como ganador 0,9987.
Mitades (elegir en A, medir en B, R = 2000): optimismo del nivel -0,0004; de la ventaja sobre FCN 0,0003.
Checkpoint por validación frente a test: test − val medio 0,0101 (2 de 10 redes con test menor). TransUNet − FCN: 0,0223 en validación y 0,0091 en test.
