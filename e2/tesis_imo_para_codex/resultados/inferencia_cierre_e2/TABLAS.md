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

<!-- TEST CON GRUPOS: generado por scripts/inferencia_test_grupos_e2.py -->

# Test con los Dice por grupo

Generado por `scripts/inferencia_test_grupos_e2.py`. 122 grupos; reproduce `analisis.json` (discrepancia 0). Exploratorio, posterior al test.

## TOST en test, red − FCN

| Red | Diferencia | IC90 t | IC90 bootstrap | p TOST t Holm(9) | p TOST boot Holm(9) | Equivalente t / boot |
|---|---:|---:|---:|---:|---:|---|
| TransUNet | 0,0091 | [-0,0061; 0,0242] | [-0,0056; 0,0244] | 0,939 | 0,932 | No / No |
| U-Net compacta | -0,0072 | [-0,0148; 0,0005] | [-0,0148; 0,0002] | 0,0284 | 0,0396 | Sí / Sí |
| VM-UNet | -0,0145 | [-0,0275; -0,0015] | [-0,0277; -0,0017] | 1 | 1 | No / No |
| SegNet compacta | -0,0289 | [-0,0372; -0,0205] | [-0,0373; -0,0208] | 1 | 1 | No / No |
| DeepLabv3+ R18 | -0,0395 | [-0,0563; -0,0228] | [-0,0568; -0,0236] | 1 | 1 | No / No |
| U-Net++ R18 | -0,0459 | [-0,0676; -0,0243] | [-0,0678; -0,0250] | 1 | 1 | No / No |
| PIDNet-S | -0,0599 | [-0,0818; -0,0380] | [-0,0822; -0,0388] | 1 | 1 | No / No |
| SegNeXt-T | -0,0744 | [-0,0922; -0,0565] | [-0,0924; -0,0573] | 1 | 1 | No / No |
| SegFormer-B0 | -0,1483 | [-0,1751; -0,1214] | [-0,1754; -0,1218] | 1 | 1 | No / No |

## TOST en test, red − TransUNet

| Red | Diferencia | IC90 t | IC90 bootstrap | p TOST t Holm(9) | p TOST boot Holm(9) | Equivalente t / boot |
|---|---:|---:|---:|---:|---:|---|
| FCN compacta | -0,0091 | [-0,0242; 0,0061] | [-0,0244; 0,0056] | 1 | 1 | No / No |
| U-Net compacta | -0,0162 | [-0,0336; 0,0011] | [-0,0341; 0,0004] | 1 | 1 | No / No |
| VM-UNet | -0,0236 | [-0,0377; -0,0094] | [-0,0377; -0,0095] | 1 | 1 | No / No |
| SegNet compacta | -0,0379 | [-0,0545; -0,0213] | [-0,0546; -0,0219] | 1 | 1 | No / No |
| DeepLabv3+ R18 | -0,0486 | [-0,0662; -0,0310] | [-0,0667; -0,0320] | 1 | 1 | No / No |
| U-Net++ R18 | -0,0550 | [-0,0767; -0,0333] | [-0,0767; -0,0342] | 1 | 1 | No / No |
| PIDNet-S | -0,0690 | [-0,0907; -0,0472] | [-0,0912; -0,0483] | 1 | 1 | No / No |
| SegNeXt-T | -0,0834 | [-0,1024; -0,0645] | [-0,1026; -0,0653] | 1 | 1 | No / No |
| SegFormer-B0 | -0,1573 | [-0,1851; -0,1296] | [-0,1855; -0,1304] | 1 | 1 | No / No |

Diferencia máxima con el TOST desde resúmenes (p e IC90): 0,0e+00.

## Sensibilidad en test, red − FCN (p Holm(9) por procedimiento)

| Red | Media | Mediana | HL | t | Intercambio | Wilcoxon | Signo | Concuerdan | Simetría p | Asimetría |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| TransUNet | 0,0091 | 0,0042 | 0,0051 | 0,325 | 0,333 | 0,205 | 0,0567 | Sí | 0,561 | 1,22 |
| U-Net compacta | -0,0072 | -0,0001 | -0,0013 | 0,245 | 0,251 | 0,563 | 1 | Sí | 0,0682 | -0,79 |
| VM-UNet | -0,0145 | -0,0034 | -0,0057 | 0,201 | 0,197 | 0,0934 | 0,111 | Sí | 0,126 | -0,69 |
| SegNet compacta | -0,0289 | -0,0100 | -0,0168 | 5,8e-07 | 1,8e-04 | 7,3e-08 | 2,4e-06 | Sí | 2,0e-04 | -1,17 |
| DeepLabv3+ R18 | -0,0395 | -0,0100 | -0,0161 | 7,5e-04 | 3,0e-04 | 8,3e-06 | 1,0e-06 | Sí | 0,0014 | -1,80 |
| U-Net++ R18 | -0,0459 | -0,0009 | -0,0105 | 0,00243 | 0,00168 | 0,0934 | 1 | No | 2,0e-04 | -0,68 |
| PIDNet-S | -0,0599 | -0,0152 | -0,0265 | 8,3e-05 | 1,8e-04 | 7,9e-07 | 2,4e-06 | Sí | 6,0e-04 | -1,35 |
| SegNeXt-T | -0,0744 | -0,0308 | -0,0479 | 1,9e-09 | 1,8e-04 | 3,5e-12 | 2,7e-10 | Sí | 2,0e-04 | -1,16 |
| SegFormer-B0 | -0,1483 | -0,0915 | -0,1268 | 1,5e-14 | 1,8e-04 | 9,6e-16 | 1,0e-15 | Sí | 2,0e-04 | -0,96 |

## Sin un grupo o sin una semilla (test, red − FCN)

| Red | Media sin un grupo | p TOST sin un grupo | Cambios t / TOST sin un grupo | Media sin una semilla | Cambios t / TOST sin una semilla |
|---|---:|---:|---:|---:|---:|
| TransUNet | [0,0043; 0,0120] | [0,025; 0,182] | 0 / 1 | [0,0042; 0,0139] | 0 / 1 |
| U-Net compacta | [-0,0088; -0,0052] | [3,1e-04; 0,00566] | 1 / 0 | [-0,0117; -0,0027] | 1 / 1 |
| VM-UNet | [-0,0177; -0,0110] | [0,102; 0,375] | 6 / 0 | [-0,0270; -0,0089] | 1 / 0 |
| SegNet compacta | [-0,0301; -0,0275] | [0,935; 0,979] | 0 / 0 | [-0,0380; -0,0198] | 0 / 0 |
| DeepLabv3+ R18 | [-0,0427; -0,0358] | [0,951; 0,99] | 0 / 0 | [-0,0472; -0,0266] | 0 / 0 |
| U-Net++ R18 | [-0,0511; -0,0413] | [0,957; 0,994] | 0 / 0 | [-0,0516; -0,0407] | 0 / 0 |
| PIDNet-S | [-0,0635; -0,0556] | [0,997; 1] | 0 / 0 | [-0,0673; -0,0534] | 0 / 0 |
| SegNeXt-T | [-0,0772; -0,0710] | [1; 1] | 0 / 0 | [-0,0783; -0,0692] | 0 / 0 |
| SegFormer-B0 | [-0,1512; -0,1436] | [1; 1] | 0 / 0 | [-0,1543; -0,1415] | 0 / 0 |

## Multiplicidad simultánea en test (max-T por cambio de signo; IC95 simultáneos bootstrap)

Cuantil 95 % del máximo |t| bootstrap: 2,771.

| Red | t | p max-T | IC95 simultáneo |
|---|---:|---:|---:|
| TransUNet | 0,99 | 0,904 | [-0,0163; 0,0344] |
| U-Net compacta | -1,56 | 0,533 | [-0,0200; 0,0056] |
| VM-UNet | -1,85 | 0,333 | [-0,0362; 0,0072] |
| SegNet compacta | -5,71 | 2,0e-05 | [-0,0429; -0,0148] |
| DeepLabv3+ R18 | -3,91 | 6,8e-04 | [-0,0675; -0,0115] |
| U-Net++ R18 | -3,52 | 0,00322 | [-0,0821; -0,0098] |
| PIDNet-S | -4,53 | 6,0e-05 | [-0,0965; -0,0233] |
| SegNeXt-T | -6,91 | 2,0e-05 | [-0,1041; -0,0446] |
| SegFormer-B0 | -9,15 | 2,0e-05 | [-0,1932; -0,1034] |

## Sesgo del ganador en test

Ganador: TransUNet, Dice 0,8626; gana en el 83,3 % de los bootstrap; sesgo del nivel 0,0023, nivel corregido 0,8603; optimismo por mitades 0,0050.
