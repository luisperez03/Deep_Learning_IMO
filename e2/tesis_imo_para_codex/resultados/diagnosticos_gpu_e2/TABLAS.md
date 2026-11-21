# Complemento GPU: tablas de lectura

Generado por `scripts/leer_diagnosticos_gpu_e2.py` desde `RESUMEN_DIAGNOSTICOS.json` (status `complete`). Exploratorio, posterior a la exposición del test. Ningún umbral, modelo ni checkpoint se cambia con estos resultados. Media de 5 semillas; IC95 por bootstrap de grupos con semillas fijas.

## Reproducción del test cerrado y brecha train/validación (fila 57)

| Red | Dice test clean | Dice test cerrado | Dif. | Dice train | Dice val | Brecha train−val [IC95] |
|---|---:|---:|---:|---:|---:|---:|
| transunet_2d | 0,8626 | 0,8626 | -0,0000 | 0,941 | 0,853 | 0,087 [0,063; 0,115] |
| fcn | 0,8535 | 0,8535 | 0,0000 | 0,881 | 0,831 | 0,050 [0,020; 0,081] |
| unet | 0,8464 | 0,8464 | 0,0000 | 0,881 | 0,832 | 0,049 [0,020; 0,081] |
| vmunet | 0,8391 | 0,8391 | 0,0000 | 0,878 | 0,821 | 0,057 [0,026; 0,090] |
| segnet | 0,8247 | 0,8247 | -0,0000 | 0,832 | 0,806 | 0,026 [-0,007; 0,061] |
| deeplabv3plus | 0,8140 | 0,8140 | -0,0000 | 0,868 | 0,813 | 0,055 [0,024; 0,089] |
| unetplusplus | 0,8076 | 0,8076 | -0,0000 | 0,865 | 0,810 | 0,055 [0,021; 0,093] |
| pidnet_s | 0,7937 | 0,7936 | 0,0000 | 0,883 | 0,796 | 0,087 [0,053; 0,123] |
| segnext_t | 0,7792 | 0,7792 | -0,0000 | 0,906 | 0,759 | 0,147 [0,110; 0,186] |
| segformer_b0 | 0,7053 | 0,7053 | -0,0000 | 0,915 | 0,703 | 0,213 [0,166; 0,260] |

Diferencia máxima con el test cerrado: 0,0000.

## Calibración en test (fila 58)

| Red | ECE15 bruto | ECE15 con temperatura | Brier bruto | Brier con temperatura | Temperatura (semillas) |
|---|---:|---:|---:|---:|---|
| transunet_2d | 0,0079 | 0,0087 | 0,0081 | 0,0073 | 1,94, 1,90, 1,91, 1,78, 1,79 |
| fcn | 0,0082 | 0,0096 | 0,0083 | 0,0076 | 2,26, 1,80, 1,81, 1,86, 2,13 |
| unet | 0,0095 | 0,0107 | 0,0095 | 0,0085 | 2,01, 1,93, 1,75, 1,83, 2,02 |
| vmunet | 0,0091 | 0,0098 | 0,0094 | 0,0086 | 1,80, 1,68, 1,91, 1,79, 1,66 |
| segnet | 0,0112 | 0,0130 | 0,0113 | 0,0101 | 1,67, 1,59, 2,07, 1,87, 1,79 |
| deeplabv3plus | 0,0092 | 0,0098 | 0,0093 | 0,0085 | 1,91, 2,09, 2,17, 2,21, 2,14 |
| unetplusplus | 0,0114 | 0,0128 | 0,0113 | 0,0100 | 2,07, 2,01, 1,91, 2,03, 1,93 |
| pidnet_s | 0,0111 | 0,0116 | 0,0112 | 0,0100 | 2,24, 2,05, 2,47, 2,50, 2,39 |
| segnext_t | 0,0126 | 0,0136 | 0,0129 | 0,0113 | 2,47, 2,42, 2,41, 2,63, 2,54 |
| segformer_b0 | 0,0138 | 0,0144 | 0,0143 | 0,0131 | 1,98, 2,43, 1,97, 2,25, 2,04 |

## Bordes, precisión y recall en test (fila 64)

| Red | HD95 (px) | ASSD (px) | Precisión | Recall | Predicción vacía |
|---|---:|---:|---:|---:|---:|
| transunet_2d | 17,1 | 4,9 | 0,865 | 0,886 | 0,003 |
| fcn | 22,8 | 4,8 | 0,844 | 0,889 | 0,000 |
| unet | 24,6 | 6,2 | 0,832 | 0,891 | 0,003 |
| vmunet | 20,0 | 4,8 | 0,839 | 0,868 | 0,000 |
| segnet | 25,9 | 5,8 | 0,809 | 0,881 | 0,000 |
| deeplabv3plus | 23,7 | 9,2 | 0,831 | 0,829 | 0,011 |
| unetplusplus | 23,7 | 7,8 | 0,814 | 0,840 | 0,007 |
| pidnet_s | 21,3 | 7,7 | 0,817 | 0,800 | 0,007 |
| segnext_t | 25,2 | 6,5 | 0,776 | 0,816 | 0,002 |
| segformer_b0 | 34,3 | 8,6 | 0,788 | 0,688 | 0,002 |

## Costo de inferencia

| Red | Parámetros | GFLOPs | Latencia mediana (ms) |
|---|---:|---:|---:|
| transunet_2d | 105.322.001 | 76,8 | 39,0 |
| fcn | 72.145 | 1,3 | 2,1 |
| unet | 118.273 | 3,7 | 1,3 |
| vmunet | 44.273.833 | 19,5 | 44,2 |
| segnet | 85.761 | 2,1 | 1,5 |
| deeplabv3plus | 12.329.297 | 9,1 | 9,5 |
| unetplusplus | 15.970.449 | 31,9 | 9,0 |
| pidnet_s | 7.716.291 | 3,1 | 32,4 |
| segnext_t | 4.225.121 | 3,1 | 28,8 |
| segformer_b0 | 3.714.401 | 3,5 | 17,8 |

## Robustez en test (fila 59)

Área bajo la curva de Dice en severidad ordinal 0–3, normalizada por el Dice limpio (1 = no cae).

| Red | Ruido [IC95] | Desenfoque [IC95] | Brillo [IC95] |
|---|---:|---:|---:|
| transunet_2d | 0,921 [0,903; 0,937] | 0,917 [0,894; 0,938] | 0,980 [0,968; 0,990] |
| fcn | 0,684 [0,670; 0,699] | 0,729 [0,695; 0,763] | 0,749 [0,722; 0,776] |
| unet | 0,729 [0,714; 0,744] | 0,714 [0,681; 0,747] | 0,765 [0,738; 0,791] |
| vmunet | 0,936 [0,925; 0,945] | 0,954 [0,943; 0,965] | 0,882 [0,863; 0,900] |
| segnet | 0,742 [0,729; 0,756] | 0,730 [0,699; 0,760] | 0,730 [0,703; 0,757] |
| deeplabv3plus | 0,943 [0,929; 0,955] | 0,942 [0,924; 0,957] | 0,838 [0,813; 0,862] |
| unetplusplus | 0,964 [0,954; 0,973] | 0,953 [0,937; 0,967] | 0,858 [0,830; 0,883] |
| pidnet_s | 0,942 [0,931; 0,951] | 0,948 [0,930; 0,963] | 0,839 [0,816; 0,861] |
| segnext_t | 0,818 [0,797; 0,838] | 0,962 [0,946; 0,976] | 0,856 [0,834; 0,876] |
| segformer_b0 | 0,855 [0,833; 0,875] | 1,018 [1,006; 1,033] | 0,785 [0,751; 0,821] |

## Ensamble de semillas y predicción conforme (fila 62)

| Red | Dice ensamble [IC95] | Ganancia sobre la semilla media [IC95] | Cobertura conforme (nominal 0,90) | Grupos bajo el nominal | Tamaño medio del conjunto |
|---|---:|---:|---:|---:|---:|
| transunet_2d | 0,883 [0,855; 0,907] | 0,021 [0,012; 0,029] | 0,912 | 0,36 | 0,916 |
| unet | 0,874 [0,847; 0,899] | 0,028 [0,021; 0,035] | 0,903 | 0,37 | 0,908 |
| fcn | 0,878 [0,855; 0,898] | 0,024 [0,018; 0,031] | 0,904 | 0,34 | 0,909 |

## Explicabilidad: Integrated Gradients en validación (fila 60)

Ventaja de IG sobre un orden aleatorio al borrar (caída) o insertar (subida) el 10 % de píxeles más atribuidos, en unidades del logit medio de la región del rótulo: no es comparable entre redes, que tienen escalas de logit distintas. Estabilidad como Spearman medio entre pares de semillas.

| Red | Borrado: ventaja [IC95] | Casos donde IG gana | Inserción: ventaja [IC95] | Spearman entre semillas [IC95] |
|---|---:|---:|---:|---:|
| transunet_2d | 4,50 [3,50; 5,58] | 0,89 | 9,05 [8,24; 9,85] | 0,730 [0,712; 0,748] |
| unet | 18,14 [16,72; 19,53] | 1,00 | 7,08 [6,46; 7,68] | 0,917 [0,896; 0,935] |
| fcn | 18,31 [16,92; 19,62] | 1,00 | 6,77 [6,21; 7,35] | 0,915 [0,891; 0,934] |
