# Complemento de entrenamiento E2 — resumen

Generado por `scripts/resumir_complemento_entreno_e2.py`. Exploratorio: el test del benchmark ya fue observado; aquí todo se elige y se mide en validación.

## Sobreajuste a un lote de 4 imágenes

| red | Dice lote | pérdida inicial | pérdida final | pasa |
|---|---|---|---|---|
| deeplabv3plus | 0.9964 | 0.8211 | 0.0046 | sí |
| fcn | 0.9782 | 0.8111 | 0.0161 | sí |
| pidnet_s | 0.9863 | 5.4047 | 0.0313 | sí |
| segformer_b0 | 0.9811 | 0.7293 | 0.0246 | sí |
| segnet | 0.9923 | 0.7872 | 0.0061 | sí |
| segnext_t | 0.9665 | 0.7744 | 0.0484 | sí |
| transunet_2d | 0.9995 | 0.8088 | 0.0075 | sí |
| unet | 0.9875 | 0.8373 | 0.0058 | sí |
| unetplusplus | 1.0000 | 0.8939 | 0.0016 | sí |
| vmunet | 0.7499 | 0.7684 | 0.1371 | no |

## Ablación de TransUNet (referencia: benchmark original)

| variante | corridas | Dice medio | diferencia | IC95 t | p t | p Holm | p Wilcoxon | IC95 grupos |
|---|---|---|---|---|---|---|---|---|
| sin_transformer | 5 | 0.8606 | 0.0074 | [-0.0007; 0.0155] | 0.0651 | 0.0651 | 0.1250 | [0.0009; 0.0140] |
| sin_saltos | 5 | 0.7613 | -0.0919 | [-0.1053; -0.0786] | 0.0000 | 0.0001 | 0.0625 | [-0.1085; -0.0764] |
| sin_dice | 5 | 0.8163 | -0.0369 | [-0.0431; -0.0308] | 0.0001 | 0.0002 | 0.0625 | [-0.0492; -0.0257] |

## Permutación de máscaras en train

- pipeline_fcn (fcn): B = 200 de 200; s_obs = 0.8230; s_b media 0.4651 (máx. 0.5915); p = 0.0050 (mínimo posible 0.0050).
- transunet: sin réplicas completas.

## Búsqueda de hiperparámetros (T completo para las 10: 4)

| red | t ganador | lr | weight decay | Dice val | receta mismo presupuesto | rango |
|---|---|---|---|---|---|---|
| fcn | 3 | 1.19e-03 | 0.0 | 0.7714 | 0.7528 | 0.1041 |
| segnet | 3 | 1.19e-03 | 0.0 | 0.7813 | 0.7531 | 0.7813 |
| unet | 3 | 1.19e-03 | 0.0 | 0.7698 | 0.7667 | 0.1419 |
| deeplabv3plus | 3 | 5.95e-04 | 0.0 | 0.7276 | 0.7148 | 0.0397 |
| unetplusplus | 1 | 5.10e-05 | 0.0001 | 0.7845 | 0.7738 | 0.1254 |
| pidnet_s | 3 | 1.19e-03 | 0.0 | 0.7783 | 0.7564 | 0.1222 |
| segnext_t | 2 | 9.29e-04 | 0.0001 | 0.7794 | 0.7348 | 0.1256 |
| segformer_b0 | 2 | 5.57e-04 | 0.0001 | 0.7820 | 0.6603 | 0.2419 |
| transunet_2d | 2 | 9.29e-03 | 0.0001 | 0.8140 | 0.8126 | 0.0717 |
| vmunet | 0 | 1.00e-03 | 0.01 | 0.8013 | 0.8013 | 0.7918 |

Una semilla y 10 épocas por intento: el ganador lleva ruido de semilla. Exploratorio.
