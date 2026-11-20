### Cómo contar capas y neuronas

Contamos como capa entrenable cada convolución, incluida la cabeza final de 1 × 1. Un filtro produce un canal y reutiliza los mismos pesos en muchas posiciones. Por eso conviene distinguir filtros, activaciones y parámetros. Por ejemplo, la primera convolución de las tres redes tiene 16 filtros, produce 16 × 256 × 256 = 1.048.576 activaciones por imagen y aprende (3 × 3 × 3 + 1) × 16 = 448 parámetros.

Si por «neuronas» se entienden las **unidades de salida de las convoluciones**, la columna de activaciones suma esas unidades en todas las capas por imagen. Depende de la resolución. Los parámetros son los pesos y sesgos aprendidos, compartidos entre posiciones. Pooling, ReLU, ampliaciones y concatenaciones intervienen en el cálculo y no añaden parámetros.

| Red | Capas entrenables | Filtros sumados | Activaciones convolucionales por imagen | Parámetros |
|---|---:|---:|---:|---:|
| FCN | 7 | 225 | 3.674.112 | 72.145 |
| SegNet | 12 | 321 | 6.488.064 | 85.761 |
| U-Net | 11 | 321 | 6.881.280 | 118.273 |

### FCN

![Camino del codificador y reconstrucción espacial de FCN](figuras/mapa_fcn.png)

La cabeza calcula un solo logit por posición en el mapa de 64 × 64. La ampliación bilineal recupera el tamaño de la máscara sin añadir pesos.

```{dropdown} Ver las 7 capas convolucionales de FCN

| Capa en PyTorch | Núcleo | Canales entrada → salida | Salida (C × H × W) | Parámetros |
|---|---|---:|---|---:|
| `features.0.0` | 3 × 3 | 3 → 16 | 16 × 256 × 256 | 448 |
| `features.0.2` | 3 × 3 | 16 → 16 | 16 × 256 × 256 | 2.320 |
| `features.2.0` | 3 × 3 | 16 → 32 | 32 × 128 × 128 | 4.640 |
| `features.2.2` | 3 × 3 | 32 → 32 | 32 × 128 × 128 | 9.248 |
| `features.4.0` | 3 × 3 | 32 → 64 | 64 × 64 × 64 | 18.496 |
| `features.4.2` | 3 × 3 | 64 → 64 | 64 × 64 × 64 | 36.928 |
| `head` | 1 × 1 | 64 → 1 | 1 × 64 × 64 | 65 |
```

### SegNet

![Camino del codificador y reconstrucción espacial de SegNet](figuras/mapa_segnet.png)

Las ramas laterales transportan los índices del máximo de cada ventana. El decodificador usa estos índices para recolocar los valores antes de aplicar sus convoluciones.

```{dropdown} Ver las 12 capas convolucionales de SegNet

| Capa en PyTorch | Núcleo | Canales entrada → salida | Salida (C × H × W) | Parámetros |
|---|---|---:|---|---:|
| `e1.0` | 3 × 3 | 3 → 16 | 16 × 256 × 256 | 448 |
| `e1.2` | 3 × 3 | 16 → 16 | 16 × 256 × 256 | 2.320 |
| `e2.0` | 3 × 3 | 16 → 32 | 32 × 128 × 128 | 4.640 |
| `e2.2` | 3 × 3 | 32 → 32 | 32 × 128 × 128 | 9.248 |
| `e3.0` | 3 × 3 | 32 → 64 | 64 × 64 × 64 | 18.496 |
| `e3.2` | 3 × 3 | 64 → 64 | 64 × 64 × 64 | 36.928 |
| `reduce` | 1 × 1 | 64 → 32 | 32 × 64 × 64 | 2.080 |
| `d2.0` | 3 × 3 | 32 → 16 | 16 × 128 × 128 | 4.624 |
| `d2.2` | 3 × 3 | 16 → 16 | 16 × 128 × 128 | 2.320 |
| `d1.0` | 3 × 3 | 16 → 16 | 16 × 256 × 256 | 2.320 |
| `d1.2` | 3 × 3 | 16 → 16 | 16 × 256 × 256 | 2.320 |
| `head` | 1 × 1 | 16 → 1 | 1 × 256 × 256 | 17 |
```

### U-Net

![Camino del codificador y reconstrucción espacial de U-Net](figuras/mapa_unet.png)

Las ramas laterales copian activaciones completas. Tras ampliar por vecino más cercano, se concatenan 64 + 32 canales a 128 × 128 y 32 + 16 canales a 256 × 256.

```{dropdown} Ver las 11 capas convolucionales de U-Net

| Capa en PyTorch | Núcleo | Canales entrada → salida | Salida (C × H × W) | Parámetros |
|---|---|---:|---|---:|
| `e1.0` | 3 × 3 | 3 → 16 | 16 × 256 × 256 | 448 |
| `e1.2` | 3 × 3 | 16 → 16 | 16 × 256 × 256 | 2.320 |
| `e2.0` | 3 × 3 | 16 → 32 | 32 × 128 × 128 | 4.640 |
| `e2.2` | 3 × 3 | 32 → 32 | 32 × 128 × 128 | 9.248 |
| `e3.0` | 3 × 3 | 32 → 64 | 64 × 64 × 64 | 18.496 |
| `e3.2` | 3 × 3 | 64 → 64 | 64 × 64 × 64 | 36.928 |
| `d2.0` | 3 × 3 | 96 → 32 | 32 × 128 × 128 | 27.680 |
| `d2.2` | 3 × 3 | 32 → 32 | 32 × 128 × 128 | 9.248 |
| `d1.0` | 3 × 3 | 48 → 16 | 16 × 256 × 256 | 6.928 |
| `d1.2` | 3 × 3 | 16 → 16 | 16 × 256 × 256 | 2.320 |
| `head` | 1 × 1 | 16 → 1 | 1 × 256 × 256 | 17 |
```
