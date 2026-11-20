| Red | Optimizador | Tasa | Weight decay |
|---|---|---:|---:|
| FCN compacta | Adam | 0.001 | 0.0 |
| SegNet compacta | Adam | 0.001 | 0.0 |
| U-Net compacta | Adam | 0.001 | 0.0 |
| DeepLabv3+ R18 | AdamW | 0.0005 | 0.01 |
| U-Net++ R18 | Adam | 0.0005 | 0.0 |
| PIDNet-S | AdamW | 0.001 | 0.01 |
| SegNeXt-T | AdamW | 0.0001 | 0.01 |
| SegFormer-B0 | AdamW | 6e-05 | 0.01 |
| TransUNet | SGD | 0.001 | 0.0001 |
| VM-UNet | AdamW | 0.001 | 0.01 |

SGD de TransUNet utiliza momentum 0,9. No hubo búsqueda posterior para seleccionar estas tasas.
