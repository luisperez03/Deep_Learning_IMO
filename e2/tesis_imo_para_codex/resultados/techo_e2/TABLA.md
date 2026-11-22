# Techo humano y γ

Generado por `scripts/tabla_techo_e2.py` desde `techo_reanotacion.json` y `gamma.json`. Techo: Dice entre una segunda anotación independiente y la máscara del conjunto en 50 imágenes de validación (una por grupo), con la geometría y la métrica de los modelos. γ = (techo − s_red)/(techo − s_clásico): fracción del margen sobre el clásico que la red aún no cubre.

**Techo:** Dice 0,941 (IC95 bootstrap por grupos 0,928–0,953); mediana 0,953; IoU 0,892; por tercil de área (pequeño, medio, grande): 0,933, 0,941, 0,950. Clásico: 0,653 en validación y 0,645 en test.

| Red | Dice val (122 grupos) | Dice en las 50 del techo | Distancia al techo (mismas 50) | γ val | γ test |
|---|---:|---:|---:|---:|---:|
| TransUNet | 0,853 | 0,866 | 0,075 | 0,306 | 0,265 |
| U-Net compacta | 0,832 | 0,864 | 0,077 | 0,379 | 0,320 |
| FCN compacta | 0,831 | 0,851 | 0,091 | 0,383 | 0,296 |
| VM-UNet | 0,821 | 0,837 | 0,105 | 0,419 | 0,345 |
| DeepLabv3+ R18 | 0,813 | 0,820 | 0,121 | 0,444 | 0,429 |
| U-Net++ R18 | 0,810 | 0,804 | 0,137 | 0,456 | 0,451 |
| SegNet compacta | 0,806 | 0,828 | 0,114 | 0,469 | 0,393 |
| PIDNet-S | 0,796 | 0,797 | 0,144 | 0,503 | 0,498 |
| SegNeXt-T | 0,759 | 0,765 | 0,177 | 0,632 | 0,546 |
| SegFormer-B0 | 0,703 | 0,717 | 0,225 | 0,829 | 0,796 |
