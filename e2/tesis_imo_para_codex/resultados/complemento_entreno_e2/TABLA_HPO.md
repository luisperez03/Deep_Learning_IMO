# Búsqueda de hiperparámetros: Dice de validación por intento

Generado por `scripts/tablas_complemento_entreno_e2.py` desde `RESUMEN.json`. Una semilla (11), máximo 10 épocas por intento, mismo espacio relativo para las 10 redes: cada intento multiplica la tasa de aprendizaje de la receta y fija el weight decay (1e-4 en t1 y t2, 0 en t3). En negrita, el mejor intento de cada red. Ordenado por el mejor Dice.

| Red | Receta (t0) | t1: lr × 0,1 | t2: lr × 9,3 | t3: lr × 1,19, sin wd | Mejor − receta |
|---|---:|---:|---:|---:|---:|
| TransUNet | 0,813 | 0,742 | **0,814** | 0,805 | 0,001 |
| VM-UNet | **0,801** | 0,750 | 0,009 | 0,758 | 0,000 |
| U-Net++ R18 | 0,774 | **0,785** | 0,659 | 0,764 | 0,011 |
| SegFormer-B0 | 0,660 | 0,540 | **0,782** | 0,668 | 0,122 |
| SegNet compacta | 0,753 | 0,736 | 0,000 | **0,781** | 0,028 |
| SegNeXt-T | 0,735 | 0,654 | **0,779** | 0,746 | 0,045 |
| PIDNet-S | 0,756 | 0,656 | 0,737 | **0,778** | 0,022 |
| FCN compacta | 0,753 | 0,710 | 0,667 | **0,771** | 0,019 |
| U-Net compacta | 0,767 | 0,734 | 0,628 | **0,770** | 0,003 |
| DeepLabv3+ R18 | 0,715 | 0,722 | 0,688 | **0,728** | 0,013 |
