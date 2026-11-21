# Sobreajuste a un lote: Dice por imagen

Generado por `scripts/tabla_sobreajuste_e2.py` desde `resumen_sobreajuste.csv` y `areas_lote.csv`. Entre paréntesis, el área del rótulo como fracción de la imagen original. Umbral fijado antes: Dice medio del lote ≥ 0,90 tras 500 pasos. La pérdida es la de la receta de cada red (en PIDNet-S suma términos auxiliares), así que se lee dentro de cada fila y no entre redes.

| Red | image_0264 (3,0 %) | image_0796 (13,4 %) | image_1231 (1,1 %) | image_1250 (13,2 %) | Dice del lote | Pasa | Pérdida paso 1 → 200 → 500 |
|---|---:|---:|---:|---:|---:|:---:|---:|
| U-Net++ R18 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | sí | 0,894 → 0,015 → 0,002 |
| TransUNet | 0,999 | 1,000 | 1,000 | 0,999 | 1,000 | sí | 0,809 → 0,024 → 0,007 |
| DeepLabv3+ R18 | 0,994 | 1,000 | 0,993 | 1,000 | 0,996 | sí | 0,821 → 0,011 → 0,005 |
| SegNet compacta | 0,990 | 0,998 | 0,985 | 0,996 | 0,992 | sí | 0,787 → 0,027 → 0,006 |
| U-Net compacta | 0,981 | 0,993 | 0,981 | 0,995 | 0,988 | sí | 0,837 → 0,018 → 0,006 |
| PIDNet-S | 0,973 | 0,994 | 0,981 | 0,997 | 0,986 | sí | 5,405 → 0,035 → 0,031 |
| SegFormer-B0 | 0,969 | 0,996 | 0,964 | 0,996 | 0,981 | sí | 0,729 → 0,041 → 0,025 |
| FCN compacta | 0,969 | 0,996 | 0,955 | 0,994 | 0,978 | sí | 0,811 → 0,026 → 0,016 |
| SegNeXt-T | 0,956 | 0,993 | 0,924 | 0,993 | 0,967 | sí | 0,774 → 0,131 → 0,048 |
| VM-UNet | 1,000 | 1,000 | 0,000 | 1,000 | 0,750 | **no** | 0,768 → 0,141 → 0,137 |
