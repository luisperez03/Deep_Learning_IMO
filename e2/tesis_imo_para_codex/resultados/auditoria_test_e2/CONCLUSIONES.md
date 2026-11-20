# Resultados finales E2: auditoría y conclusiones

Se verificaron 50 corridas, 10900 registros por imagen y 2180 máscaras de semilla 11. Incidencias: 0.
Test: 218 imágenes, 122 grupos; máscaras vacías: 0.
Entrenamiento: NVIDIA L4; evaluación: Tesla T4. No comparar tiempos entre ambas GPUs como si fueran el mismo dispositivo.

| Red | Dice validación | Dice test ± DE semillas | IC95 test por grupos | IoU test | Parámetros |
|---|---:|---:|---:|---:|---:|
| transunet_2d | 0.8532 | 0.8626 ± 0.0057 | [0.8363, 0.8852] | 0.7835 | 105,322,001 |
| fcn | 0.8309 | 0.8535 ± 0.0136 | [0.8282, 0.8744] | 0.7683 | 72,145 |
| unet | 0.8320 | 0.8464 ± 0.0137 | [0.8177, 0.8702] | 0.7616 | 118,273 |
| vmunet | 0.8207 | 0.8391 ± 0.0297 | [0.8101, 0.8630] | 0.7526 | 44,273,833 |
| segnet | 0.8063 | 0.8247 ± 0.0334 | [0.7949, 0.8499] | 0.7346 | 85,761 |
| deeplabv3plus | 0.8133 | 0.8140 ± 0.0246 | [0.7784, 0.8438] | 0.7282 | 12,329,297 |
| unetplusplus | 0.8099 | 0.8076 ± 0.0059 | [0.7685, 0.8417] | 0.7264 | 15,970,449 |
| pidnet_s | 0.7963 | 0.7936 ± 0.0231 | [0.7526, 0.8287] | 0.7078 | 7,716,291 |
| segnext_t | 0.7594 | 0.7792 ± 0.0104 | [0.7406, 0.8124] | 0.6813 | 4,225,121 |
| segformer_b0 | 0.7025 | 0.7053 ± 0.0104 | [0.6613, 0.7439] | 0.5946 | 3,714,401 |

## Conclusión principal

TransUNet conserva el mayor promedio de Dice en test, mientras FCN pasa al segundo lugar y U-Net al tercero.
La diferencia principal TransUNet–FCN es 0.00906 (aproximadamente 0.91 puntos porcentuales). IC95 pareado por grupos: [-0.00828, 0.02779].
El intervalo incluye cero: esta evaluación no establece una mejora clara de TransUNet sobre FCN con este procedimiento. Tampoco demuestra equivalencia entre ambas.
FCN compacta es una referencia favorable en coste/tamaño: 72.145 parámetros frente a 105.322.001 de TransUNet. Una recomendación basada en parsimonia es defendible; no confundirla con declarar a FCN ganadora en Dice.
U-Net permanece competitiva. VM-UNet presenta mayor variación entre semillas. Los resultados bajos de SegFormer corresponden a esta adaptación desde cero y receta fija; no justifican inferioridad universal de los Transformers.

## Límites de interpretación

- Los IC remuestrean grupos con cinco semillas fijas; la DE entre semillas se presenta aparte. Los grupos de similitud no equivalen necesariamente a escenas de captura independientes.
- El contraste principal se fijó tras validación y antes de test. Los contrastes restantes son exploratorios y no tienen ajuste de multiplicidad.
- Test tuvo exposición en E1. Se conservaron los checkpoints y umbral 0,5 fijados antes de esta evaluación; no describirlo como holdout nunca observado.
- No atribuir diferencias sólo a arquitectura: capacidad, optimizador y auxiliares PIDNet también difieren; no hubo búsqueda exhaustiva de hiperparámetros.
- Ausencia de imágenes negativas limita la evaluación de falsas alarmas en escenas sin etiquetas, si el conteo de máscaras vacías es cero.
- Los tiempos registrados incluyen carga, transferencia y escritura de máscaras de semilla 11; no son una medición de latencia pura ni un benchmark de FPS.

## Qué procede

Conservar estos resultados como cierre del benchmark E2. Integrar tabla, contraste principal, curvas de validación y ejemplos de errores en el entregable. Mantener la réplica del paper guía y el cierre bibliográfico/PRISMA como requisitos separados. Cualquier nuevo tuning tras ver este test sería exploratorio y requeriría una evaluación independiente futura.

## Incidencias detectadas

[]
