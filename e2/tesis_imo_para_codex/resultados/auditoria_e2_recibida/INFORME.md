# Auditoría de las 50 corridas E2 recibidas

Carpeta leída: `C:\Users\luisa\Downloads\DEEP LEARNING\tesis_imo_e2_arquitecturas_reales_v2`. No se modificaron corridas ni se ejecutó inferencia de test.
Corridas: 50; modelos: 10; incidencias: 0.
Validación: 218 imágenes, 122 grupos. Hashes de checkpoints comprobados: 100.
Variantes de configuración/identidad/entorno: 1/1/1.

| Modelo | Dice medio ± DE | IoU medio | Parámetros | Horas registradas | Corridas a 30 épocas |
|---|---:|---:|---:|---:|---:|
| transunet_2d | 0.8532 ± 0.0064 | 0.7691 | 105,322,001 | 3.46 | 1/5 |
| unet | 0.8320 ± 0.0165 | 0.7441 | 118,273 | 0.74 | 2/5 |
| fcn | 0.8309 ± 0.0071 | 0.7411 | 72,145 | 0.80 | 4/5 |
| vmunet | 0.8207 ± 0.0300 | 0.7308 | 44,273,833 | 2.91 | 1/5 |
| deeplabv3plus | 0.8133 ± 0.0231 | 0.7210 | 12,329,297 | 1.03 | 0/5 |
| unetplusplus | 0.8099 ± 0.0107 | 0.7223 | 15,970,449 | 1.00 | 0/5 |
| segnet | 0.8063 ± 0.0321 | 0.7124 | 85,761 | 0.53 | 1/5 |
| pidnet_s | 0.7963 ± 0.0252 | 0.7028 | 7,716,291 | 2.29 | 1/5 |
| segnext_t | 0.7594 ± 0.0105 | 0.6576 | 4,225,121 | 2.04 | 1/5 |
| segformer_b0 | 0.7025 ± 0.0091 | 0.5990 | 3,714,401 | 1.60 | 1/5 |

DE: desviación estándar muestral entre cinco semillas, no intervalo de confianza. Cada puntuación corresponde al mejor checkpoint seleccionado por validación agrupada. Las horas suman entrenamiento + validación por época; no incluyen necesariamente guardado, instalación, interrupciones o cobro total.

## Interpretación y pasos siguientes

- TransUNet lidera esta validación; no se afirma superioridad general ni en test. U-Net y FCN compactas ofrecen una relación tamaño/rendimiento favorable.
- No seleccionar la mejor semilla para presentar el modelo: mantener las cinco y reportar dispersión.
- No concluir que Transformer/SSM son inferiores por los resultados de una receta desde cero y presupuesto limitado. Optimizadores, capacidades y pérdidas auxiliares difieren.
- Congelar los 50 checkpoints y recetas. Revisar errores cualitativos en validación (casos comunes, etiquetas pequeñas, ausencia de etiqueta y bordes).
- Preparar evaluación final de test de las diez redes con las cinco semillas, mismos grupos/métricas y umbral 0,5; ejecutar tras autorización explícita del cierre. Declarar exposición previa al test del E1.
- Después: comparación pareada por grupos/semillas, incertidumbre y coste, figuras y actualización del libro. No usar test para nuevas decisiones de hiperparámetros.
- La réplica del paper guía y el cierre de literatura/PRISMA siguen siendo tareas separadas.

## Incidencias

[]

Comprobación adicional de procedencia: la identidad reconstruida de las fuentes del ZIP v2 entregado coincide con la registrada en las corridas: `59a30d289ce7995b64b362d60e7859bf3445fa9e576e479abcca6b3f4ac77a62`.
