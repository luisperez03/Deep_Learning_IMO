(sec-conclusiones)=
# Conclusiones del estado verificado

```{admonition} Actualización: paquete de Colab recibido
:class: note
Se integraron 20 corridas del original y 9 de SegFormer, verificadas contra sus checkpoints.
Las tablas, curvas y la lectura de estos resultados están en {ref}`sec-avance-colab`.
Las menciones a evaluación pendiente se refieren al cierre final, no a la ausencia de entrenamiento.
```


El benchmark E2 proporciona una referencia reproducible de diez redes. El mayor Dice puntual corresponde a una arquitectura considerablemente más grande que la FCN histórica. Esto justifica estudiar conjuntamente segmentación, dispersión, tamaño y costo de inferencia, sin confundir orden de medias con superioridad.

Se recuperaron y verificaron quince checkpoints de las compactas a0498. FCN y U-Net superaron la comprobación ONNX en validación; SegNet conserva un respaldo TorchScript verificado. El tablero local compara diez redes sobre los 218 casos de test y distingue resultados históricos, artefactos seleccionados y predicciones pendientes.

La discrepancia geométrica se reprodujo y se explicó por la versión de rasterización de OpenCV, sin relajar la tolerancia. Esto resuelve un problema de reproducibilidad de la implementación; no prueba eficacia del mecanismo ni novedad.

Las diez semillas de UNet-AT arrojan Dice medio de validación 0,7862, frente a 0,8077 de la base. No respaldan una mejora global en validación. El cierre científico permanece abierto hasta completar evaluación y ablaciones. Tampoco están acreditados los tres conjuntos adicionales, la potencia requerida ni la revisión independiente de novedad y demostración. El libro no declara superioridad del original ni cumplimiento completo del E3 a partir de pruebas técnicas. La conclusión final deberá responder a la hipótesis con efectos e intervalos, incluyendo resultados desfavorables si se presentan.
