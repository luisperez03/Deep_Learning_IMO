(sec-exportacion)=
# Exportación y tablero de comparación

```{admonition} Actualización: paquete de Colab recibido
:class: note
Se integraron 20 corridas del original y 9 de SegFormer, verificadas contra sus checkpoints.
Las tablas, curvas y la lectura de estos resultados están en {ref}`sec-avance-colab`.
Las menciones a evaluación pendiente se refieren al cierre final, no a la ausencia de entrenamiento.
```


## Inferencia verificada

FCN y U-Net a0498 ejecutan inferencia local con ONNX Runtime. La comparación contra PyTorch cubrió las 218 imágenes de validación, con tolerancia absoluta de probabilidad de 1e-4 y coincidencia de máscaras. La aplicación verifica la huella del artefacto antes de utilizarlo. Los detalles y las semillas seleccionadas figuran en {ref}`sec-compactas-integradas`.

SegNet no superó la tolerancia de probabilidades en ONNX: el error máximo fue 0,0212132. Su respaldo TorchScript produjo error cero en validación. Por ello no se habilita SegNet en la carga de fotografías de la aplicación ONNX; sí se muestran sus predicciones precalculadas con PyTorch en la comparación de test.

## Una misma imagen para todas las redes

En la página Modelos, **Elegir imagen aleatoria de test** selecciona entre los 218 casos registrados. Se muestran la fotografía, la máscara real en verde y las predicciones en naranja. TransUNet ocupa la columna izquierda como referente por el mayor Dice medio de validación del E2; las demás redes aparecen a su lado. No se declara un ganador de la campaña E3.

Las predicciones precalculadas cubren diez redes: TransUNet, FCN, U-Net, SegNet, DeepLabV3+, U-Net++, PIDNet-S, SegNeXt-T, SegFormer-B0 y UNet-AT. FCN, U-Net y SegNet usan a0498; las otras seis corresponden a checkpoints históricos. UNet-AT usa la semilla 11 con C2 completo; VM-UNet permanece pendiente en esta modalidad. Cada tarjeta identifica arquitectura y semilla; Dice e IoU son métricas de esa imagen a 256 × 256, excluyendo el relleno.

Cada manifiesto registra las huellas de pesos, imágenes, máscaras reales y predicciones. Un archivo ausente o que falla la comprobación no se sustituye por otra red. Esta inspección constituye exposición al test y se declara como exploración visual; no se utiliza para ajustar recetas ni escoger semillas.

## Arquitectura y ejecución local

Dash organiza páginas de contexto, EDA y modelos. Los resultados históricos se leen de las mismas tablas que utiliza este libro. La comparación visual lee máscaras precalculadas y no carga diez modelos en memoria. La carga de una fotografía propia usa los artefactos ONNX habilitados de FCN o U-Net.

En el equipo de trabajo, `ABRIR_DASHBOARD.cmd` conecta el paquete local y sirve el tablero en `http://[::1]:8051/modelos`. Se verificaron respuestas HTTP 200 de página, layout, callbacks y estilos. El servidor debe permanecer activo. Las 29 pruebas de aplicación, consistencia, inferencia y comparador aprobaron antes de esta revisión editorial. Esta comprobación local no acredita un despliegue público.

## Pendientes de entrega

La ficha ya está disponible en `assets/ficha_modelo.html`. Faltan los artefactos finales de la campaña, la integración de VM-UNet en la demostración y la inferencia en vivo del original, los autores definitivos y la verificación del despliegue publicado. `render.yaml` prepara una configuración, pero no prueba que Render haya ejecutado la aplicación. La galería pública también depende de resolver los permisos de redistribución de fotografías.
