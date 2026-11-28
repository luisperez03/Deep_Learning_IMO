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

## Despliegue publicado

El tablero está publicado en <https://hazmat13-e3.onrender.com>, desde el repositorio
[ddcarbono/hazmat13-e3](https://github.com/ddcarbono/hazmat13-e3) (licencia MIT, Python 3.11.9, versiones exactas,
Dockerfile). Cada cambio corre las pruebas, construye el contenedor y pasa la prueba de humo contra él; la prueba de humo
contra el servicio publicado está en {ref}`sec-anexo-humo`.

La página Modelos sigue el orden de la sustentación en cuatro pestañas:

1. **Benchmark:** tabla de las diez redes y del referente, IC por grupos, Pareto, diferencia crítica y Dice por tamaño.
2. **Modelo original:** mecanismo, hipótesis congelada, validación de las diez semillas por brazo y curvas.
3. **Validación estadística:** diferencias pareadas por semilla entre UNet-AT y su base, con Wilcoxon exacto e IC
   bootstrap, y la potencia del diseño con su diferencia mínima detectable.
4. **Predicción en vivo:** carga de una fotografía propia (FCN o U-Net a0498 en ONNX Runtime, CPU), con explicación
   local opcional por oclusión: se tapa cada parche del lienzo de 256 px con el gris de entrenamiento y se mide cuánto
   cae la probabilidad media dentro de la máscara predicha. Parches de 32 px en FCN y de 64 px en U-Net, para que la
   explicación quepa en el límite de 120 s del servidor gratuito. Incluye el diagnóstico por caso y el comparador de
   test.

En el sitio publicado no están la galería del EDA ni el comparador de test, porque necesitan las fotografías del
conjunto, que no se redistribuyen; en local se conectan con variables de entorno. Siguen fuera de la aplicación la
inferencia en vivo del original y la integración de VM-UNet en la comparación.
