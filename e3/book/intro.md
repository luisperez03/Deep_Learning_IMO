# Resumen: UNet-AT y benchmark de rótulos HAZMAT

```{admonition} Actualización: paquete de Colab recibido
:class: note
Se integraron 20 corridas del original y 9 de SegFormer, verificadas contra sus checkpoints.
Las tablas, curvas y la lectura de estos resultados están en {ref}`sec-avance-colab`.
Las menciones a evaluación pendiente se refieren al cierre final, no a la ausencia de entrenamiento.
```


**Estado al 27 de noviembre de 2026.** Informe en integración. Las cifras históricas proceden de `data/tabla_maestra.csv`. Se recuperaron las compactas seleccionadas, se verificaron dos exportaciones ONNX y un respaldo TorchScript, y se añadió una comparación visual de test al tablero. Las veinte corridas principales del original ya se integraron; su evaluación final y las ablaciones siguen pendientes. No se ha recibido retroalimentación docente sobre E2.

**Problema.** Segmentación binaria de rótulos de mercancías peligrosas en fotografías (HAZMAT-13
{cite:p}`sharifi2021hazmat`), con un benchmark de diez redes entrenadas desde cero y un modelo original, UNet-AT.

**Método.** UNet-AT transfiere a esta tarea la representación de núcleo encogido y binarización diferenciable de
DBNet {cite:p}`liao2020dbnet` sobre la U-Net compacta del benchmark, con dos reglas propias: encogimiento atenuado por
grosor (C1) y expansión consistente en inferencia (C2). El argumento de inversión en polígonos convexos se presenta con sus supuestos en {ref}`sec-teorema`; su revisión formal y la revisión de novedad permanecen pendientes.

**Resultados.** Pendiente: la comparación del original con los diez rivales, la ablación y la robustez dependen de
entrenar los dos brazos del original con 10 semillas (bloque D del plan). El benchmark de los rivales sale de los
resultados del E2 ({ref}`sec-resultados`).

**Alcance y límites.** No hay evaluación integrada en tres conjuntos adicionales. La potencia recuperada usa una referencia anterior y debe revisarse con las configuraciones seleccionadas y evidencia del original. Diez semillas no garantizan potencia de 0,90. La evidencia actual no acredita R3 ({ref}`sec-limitaciones`).

```{tableofcontents}
```
