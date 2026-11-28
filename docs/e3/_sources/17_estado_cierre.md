(sec-cierre)=
# Integración y cierre del entregable

```{admonition} Actualización: paquete de Colab recibido
:class: note
Se integraron 20 corridas del original y 9 de SegFormer, verificadas contra sus checkpoints.
Las tablas, curvas y la lectura de estos resultados están en {ref}`sec-avance-colab`.
Las menciones a evaluación pendiente se refieren al cierre final, no a la ausencia de entrenamiento.
```


Este registro permite incorporar los resultados de Colab sin confundir una ejecución terminada
con una conclusión científica. Estado: 27 de noviembre de 2026.

| Componente | Evidencia disponible | Condición de cierre |
|---|---|---|
| Benchmark histórico E2 | Tabla, dispersión e intervalos de diez redes | Conservar su identificación histórica |
| Compactas a0498 | Quince pesos verificados; FCN/U-Net ONNX y SegNet TorchScript | Incorporar campaña y evaluación finales |
| Geometría del original | 21 comparaciones ampliadas; OpenCV fijado | Revisión formal y recuperación del código histórico faltante |
| Entrenamiento del original | Piloto y prueba reducida de reanudación | Auditar las veinte corridas de ambos brazos |
| Rivales E3 | Secuencia preparada para completar cien corridas | Verificar semillas, recetas, hashes y curvas |
| Comparación visual | Nueve redes, 218 casos por red; manifiestos verificados | Integrar VM-UNet y el original |
| Estadística, ablación y robustez | Protocolo y capítulos preparados | Incorporar resultados, efectos, intervalos y supuestos |
| Reproducibilidad | Fuentes locales, registros y construcción del libro | Reproducción independiente y versión pública identificable |
| Evaluación adicional y novedad | Requisitos y limitaciones documentados | Ejecutar los conjuntos adicionales y completar revisión |
| Publicación | Tablero y libro locales | Verificar entrega publicada y permisos de los datos |

## Secuencia cuando llegue un paquete de Colab

1. Conservar el paquete recibido y registrar su huella. Identificar el carril, la versión y las corridas incluidas.
2. Auditar configuración, semillas, checkpoints, particiones y completitud. Una carpeta existente no acredita una corrida completa.
3. Revisar validación y la política de selección antes de incorporar una evaluación final. No ajustar recetas a partir del comparador de test.
4. Generar tablas desde los archivos auditados. Mantener diferenciadas las arquitecturas históricas y a0498.
5. Integrar artefactos de inferencia solo después de comprobar paridad. Si falla una exportación, conservar el resultado y su diagnóstico.
6. Actualizar resultados, ablación, robustez, discusión y conclusiones según la evidencia; recompilar el libro y comprobar su consistencia con el tablero.

Completar el entrenamiento original no resuelve por sí solo las evaluaciones adicionales, la potencia,
la novedad o la política final uniforme. Esas condiciones permanecen visibles hasta que exista evidencia.

## Lectura local

El libro compilado se abre desde `book/_build/html/intro.html`; su navegación no requiere el servidor
del tablero. El comparador interactivo sí requiere el servidor local en `http://[::1]:8051/modelos`.
`COMPILAR_LIBRO.cmd` selecciona el kernel local del libro. Las versiones efectivas de ese entorno
se conservan en `entorno_libro_local.freeze.txt`; no deben confundirse con las del entrenamiento en Colab.
