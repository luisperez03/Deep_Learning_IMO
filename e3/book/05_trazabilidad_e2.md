(sec-trazabilidad)=
# Trazabilidad de las correcciones del E2

Al 27 de noviembre no se ha recibido retroalimentación docente del E2. Por tanto, no se atribuyen al docente las correcciones técnicas realizadas durante la recuperación del proyecto. Cuando llegue la revisión se añadirá cada observación con su respuesta y evidencia.

## Correcciones técnicas verificadas

| Hallazgo | Acción y evidencia | Efecto sobre la interpretación |
|---|---|---|
| Artefactos dispersos | Recuperación local de 50 checkpoints históricos y 15 compactas a0498; comprobación de las huellas de las compactas | Permite reproducir los artefactos sin asumir rutas de Drive inexistentes |
| Configuraciones históricas distintas de a0498 | Separación explícita en tablas, libro y tablero | Evita atribuir resultados antiguos a arquitecturas nuevas |
| Discrepancia geométrica | Reproducción con OpenCV 4.10 y 4.11; 21 comparaciones ampliadas aprobadas con 4.11 | Se fija la versión sin relajar tolerancias |
| Exportación SegNet sin paridad de probabilidades | ONNX deshabilitado para carga de imágenes; respaldo TorchScript verificado | Una exportación existente no se considera aprobada automáticamente |
| Reanudación de entrenamiento | Prueba reducida en CPU con pausa y continuación | Verifica el mecanismo técnico, no determinismo general entre dispositivos |
| Inspección de test | Comparador aleatorio con arquitectura, semilla y hashes | Se registra la exposición al test como exploración visual |

## Pendientes que requieren evidencia adicional

La auditoría final debe reconciliar particiones, semillas, configuraciones y política de entrenamiento con el paquete de la campaña. El tag de congelamiento no está verificado localmente. La tabla antes/después del desempeño corregido se elaborará desde resultados auditados; no se infiere una mejora por recuperar pesos o habilitar una aplicación.

La equivalencia con el archivo histórico `etiquetas.py` permanece sin ejecutar porque ese archivo no se recuperó. Este faltante de procedencia se conserva separado de la discrepancia numérica de OpenCV ya resuelta.
