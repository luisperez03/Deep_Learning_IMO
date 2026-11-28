(sec-introduccion)=
# Introducción

## Problema y motivación

Identificar el rótulo de una mercancía peligrosa a partir de una fotografía es un paso previo a leer su clase de riesgo.
La segmentación binaria separa los píxeles del rótulo del fondo. La tarea es de objetos pequeños, de forma regular
(rombos y rectángulos con bordes definidos) y con pocas imágenes (1.015 de entrenamiento, 218 de validación y 218 de
prueba en la partición heredada del E2).

## Contribuciones

1. **Benchmark** de diez redes de segmentación entrenadas desde cero, con 10 semillas, pruebas pareadas por grupo con
   Holm, tamaños de efecto y diferencia mínima detectable reportados ({ref}`sec-resultados`).
2. **Medición de una limitación**: las diez redes pierden entre 0,053 y 0,120 de Dice en el tercil de menor área
   (piloto del E2, `plan.md`, decisión 8) ({ref}`sec-discusion`).
3. **UNet-AT**, un modelo original que transfiere la representación de DBNet a la segmentación de rótulos con dos
   reglas propias, C1 y C2 ({ref}`sec-metodo`).
4. **Teorema de inversión exacta**: C2 invierte exactamente a C1 y a la regla de DBNet en polígonos convexos, con
   supuestos explícitos y verificación numérica ({ref}`sec-teorema`).
5. **Paquete de reproducibilidad**: protocolo congelado, contenedor, integración continua, dashboard y ficha del modelo
   ({ref}`sec-reproducibilidad`).

## Organización

El capítulo de trabajo relacionado ({ref}`sec-trabajo-relacionado`) ubica el trabajo; el método ({ref}`sec-metodo`)
presenta el modelo; el diseño experimental ({ref}`sec-diseno`) fija el protocolo; los capítulos siguientes reportan
trazabilidad, novedad, resultados, ablación, robustez y exportación; el cierre discute, declara limitaciones y entrega
la lista de verificación de reproducibilidad.
