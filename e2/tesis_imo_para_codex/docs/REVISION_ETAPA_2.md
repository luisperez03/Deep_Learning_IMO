# Segunda etapa: similitud, origen y máscaras

Fecha: 2026-11-06. Evidencia local regenerada por `scripts/revisar_voc.py`.
Revisión visual exploratoria: asistente Codex, no anotador humano independiente.

## Resultado computacional

- Se compararon exhaustivamente los 1.418.770 pares posibles de 1.685 JPG.
- 748 pares cumplen distancia dHash <= 6 y aHash <= 6 (64 bits cada uno).
- 146 de esos pares conectan train y val originales. Son candidatos de similitud,
  no 146 duplicados certificados ni 146 escenas independientes.
- No se encontraron grupos de píxeles RGB exactamente idénticos. Esto complementa
  la ausencia previa de archivos idénticos; no contradice el parecido visual.
- Hay siete valores conocidos de `folder` XML y un grupo `unknown`. Las carpetas
  no se interpretan como escenas, personas o cámaras sin evidencia adicional.
- La validación original se concentra en `00_javad` (33) y `00_misaq` (151).
  Además hay coincidencias visuales entre carpetas: agrupar solo por `folder`
  no garantiza eliminar fuga.

Los hashes originales de JPG y metadatos se verificaron contra la auditoría previa.
Los paneles ampliados también verifican el hash de cada máscara que utilizan.

## Revisión de los 12 pares mostrados

Se inspeccionaron `pares_01.jpg`, `pares_02.jpg` y `pares_03.jpg` completos. Los
12 pares seleccionados corresponden a train/val y comparten firma perceptual.
Visualmente muestran composiciones muy semejantes: objetos, rótulos y fondos en
posiciones coincidentes, con pequeñas diferencias de apariencia o encuadre.
Ejemplos claros: image_0083/image_1591, image_0109/image_1575 e
image_0300/image_1681. No se certifica identidad exacta ni se extrapola esta
revisión a los 748 candidatos.

**Decisión:** las particiones heredadas no deben usarse aún como prueba de
generalización independiente. Proponer grupos de capturas relacionadas después
de adjudicar candidatos, preservando imágenes y anotaciones originales.

## Revisión ampliada de máscaras

Se generaron 12 paneles sin reducción de los píxeles originales. Se inspeccionaron
los seis casos prioritarios siguientes; los otros seis quedan preparados para revisión.
El visor puede reducir un panel grande, aunque el PNG conserva su resolución.

| Caso | Observación | Acción propuesta |
|---|---|---|
| image_0000 | Las dos etiquetas visibles dicen organic peroxide; colores semánticos distintos | Revisar asignación de clase; preservar siluetas hasta adjudicar |
| image_0011 | Hay etiquetas infectious substance y spontaneously combustible; XML y colores presentan discrepancias | No corregir máscaras copiando ciegamente las cajas |
| image_0025 | Rótulo infectious substance cubierto por máscara con otro color de clase | Revisar semántica; cobertura binaria aparentemente utilizable |
| image_1461 | Rótulo visible dentro de la cesta aparentemente omitido; dos rótulos inferiores sí tienen máscara | Alerta de cobertura incluso para segmentación binaria |
| image_0677 | Se ven cuatro regiones anotadas, con oclusiones y corte en el borde | Revisar criterio de píxeles visibles y continuidad a resolución original |
| image_1251 | Varias etiquetas y oclusión por mano; recortes en los bordes | Fijar regla de oclusión y cobertura antes de aprobar |

No se corrigieron ni excluyeron máscaras automáticamente. Esta selección es
dirigida a casos problemáticos y no permite estimar una tasa poblacional de error.

## Qué aportan los gráficos nuevos

El cuaderno exporta 16 figuras: cobertura, splits, consistencia semántica,
superficie, aspecto, resoluciones, etiquetas pequeñas, balance de píxeles,
presencia/área de clases, similitud, igualdad RGB, origen, origen de validación,
brillo y contraste. Cada bloque explica unidades, denominadores, interpretación,
decisión y riesgo. Brillo y contraste se calculan en miniaturas 128×128 para EDA;
no son estadísticas de normalización para entrenamiento.

## Pendiente antes de dividir y entrenar

1. Adjudicar candidatos y consolidar grupos; buscar variantes que los hashes pueden
   omitir, como rotaciones y recortes. No confundir componentes de un grafo de
   similitud con escenas verdaderas.
2. Resolver cobertura de máscaras y documentar reglas de oclusión. Revisión humana
   independiente, especialmente de las discrepancias y casos de omisión.
3. Verificar fuente/licencia y literatura; no se consultaron fuentes externas en
   esta etapa ni se descargaron datos.
4. Proponer splits por grupos, después congelarlos con hashes y registrar protocolo.
   Completar EDA de regiones y comparación de distribuciones sobre esos splits.

Los originales permanecen intactos. Todavía no hay modelos, piloto ni test final.
