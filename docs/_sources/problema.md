# Problema y datos

En una inspección portuaria interesa localizar el rótulo antes de leer su símbolo o asignarle una clase IMO. La pregunta experimental es cuánto cambia el solapamiento de la máscara al cambiar el mecanismo de reconstrucción espacial, manteniendo datos, pérdida y presupuesto. La hipótesis operativa es que recuperar detalle del codificador podría beneficiar los contornos pequeños. Los tres contrastes se evalúan en ambos sentidos.

Una fotografía RGB pertenece a $X=[0,1]^{H\times W\times 3}$ y su máscara a $Y=\{0,1\}^{H\times W}$. El modelo $f_\theta$ produce un logit por píxel. Se minimiza una combinación de entropía cruzada binaria y pérdida Dice en el área válida. Una probabilidad de al menos 0,5 indica etiqueta. Las etiquetas contiguas pueden formar una sola región. No se predice identidad de instancia ni clase de mercancía.

## Corpus y anotaciones

La carpeta local `VOC2012` contiene 1.685 fotografías JPG y 1.583 máscaras semánticas utilizables. El nombre describe la organización local. La coincidencia del tamaño, las 13 categorías y los tipos de anotación con el repositorio de autores sustenta la identificación **probable** como HAZMAT-13. No se verificó identidad binaria con la descarga externa. Los autores describen fotografías de señales con cajas y máscaras. No disponemos de un registro local de anotadores, acuerdo entre ellos o criterio uniforme de oclusión. La licencia MIT figura en el repositorio. El alcance sobre imágenes alojadas externamente requiere confirmación antes de redistribuirlas. [Repositorio, documentación de obtención y aviso de licencia](https://github.com/mrl-amrl/HAZMAT13)

Los XML describen cajas y categorías. Para construir el objetivo binario se conserva la unión de los píxeles de las categorías anotadas en las máscaras. Una discrepancia de color puede afectar la clasificación aunque conserve la silueta. Una omisión de cobertura afecta directamente la segmentación. Hazmat Placards y SeaFront no participaron en las corridas. El test mide generalización interna, sin validación en una operación portuaria independiente ni en escenas exclusivamente negativas.

## Qué se congeló

De los 1.583 pares, 132 quedaron en cuarentena tras propagar alertas a todo el grupo. Los 1.451 elegibles se asignaron una sola vez con semilla 20261106, respetando componentes conexas del grafo de similitud.

| Partición | Imágenes | Grupos | Uso |
|---|---:|---:|---|
| Entrenamiento | 1.015 | 567 | Estimar normalización y pesos |
| Validación | 218 | 122 | Parada temprana y selección de checkpoint |
| Test | 218 | 122 | Una evaluación por corrida tras cerrar la configuración |

La imagen es la unidad de predicción. El grupo de similitud aproxima la unidad independiente. Cada grupo tiene el mismo peso en el Dice primario, con IoU de etiqueta, mIoU y exactitud como métricas secundarias. El protocolo previo propuso un margen absoluto de 0,02 Dice. Los hashes por imagen y máscara están en los manifiestos de `resultados/preentrenamiento`. El capítulo de resultados reproduce la huella de test registrada al evaluar. El EDA inicial examinó el corpus completo antes de la partición.

**Del grupo a la partición.** En `preparar_base.py`, cada clave de `eligible` es un grupo completo. La asignación devuelve una sola partición por clave. Las cifras de la tabla se pueden comprobar en `train.csv`, `val.csv` y `test.csv`, contando filas para las imágenes y valores distintos de `group_id` para los grupos.

```{literalinclude} ../scripts/preparar_base.py
:language: python
:lines: 105-108
:dedent: 4
```
