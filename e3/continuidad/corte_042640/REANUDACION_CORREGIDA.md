# Corrección de reanudación E3 v4

## Defecto y corrección

La etapa 05 añadía `busqueda_arquitectura_e2.py` y `complemento_entreno_e2.py` a `PROJECT/scripts`.
El arnés firma todos los `.py` de esa carpeta. Una máquina nueva que omitía 05 por estar completa
no recuperaba esos archivos y no podía reanudar las corridas de 06 con su identidad original.

El lanzador actualizado separa `prepare_sources()` del entrenamiento. Tras extraer y verificar
el paquete, antes del entorno y del preflight, lee **el literal histórico FUENTES_E2** del cuaderno
congelado; nunca ejecuta 05 para recrear sus efectos. Recupera el perfil base o ampliado de acuerdo
con `identity.source` de los configs/preflight existentes. Para nuevas A/B usa el perfil ampliado.
Si hay firmas incompatibles, fuentes desconocidas o no puede reproducir exactamente la firma,
se detiene. No cambia fingerprints para hacerlos coincidir.

Firmas verificadas localmente:

- Base: `59a30d289ce7995b64b362d60e7859bf3445fa9e576e479abcca6b3f4ac77a62`, coincide con las diez configuraciones reales de SegFormer recibidas.
- Ampliada: `417615c225083c0d6be663d84f9b5f19f63868e80b1db64022d669a33cd01f22`, reproducida tras eliminar los efectos de 05 y restaurarlos desde su literal congelado.

El último ZIP no contiene todavía config/checkpoint de una corrida de A: su identidad real se
comprobará al ejecutar sobre Drive. Una incompatibilidad adicional de GPU, versión de Python o
paquetes seguirá siendo rechazada por el arnés; esta corrección no la oculta.

## Entorno y registros

Se incorpora y comprueba `setuptools==84.2.1` en el entorno aislado de rivales, antes de cargar Mamba.
Los subprocesos se ejecutan con salida sin búfer de Python y ambas tuberías se muestran en tiempo
real. Los registros se guardan en:

`MyDrive/tesis_imo_e3_secuencia_v4/logs/<fecha_carril_identificador>/<etapa>.stdout.log`

y el archivo correspondiente `.stderr.log`. Se vacían los búferes al escribir; los errores y
códigos de salida se conservan. `progreso_<carril>.json` registra las rutas de los logs y la firma recuperada.

No se cambiaron archivos del entrenador que participan en la firma, payloads históricos,
checkpoints, hiperparámetros, datasets, arquitecturas ni semillas. No se regeneraron los diez
cuadernos congelados; se regeneró solo el lanzador autocontenido que los ejecuta.
Los cuadernos individuales antiguos no incorporan esta reparación: usar el lanzador actualizado.

## Continuar sin subir otro cuaderno

Si una celda todavía está entrenando, dejarla terminar o esperar al punto de relevo acordado.
No lanzar un segundo proceso sobre la misma campaña. Después de un fallo de reanudación, ejecutar
esta celda en el cuaderno que ya tiene la GPU:

```python
import os
from urllib.request import urlopen
url = 'https://raw.githubusercontent.com/luisperez03/Deep_Learning_IMO/e3-reanudacion-20261128-v2/e3/continuidad/E3_CONTINUAR_EN_ESTA_MAQUINA.py'
codigo_e3 = urlopen(url).read().decode('utf-8')
os.environ['E3_CARRIL'] = 'todo'
exec(compile(codigo_e3, 'lanzador_e3_corregido', 'exec'), {'__name__': '__main__'})
```

Conserva la raíz de Drive y las etapas completas. No borres `config.json`, `last.pt`, `best_epoch_*.pt`
ni cambies fingerprints a mano. Si vuelve a detenerse, conservar los nuevos logs para identificar
qué parte exacta de la identidad difiere.

## Comprobaciones

16 pruebas locales de orquestación, reanudación de fuentes, inmutabilidad de los payloads,
salida en vivo, persistencia de errores y auditoría de campaña. No se ejecutó entrenamiento GPU
en estas pruebas. La evidencia está en `pruebas_reanudacion.txt` y `verificacion_firmas_historicas.json`.

## Revisión v2: preflight regenerado

El error «varias firmas históricas» podía resultar de mezclar la firma de un `config.json` de corrida con la de un `preflight.json` regenerado tras un intento fallido. Ahora los configs de las corridas tienen prioridad. Si solo el preflight difiere, se guarda una copia en los logs, se recupera la firma del config y se permite regenerar el preflight mediante su comprobación normal. No se edita el fingerprint ni el checkpoint.

Si difieren entre sí los configs de corridas, la ejecución se detiene y muestra las rutas y firmas precisas: no selecciona una arbitrariamente. El diagnóstico muestra también GPU y Python históricos. Una L4 no se considera automáticamente compatible con un checkpoint creado en T4; el arnés conserva su validación completa del entorno.

El registro recibido no incluye las firmas concretas ni prueba que el conflicto sea solo del preflight. La revisión cubre ese caso con una prueba de regresión y mantiene la protección para conflictos reales. Si vuelve a detenerse, compartir el bloque «Diagnóstico de identidades».
