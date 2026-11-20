# Reproducir la preparación previa a los modelos

Desde `tesis_imo_para_codex`, usar el Python del entorno con
`requirements-inicio.lock.txt` instalado. Los originales deben estar en `../VOC2012`.
No se necesitan PyTorch, GPU ni dependencias nuevas para esta preparación.

Si se parte de cero, ejecutar primero el flujo de `INICIO_REPRODUCIBLE.md`.
Después:

```powershell
python -m unittest discover -s tests -p test_preparacion.py -v
python scripts/preparar_base.py --data-root ".."
python scripts/informe_preentrenamiento.py
python scripts/preflight.py
```

`preparar_base.py` verifica hashes, marca cuarentenas, agrupa todos los candidatos,
crea particiones y copias binarias y estima normalización SOLO con train.
Es secuencial y carga una imagen a la vez. No entrena ni necesita mantener todo
el corpus en RAM. No transforma una caja en verdad terreno de segmentación.

El script rechaza entradas/configuraciones cambiadas respecto al manifiesto
existente, o cambios a los IDs de test. Para una nueva depuración, crear una
versión nueva del artefacto de preparación conservando el anterior. No borrar
el manifiesto congelado para forzar una nueva partición silenciosamente.

`preflight.py` devuelve error de proceso si falla integridad. Los pendientes de
aprobación se reportan en JSON y **no equivalen a entrenamiento autorizado**.
`training_allowed` y `training_started` permanecen falsos. Un exit code 0 solo
indica que las comprobaciones de integridad no fallaron.

Lectura guiada: `docs/PREENTRENAMIENTO.md` y `docs/PC_Y_PRESUPUESTO.md`.
El cargador `scripts/dataset_ready.py` carga un ejemplo de train/val bajo demanda;
test está bloqueado por defecto. La función de métricas se verifica con arrays
sintéticos; no se presentan esos tests como resultados de modelos.

## Lo que no está certificado

- Calidad visual individual de todas las máscaras; se aplicó QC conservador.
- Independencia completa de escenas: los grupos solo cubren similitudes conocidas.
- Selección final de los tres baselines según frecuencia en la literatura.
- Consumo/tiempo del entrenamiento ni temperatura del equipo.
- Entorno de entrenamiento instalado o arquitectura final aprobada.

No hay modificaciones a los originales, descargas de datos o modelos, ni ejecución
de redes. El diagnóstico inicial de PC guardado procede de una consulta real en esta
conversación. `scripts/diagnostico_pc.ps1` permite renovarlo con acceso local CIM.
