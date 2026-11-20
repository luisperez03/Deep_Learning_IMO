"""Informe y figuras finales de preparación, sin ejecutar modelos."""
import json
from pathlib import Path
from inicio_voc import PROJECT
from graficos_eda import bars

def main():
    out=PROJECT/'resultados/preentrenamiento'
    data=json.loads((out/'resumen.json').read_text())
    freeze=json.loads((out/'freeze_manifest.json').read_text())
    plots=out/'graficos';plots.mkdir(exist_ok=True)
    counts=data['status_counts'];splits=data['splits']
    bars(['Elegibles por control automatico','Cuarentena: grupo con alerta','Sin mascara utilizable'],
         [counts['eligible_automatic_qc'],counts['quarantine'],counts['excluded_no_usable_mask']],
         'Base de trabajo conservadora','Elegible no equivale a anotacion certificada manualmente').save(plots/'01_calidad.png')
    names=['train','val','test']
    bars(names,[splits[k]['images'] for k in names],
         'Particiones por grupos de similitud','Una asignacion, semilla fija; sin optimizar con resultados de test').save(plots/'02_particiones.png')
    bars(names,[splits[k]['groups'] for k in names],
         'Unidades agrupadas por particion','Grupos conservadores; no se certifica independencia de todas las escenas',unit='grupos').save(plots/'03_grupos.png')
    text=f'''# Preparación técnica antes de los modelos base

**Estado: datos derivados y protocolo preparados; entrenamiento deshabilitado.**
Esta versión es una propuesta reproducible para revisión, no una certificación de
calidad perfecta ni una selección bibliográfica final de modelos.

![Calidad](../resultados/preentrenamiento/graficos/01_calidad.png)

![Particiones](../resultados/preentrenamiento/graficos/02_particiones.png)

![Grupos](../resultados/preentrenamiento/graficos/03_grupos.png)

## Resultado verificado

- {counts['eligible_automatic_qc']} pares técnicamente elegibles; {counts['quarantine']} en cuarentena.
- {counts['excluded_no_usable_mask']} imágenes sin máscara utilizable.
- Train: {splits['train']['images']} imágenes / {splits['train']['groups']} grupos.
- Validación: {splits['val']['images']} imágenes / {splits['val']['groups']} grupos.
- Prueba: {splits['test']['images']} imágenes / {splits['test']['groups']} grupos.
- Enlaces candidatos conocidos que cruzan particiones: {data['known_candidate_edges_crossing_splits']}.
- Grupos de similitud en todo el corpus: {data['all_similarity_groups']}; tamaño máximo {data['largest_similarity_group']}.

La cuarentena propaga alertas a las imágenes relacionadas: discrepancias semánticas,
posibles omisiones y cajas con muy poca cobertura de máscara. Es una política
conservadora que puede excluir siluetas útiles; no afirma que todos esos casos sean
errores. Está registrada y se puede revisar sin tocar los originales.

## Artefactos de esta versión

`resultados/preentrenamiento/manifest.csv` registra grupo, estado, motivo, split,
rutas y hashes. `train.csv`, `val.csv`, `test.csv` son sus vistas de trabajo.
`masks_binary/` conserva copias 0/1 a resolución original; no son cajas convertidas
en máscaras. Solo se cargan los archivos listados en el manifiesto, nunca glob
de todas las copias derivadas.

Normalización calculada únicamente sobre train, después de letterbox bilineal
256×256 y excluyendo padding: `normalization_train.json`. El cargador convierte
RGB a CHW float32, mantiene máscaras 0/1 y devuelve la zona válida. No se activa
augmentación ni se cachea el dataset completo en RAM.

El diagnóstico en train compara 192, 256 y 320: ningún primer plano completo
desaparece ni queda por debajo de 16 píxeles totales. Esto NO demuestra que cada
etiqueta pequeña individual sobreviva: el informe es por imagen, no por componente.
256 es una resolución candidata sujeta a revisar detalle y costo.

## Identidad del test reservado

SHA256 de test.csv: `{freeze['artifacts']['test.csv']}`.

El generador rechaza cambios silenciosos de entradas/configuración y de IDs del
test existente. Cualquier depuración posterior exige una nueva versión documentada.
El cargador niega el acceso al test por defecto. No hay evaluación de modelos.

## Validaciones antes de autorizar entrenamiento

1. Revisar agrupamiento conservador y política de cuarentena. No afirmar ausencia
   absoluta de fuga: faltan escenas verificadas y variantes no detectadas por hashes.
2. Aprobar margen delta, métricas, hipótesis, arquitectura y presupuesto experimental.
3. Resolver selección de tres baselines mediante literatura; los candidatos actuales
   no están demostrados como los tres más recurrentes en HAZMAT.
4. Validar recursos del PC y entorno PyTorch CPU aislado. No instalar ni arrancar
   modelos sin acordarlo. Plan de prueba mínima descrito en PC_Y_PRESUPUESTO.md.

Guías: `PREENTRENAMIENTO_REPRODUCIBLE.md`, `PROTOCOLO_PREVIO.md`,
`PC_Y_PRESUPUESTO.md`, `PROCEDENCIA_Y_LITERATURA.md`.
'''
    (PROJECT/'docs/PREENTRENAMIENTO.md').write_text(text,encoding='utf-8')
    print('Informe y tres figuras regenerados')

if __name__=='__main__':main()
