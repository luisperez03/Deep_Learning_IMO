"""Genera colab/Replica_SegFormer_ADE20K.ipynb (ítem 32), determinista.

Incrusta scripts/replica_segformer_ade.py y config/replica_segformer.json con su
SHA-256; el cuaderno los escribe en /content/replica y verifica el hash antes de
usarlos. Regenerar después de cambiar cualquiera de los dos:
  python scripts/crear_colab_replica.py
"""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / 'scripts/replica_segformer_ade.py'
PROTOCOLO = PROJECT / 'config/replica_segformer.json'
SALIDA = PROJECT / 'colab/Replica_SegFormer_ADE20K.ipynb'


def sha(texto):
    return hashlib.sha256(texto.encode('utf-8')).hexdigest()


def md(texto):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': texto.strip('\n').splitlines(True)}


def code(texto):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': texto.strip('\n').splitlines(True)}


def celdas():
    script = SCRIPT.read_text(encoding='utf-8')
    protocolo = PROTOCOLO.read_text(encoding='utf-8')
    p = json.loads(protocolo)
    ds = p['dataset_descarga']
    return [
        md(f"""
# Réplica del paper guía: SegFormer-B0 en ADE20K (ítem 32)

Evalúa el checkpoint publicado de SegFormer-B0 en las 2.000 imágenes de validación de ADE20K, escala única,
con el preprocesamiento de la configuración oficial, y lo compara con el **37,4 % de mIoU** del artículo
según el criterio fijado antes en `config/replica_segformer.json` (δ = 1 punto, IC95 bootstrap por imagen).

- **Qué es:** réplica de **evaluación** (nivel 1). Valida datos, preprocesamiento y métrica; **no** replica el
  entrenamiento. El ajuste corto desde ImageNet (nivel 2) está solo diseñado en el protocolo y este cuaderno no lo ejecuta.
- **Checkpoint:** `{p['checkpoint']['fuente_principal'].split(',')[0]}`, archivo `model.safetensors` con SHA-256 verificado.
- **Datos:** `{ds['url']}` ({ds['tamano_bytes_observado']:,} bytes). Términos de uso: {ds['licencia_y_terminos']}
- **Tiempo estimado en T4 (sin medir aún):** descarga 3-10 min según el servidor del MIT, extracción de validación
  1-2 min, carga del modelo < 1 min, inferencia 3-6 min por variante (2 variantes), bootstrap < 1 min.
  Total previsto 15-30 min. La celda 5 mide 50 imágenes y extrapola antes de la corrida completa.
- **Salidas:** `resumen_replica.json` (veredicto, mIoU, IC95, IoU por clase, desviaciones D1, D2, D4, D5, hashes) y
  `areas_*.npz` (intersección y unión por imagen y clase, para rehacer el bootstrap sin GPU).

Orden: ejecutar las celdas de arriba abajo. Entorno: GPU T4 (Entorno de ejecución > Cambiar tipo > T4).
"""),
        code("""
# 1. Parámetros. Leer los términos de ADE20K antes de poner True.
ACEPTO_TERMINOS_ADE20K = False      # https://ade20k.csail.mit.edu/terms/
SHA256_ADE_FIJADO = None            # tras la primera descarga, copiar aquí el SHA-256 del registro
GUARDAR_EN_DRIVE = False            # copia resultados a /content/drive/MyDrive/replica_segformer
VARIANTES = ['primaria', 'hf_512']  # primaria decide el veredicto; hf_512 es la desviación D1
RAIZ = '/content/replica'
"""),
        code("""
# 2. Entorno: GPU y librerías ya incluidas en Colab (no se instala nada).
import importlib, json, platform, subprocess
faltan = [m for m in ['torch', 'transformers', 'huggingface_hub', 'cv2', 'PIL', 'numpy']
          if importlib.util.find_spec(m) is None]
if faltan:
    raise SystemExit(f'Faltan {faltan}. Instalar a mano y anotar la versión en el informe.')
import torch, transformers, cv2, numpy, PIL
print('Python', platform.python_version(), '| torch', torch.__version__, '| transformers', transformers.__version__,
      '| cv2', cv2.__version__, '| numpy', numpy.__version__, '| Pillow', PIL.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA (correrá en CPU, mucho más lento)')
"""),
        code(f"""
# 3. Escribe el script y el protocolo versionados y verifica su SHA-256.
import hashlib, json, pathlib
ARCHIVOS = {{
    'scripts/replica_segformer_ade.py': ({json.dumps(script, ensure_ascii=False)}, '{sha(script)}'),
    'config/replica_segformer.json': ({json.dumps(protocolo, ensure_ascii=False)}, '{sha(protocolo)}'),
}}
for ruta, (texto, esperado) in ARCHIVOS.items():
    destino = pathlib.Path(RAIZ) / ruta
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(texto.encode('utf-8'))
    real = hashlib.sha256(destino.read_bytes()).hexdigest()
    assert real == esperado, f'SHA-256 distinto en {{ruta}}'
    print(ruta, real[:16])
import sys; sys.path.insert(0, f'{{RAIZ}}/scripts')
import replica_segformer_ade as R
PROTOCOLO = json.loads(pathlib.Path(RAIZ, 'config/replica_segformer.json').read_text(encoding='utf-8'))
"""),
        code(f"""
# 4. ADE20K desde la fuente oficial: tamaño, SHA-256, conteo y registro de licencia.
import os, time, urllib.request, zipfile, datetime
assert ACEPTO_TERMINOS_ADE20K, 'Leer los términos de ADE20K y poner ACEPTO_TERMINOS_ADE20K = True'
URL = '{ds['url']}'
TAMANO = {ds['tamano_bytes_observado']}
ZIP = '/content/ade/ADEChallengeData2016.zip'
os.makedirs('/content/ade', exist_ok=True)
if not os.path.exists(ZIP) or os.path.getsize(ZIP) != TAMANO:
    t0 = time.time()
    with urllib.request.urlopen(URL) as r, open(ZIP, 'wb') as f:
        cabeceras = dict(r.headers)
        while bloque := r.read(1 << 22):
            f.write(bloque)
    print(f'Descarga: {{time.time() - t0:.0f}} s')
else:
    cabeceras = {{}}
assert os.path.getsize(ZIP) == TAMANO, f'Tamaño distinto: {{os.path.getsize(ZIP)}}'
SHA_ADE = R.sha256(ZIP)
print('SHA-256 ADEChallengeData2016.zip:', SHA_ADE)
if SHA256_ADE_FIJADO:
    assert SHA_ADE == SHA256_ADE_FIJADO, 'El zip no coincide con el fijado en una corrida anterior'
t0 = time.time()
with zipfile.ZipFile(ZIP) as z:
    miembros = [m for m in z.namelist() if '/validation/' in m and not m.endswith('/')]
    z.extractall('/content/ade', members=miembros)
ADE_ROOT = '/content/ade/ADEChallengeData2016'
PARES = R.pares_validacion(ADE_ROOT)
assert len(PARES) == 2000, len(PARES)
print(f'Extracción de validación: {{time.time() - t0:.0f}} s, {{len(PARES)}} pares imagen/anotación')
REGISTRO_DATOS = {{'url': URL, 'tamano_bytes': TAMANO, 'sha256': SHA_ADE,
    'sha256_era_fijado': bool(SHA256_ADE_FIJADO), 'etag': cabeceras.get('ETag'),
    'last_modified': cabeceras.get('Last-Modified'), 'imagenes_validacion': len(PARES),
    'terminos': 'https://ade20k.csail.mit.edu/terms/', 'terminos_aceptados_por_quien_ejecuta': True,
    'licencia_kit': 'CSAILVision/sceneparsing BSD-3-Clause',
    'fecha_utc': datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z'}}
"""),
        code("""
# 5. Modelo con revisión y SHA-256 fijados; medición de 50 imágenes y extrapolación.
DISPOSITIVO = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELO_FN, INFO = R.cargar_modelo_hf(DISPOSITIVO)
print(INFO['modelo'], INFO['revision'][:12], 'parámetros:', f"{INFO['parametros']:,}")
prueba = R.evaluar(MODELO_FN, PARES, R.preprocesar_primario, DISPOSITIVO, limite=50, avisar=0)
por_imagen = prueba['segundos'] / 50
print(f'{por_imagen*1000:.0f} ms por imagen -> {por_imagen*2000/60:.1f} min por variante (2.000 imágenes)')
"""),
        code("""
# 6. Evaluación completa de cada variante. La primaria decide el veredicto.
import numpy as np
SALIDA = pathlib.Path('/content/replica_segformer'); SALIDA.mkdir(exist_ok=True)
RESULTADOS = {}
for variante in VARIANTES:
    areas = R.evaluar(MODELO_FN, PARES, R.PREPROCESOS[variante], DISPOSITIVO)
    np.savez_compressed(SALIDA / f'areas_{variante}.npz', nombres=np.array(areas['nombres']),
                        inter=areas['inter'], union=areas['union'], area_gt=areas['area_gt'])
    RESULTADOS[variante] = R.resumir(areas, PROTOCOLO, variante, INFO, INFO['id2label'])
    r = RESULTADOS[variante]
    print(f"{variante}: mIoU {r['mIoU_pct']:.2f} % IC95 [{r['IC95_pct'][0]:.2f}; {r['IC95_pct'][1]:.2f}] "
          f"| media por imagen (D2) {r['mIoU_medio_por_imagen_pct_D2']:.2f} % | {areas['segundos']/60:.1f} min")
"""),
        code("""
# 7. Resumen único: veredicto a priori y análisis de la desviación (D1, D2, D4, D5).
p = RESULTADOS['primaria']
resumen = {
    'item_checklist': 32, 'nivel': 'nivel_1_evaluacion (no replica el entrenamiento)',
    'sha256_protocolo': hashlib.sha256(pathlib.Path(RAIZ, 'config/replica_segformer.json').read_bytes()).hexdigest(),
    'sha256_script': hashlib.sha256(pathlib.Path(RAIZ, 'scripts/replica_segformer_ade.py').read_bytes()).hexdigest(),
    'dataset': REGISTRO_DATOS, 'modelo': p['modelo'], 'versiones_D4': p['versiones'],
    'reportado_articulo_pct': PROTOCOLO['paper_guia']['valor_reportado_miou'],
    'primaria': {k: v for k, v in p.items() if k != 'IoU_por_clase_pct'},
    'veredicto': p['criterio']['veredicto'],
    'D1_preprocesamiento_hf_512': ({'mIoU_pct': RESULTADOS['hf_512']['mIoU_pct'],
                                    'IC95_pct': RESULTADOS['hf_512']['IC95_pct'],
                                    'diferencia_con_primaria': RESULTADOS['hf_512']['mIoU_pct'] - p['mIoU_pct']}
                                   if 'hf_512' in RESULTADOS else 'no ejecutada'),
    'D2_media_por_imagen_pct': p['mIoU_medio_por_imagen_pct_D2'],
    'D3_pesos': 'No evaluada: sin el .pth oficial no se descarta una diferencia de conversión.',
    'D5_IoU_por_clase_pct': p['IoU_por_clase_pct'],
}
(SALIDA / 'resumen_replica.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
clases = sorted(((v, k) for k, v in p['IoU_por_clase_pct'].items() if v is not None))
print('Veredicto:', resumen['veredicto'], '| D =', round(p['criterio']['D_puntos'], 2), 'puntos, IC95',
      [round(x, 2) for x in p['criterio']['IC95_D']])
print('37,4 dentro del IC95:', p['criterio']['reportado_dentro_IC95'])
print('Peores 5 clases:', [(k, round(v, 1)) for v, k in clases[:5]])
"""),
        code("""
# 8. Guardar: Drive opcional y zip para descargar (subirlo al repo en resultados/replica_segformer/).
import shutil
if GUARDAR_EN_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    shutil.copytree(SALIDA, '/content/drive/MyDrive/replica_segformer', dirs_exist_ok=True)
shutil.make_archive('/content/replica_segformer', 'zip', SALIDA)
print('Listo: /content/replica_segformer.zip')
"""),
        md("""
## Cómo informar

- Citar el veredicto de `resumen_replica.json` tal como sale (`replicado`, `compatible` o `no_replicado`), con D y su IC95.
- Si no es `replicado`, presentar D1, D2, D4 y D5 como posibles causas; D3 queda como causa no descartada.
- No cambiar el δ, el preprocesamiento primario ni la métrica después de ver el número.
- Esto evalúa un checkpoint: no dice que el entrenamiento del artículo se haya reproducido.
"""),
    ]


def main():
    cuaderno = {'cells': celdas(), 'metadata': {
        'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}
    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(json.dumps(cuaderno, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(SALIDA, hashlib.sha256(SALIDA.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
