"""Genera colab/Atajos_Errores_E2.ipynb, autocontenido (checklist 13, 65 y 67).

Reutiliza las celdas de parámetros, paquete y entorno de crear_colab_complemento_gpu_e2.py (mismo ZIP cerrado, mismo
ZIP de entrenamiento, mismo entorno que el entrenamiento) y agrega, embebidos con su SHA-256, scripts/atajos_gpu_e2.py
y config/errores_atajos_e2_20261122.json. Salida en MyDrive/tesis_imo_e2_atajos_v1/. Reejecutar este generador tras
cambiar cualquiera de esos archivos.
  python scripts/crear_colab_atajos_e2.py
"""
import base64, hashlib, json
from pathlib import Path
import crear_colab_complemento_gpu_e2 as G
from empaquetar_complemento_gpu_e2 import COMPLEMENT

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'colab/Atajos_Errores_E2.ipynb'
PROPIOS = ['scripts/atajos_gpu_e2.py', 'config/errores_atajos_e2_20261122.json']

INTRO = '''# Atajos y taxonomía exacta de errores sobre los 50 checkpoints cerrados

**Exploratorio.** El test ya fue observado y aquí no se abre: todo es en **validación**. Nada cambia modelos, umbral
(0,5) ni checkpoints; no hay optimizador. Reglas fijadas antes en `config/errores_atajos_e2_20261122.json`.

Por corrida (10 redes × 5 semillas) y por imagen de validación (218):
- **C0 limpia**: Dice, TP/FP/FN, HD95 y componentes predichas (exceso distante, rótulos omitidos).
- **C1 sin contexto**: fuera de las cajas de los rótulos se pone la media de train.
- **C2 sin rótulo**: el rótulo se borra (relleno del entorno y suavizado): negativo sintético.
- **C3 gris**: se quita el color.

Para el top 3 guarda además la predicción limpia binaria, que sirve para los paneles de revisión humana.

Necesitas en Drive lo mismo que el complemento GPU: `tesis_imo_e2_arquitecturas_reales_v2/`,
`paquete_test_e2_cerrado.zip` y `paquete_colab_e2_v2.zip`. Entorno con **GPU**. Tiempo esperado en L4: unos 30 a 45 min
(4 inferencias por imagen; el complemento midió ~0,6 h para 9 por imagen de test); el piloto lo estima antes.

**Orden:** celdas 1 a 4 siempre (también tras reiniciar), piloto, corrida y descarga. Lo terminado se salta.
'''

PROPIOS_HEAD = '''# 4. Scripts propios de esta corrida (verifica SHA-256) y función de lanzamiento
PROPIOS = '''

PROPIOS_BODY = '''
for rel, (sha, b64) in PROPIOS.items():
    data = base64.b64decode(b64)
    assert hashlib.sha256(data).hexdigest() == sha, rel
    destino = PROJECT/rel
    if destino.exists() and sha_file(destino) != sha: raise RuntimeError(f'El paquete ya trae otra versión de {rel}')
    destino.parent.mkdir(parents=True, exist_ok=True); destino.write_bytes(data)
print('Scripts propios:', {k: v[0][:12] for k, v in PROPIOS.items()})
def atajos(*extra, output=OUTPUT):
    run([PY, '-u', SCRIPTS/'atajos_gpu_e2.py', '--runs', RUNS, '--output', output, *extra],
        log=LOGS/f'atajos_{stamp()}.log')
'''

PILOT = '''# 5. Piloto: semilla 11 de las 10 redes con 8 imágenes, en disco local. Estima el tiempo total (ESTIMADO).
if PILOT.exists(): shutil.rmtree(PILOT)
PILOT.mkdir(parents=True)
atajos('--seeds', '11', '--limit', '8', output=PILOT)
log = json.loads(sorted(PILOT.glob('tiempos/*.json'))[-1].read_text())
total = sum(5 * r['seconds'] / 8 * 218 for r in log['runs']) + log['verification_seconds']
estimado = {'ESTIMADO': 'extrapolado de 8 imágenes, semilla 11', 'gpu': log['device'],
            'por_red_min': {r['folder']: round(5 * r['seconds'] / 8 * 218 / 60, 1) for r in log['runs']},
            'total_horas': round(total / 3600, 2)}
(OUTPUT/'PILOTO_TIEMPOS_ATAJOS.json').write_text(json.dumps(estimado, indent=1, ensure_ascii=False))
print(json.dumps(estimado, indent=1, ensure_ascii=False))
'''

MAIN = '''# 6. Corrida completa: 50 corridas en validación. Para repartirla entre sesiones, pon una lista de redes.
MODELS = None
atajos(*(['--models', *MODELS] if MODELS else []))
hechos = sorted(OUTPUT.glob('*_seed*_atajos.json'))
print(len(hechos), 'de 50 corridas terminadas')
'''

EXPORT = '''# 7. Descargar JSON, máscaras del top 3 (NPZ, ~10-30 MB) y registros. Va a la carpeta que se indique en local.
EXPORT = Path('/content/atajos_errores_e2.zip')
with zipfile.ZipFile(EXPORT, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUTPUT.rglob('*')):
        if p.is_file() and p.suffix in {'.json', '.npz', '.log', '.txt'}: z.write(p, p.relative_to(OUTPUT))
print(EXPORT, round(EXPORT.stat().st_size / 2**20, 1), 'MiB')
from google.colab import files
files.download(str(EXPORT))
'''


def main():
    embebidos = {rel: [hashlib.sha256((PROJECT / rel).read_bytes()).hexdigest(), base64.b64encode((PROJECT / rel).read_bytes()).decode()]
                 for rel in COMPLEMENT}
    propios = {rel: [hashlib.sha256((PROJECT / rel).read_bytes()).hexdigest(), base64.b64encode((PROJECT / rel).read_bytes()).decode()]
               for rel in PROPIOS}
    params = G.PARAMS.replace("DRIVE/'tesis_imo_e2_diagnosticos_v1'", "DRIVE/'tesis_imo_e2_atajos_v1'") \
                     .replace("Path('/content/piloto_diagnosticos')", "Path('/content/piloto_atajos')")
    assert 'tesis_imo_e2_atajos_v1' in params and 'piloto_atajos' in params, 'Cambió crear_colab_complemento_gpu_e2.PARAMS'
    entorno = G.ENV.split('def stage(')[0]
    cells = []

    def add(kind, text):
        cell = {'cell_type': kind, 'id': f'atajos-{len(cells):02d}', 'metadata': {}, 'source': text.splitlines(keepends=True)}
        if kind == 'code':
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    add('markdown', INTRO); add('code', params)
    add('code', G.PACKAGE_HEAD + json.dumps(embebidos, indent=0) + '\n' + G.PACKAGE_BODY)
    add('code', entorno)
    add('code', PROPIOS_HEAD + json.dumps(propios, indent=0) + '\n' + PROPIOS_BODY)
    add('code', PILOT); add('code', MAIN); add('code', EXPORT)
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                                       'colab': {'name': OUT.name, 'gpuType': 'L4'}},
          'nbformat': 4, 'nbformat_minor': 5}
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    for cell in cells:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['id'], 'exec')
    print(json.dumps({'notebook': str(OUT.relative_to(PROJECT)), 'cells': len(cells),
                      'propios': {k: v[0][:16] for k, v in propios.items()}}, indent=1))


if __name__ == '__main__':
    main()
