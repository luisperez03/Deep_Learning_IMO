"""Genera colab/Complemento_Entrenamiento_E2.ipynb. Solo escribe el cuaderno y su verificación.

El cuaderno usa el mismo paquete v2 del benchmark (paquete_colab_e2_v2.zip, mismo SHA-256) y
el mismo entorno (Python 3.11 aislado, torch 2.5.1+cu121, requirements-e2.txt, Mamba). Encima
escribe tres archivos nuevos que lleva incrustados y verificados por SHA-256: el runner, el
resumidor y la configuración del complemento. No sube nada ni ejecuta celdas.
"""
import ast
import base64
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT/'colab/Complemento_Entrenamiento_E2.ipynb'
ZIP_V2_SHA256 = '954f4fe5663d9fedd676dcee4f1f7fb9086a2a7f5c1907ce902b5df4c109cb49'
INCRUSTADOS = ['scripts/complemento_entreno_e2.py', 'scripts/resumir_complemento_entreno_e2.py',
               'config/complemento_entreno_e2.json']
# Scripts del paquete v2 de los que depende el runner; sus SHA-256 son los registrados en
# config/protocolo_test_e2_cerrado.json (training_source). Si el ZIP trae otros, se detiene.
DEPENDENCIAS = ['scripts/experimento_e2_colab.py', 'scripts/modelos_e2.py', 'scripts/dataset_ready.py']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hashes_protocolo():
    texto = (PROJECT/'config/protocolo_test_e2_cerrado.json').read_text(encoding='utf-8')
    protocolo = json.loads(texto)
    encontrados = {}

    def buscar(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in DEPENDENCIAS and isinstance(v, str):
                    encontrados[k] = v
                buscar(v)
        elif isinstance(o, list):
            for v in o:
                buscar(v)
    buscar(protocolo)
    for dep in DEPENDENCIAS:
        actual = sha((PROJECT/dep).read_bytes())
        if encontrados.get(dep) != actual:
            raise RuntimeError(f'{dep} no coincide con el protocolo cerrado; el runner no sería el mismo')
    return {d: encontrados[d] for d in DEPENDENCIAS}


def crear():
    deps = hashes_protocolo()
    archivos = {}
    for rel in INCRUSTADOS:
        data = (PROJECT/rel).read_bytes()
        if rel.endswith('.py'):
            ast.parse(data.decode('utf-8'))
        archivos[rel] = {'sha256': sha(data), 'b64': base64.b64encode(data).decode('ascii')}
    cells = []

    def md(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)})

    def code(text):
        text = text.strip('\n')
        ast.parse(text)
        cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                      'source': text.splitlines(True)})

    md('''
# Complemento de entrenamiento E2 — ablación, sanidad y búsqueda de hiperparámetros

Lo que el benchmark no cubría y exige volver a entrenar:

| bloque | qué hace | control de la guía |
|---|---|---|
| sobreajuste | las 10 redes sobreajustan un lote fijo de 4 imágenes | 11.1, prueba de sanidad (checklist 41) |
| ablación | TransUNet sin Transformer, sin conexiones de salto y sin término Dice, 5 semillas | 4.6 (checklist 61) |
| permutación | máscaras de train barajadas entre imágenes; pipeline completo | Tabla A (checklist 41 y 56) |
| hpo | búsqueda aleatoria con la misma secuencia de intentos para las 10 | 11.2 (checklist 37) |

**Todo es exploratorio**: el test del benchmark ya fue observado. Aquí no se carga el test; todo se
elige y se mide en validación. El cuaderno fue generado por `scripts/crear_colab_complemento_entreno.py`
y **no ha sido ejecutado en Colab** por quien lo preparó: los tiempos son estimados hasta que corra el piloto.

Orden: celdas 1 a 6 siempre; después el piloto; después los bloques en el orden de prioridad
(sobreajuste, ablación, permutación FCN, HPO, permutación TransUNet). Cada corrida se reanuda sola
desde su última época guardada en Drive: si la sesión se corta, vuelve a ejecutar las celdas 1 a 6 y
la del bloque.
''')
    code('''
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, os, json, hashlib, zipfile, base64, datetime
subprocess.run(['nvidia-smi'], check=True)
subprocess.run(['mountpoint', '-q', '/content/drive'], check=True)
OUTPUT = Path('/content/drive/MyDrive/tesis_imo_e2_complemento_entreno_v1')
BASE_RUNS = Path('/content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2')  # benchmark original
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT/'logs').mkdir(exist_ok=True)
print('Salida:', OUTPUT, '| benchmark original presente:', BASE_RUNS.exists())
''')
    code(f'''
# Mismo paquete que el benchmark v2: datos train/val, código y fuentes de terceros.
EXPECTED_ZIP_SHA256 = {ZIP_V2_SHA256!r}
ZIP = next((p for p in [Path('/content/paquete_colab_e2_v2.zip'), Path('/content/drive/MyDrive/paquete_colab_e2_v2.zip')] if p.exists()), None)
if ZIP is None: raise FileNotFoundError('Sube paquete_colab_e2_v2.zip al panel de archivos o a la raíz de Drive.')
def sha_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
assert sha_file(ZIP) == EXPECTED_ZIP_SHA256, 'ZIP distinto del v2 del benchmark'
BASE = Path('/content/imo_e2_v2'); BASE.mkdir(exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    for n in z.namelist():
        if not (BASE/n).resolve().is_relative_to(BASE): raise ValueError('Ruta no válida en ZIP')
    z.extractall(BASE)
manifest = json.loads((BASE/'PACKAGE_MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert sha_file(BASE/name) == expected, name
PROJECT = BASE/'tesis_imo_para_codex'
DEPENDENCIAS = {json.dumps(deps)}
for rel, esperado in DEPENDENCIAS.items():
    assert sha_file(PROJECT/rel) == esperado, f'{{rel}} no es el del protocolo cerrado'
print('Paquete íntegro:', len(manifest['files']), 'archivos; dependencias del runner verificadas')
''')
    code(f'''
# Archivos nuevos del complemento, incrustados y verificados (generados desde el repositorio).
INCRUSTADOS = {json.dumps(archivos, indent=0)}
for rel, item in INCRUSTADOS.items():
    data = base64.b64decode(item['b64'])
    assert hashlib.sha256(data).hexdigest() == item['sha256'], rel
    destino = PROJECT/rel
    destino.write_bytes(data)
    print('escrito', rel, item['sha256'][:12])
''')
    md('''
## Entorno aislado, igual al del benchmark
Python 3.11 fuera del kernel, torch 2.5.1+cu121, `requirements-e2.txt`, transformers 4.46.3 y las
ruedas de causal-conv1d y mamba-ssm, como en `Benchmark_E2_Colab_v2.ipynb`. Si `/content` se borró
(sesión nueva), esta celda lo reconstruye. Al final compara el `pip freeze` con el del benchmark.
''')
    code('''
subprocess.run([sys.executable,'-m','pip','install','uv==0.6.9'],check=True)
ENV=Path('/content/imo-e2-env')
if not (ENV/'bin/python').exists():
    subprocess.run([sys.executable,'-m','uv','venv','--python','3.11',str(ENV)],check=True)
PY=str(ENV/'bin/python')
def pip_install(*items):
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',PY,*items],check=True)
pip_install('setuptools==75.8.0')
pip_install('torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu121')
pip_install('-r',str(PROJECT/'requirements-e2.txt'))
pip_install('transformers==4.46.3')
pip_install('--no-deps','https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl')
pip_install('--no-deps','https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl')
# Como en Complemento_GPU_E2: las extensiones de Mamba cargan libc10/libcudart del venv, no las del sistema.
site=ENV/'lib/python3.11/site-packages'
dirs=[site/'torch/lib',*sorted((site/'nvidia').glob('*/lib'))]
paths=[str(p) for p in dirs if p.is_dir()]+os.environ.get('LD_LIBRARY_PATH','').split(':')
os.environ['LD_LIBRARY_PATH']=':'.join(dict.fromkeys(p for p in paths if p))
subprocess.run([PY,'-c','import torch, torchvision, mamba_ssm; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0))'],check=True)
freeze=subprocess.check_output([sys.executable,'-m','uv','pip','freeze','--python',PY],text=True)
(OUTPUT/'environment.freeze.txt').write_text(freeze)
original = BASE_RUNS/'environment.freeze.txt'
if original.exists():
    a, b = set(original.read_text().splitlines()), set(freeze.splitlines())
    print('Paquetes distintos del benchmark:', sorted(a ^ b) or 'ninguno')
else:
    print('No se encontró el freeze del benchmark en', original)
''')
    md('''
## Comprobación previa (la misma del benchmark) y lanzador con registro
`preflight_e2.py` prueba forward, gradientes y un paso de optimizador de las 10 redes con datos
sintéticos y deja `preflight.json` en la carpeta del complemento. El lanzador escribe cada salida
en `logs/` dentro de Drive además de mostrarla aquí.
''')
    code('''
MODELS=['fcn','segnet','unet','deeplabv3plus','unetplusplus','pidnet_s','segnext_t','segformer_b0','transunet_2d','vmunet']
subprocess.run([PY,str(PROJECT/'scripts/preflight_e2.py'),'--output',str(OUTPUT),'--models',*MODELS],check=True)
report=json.loads((OUTPUT/'preflight.json').read_text())
for name,result in report['models'].items(): print(name,result['status'])
assert all(r['status']=='passed' for r in report['models'].values()), 'No lanzar bloques con un preflight fallido'

RUNNER = str(PROJECT/'scripts/complemento_entreno_e2.py')
def correr(*argumentos):
    sello = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    log = OUTPUT/'logs'/f'{argumentos[0]}_{sello}.log'
    comando = ' '.join([PY, '-u', RUNNER, *map(str, argumentos), '--output', str(OUTPUT)])
    print('$', comando, '\\n  log:', log)
    subprocess.run(['bash', '-o', 'pipefail', '-c', f'cd {PROJECT}/scripts && {comando} 2>&1 | tee -a {log}'], check=True)
''')
    md('''
## Piloto: medir antes de lanzar
Corre la primera época de tres corridas reales de la ablación (sin Dice, que cuesta lo mismo que
TransUNet completo; sin Transformer; sin saltos) y dos épocas de la primera permutación de FCN.
Esas épocas **no se pierden**: cada bloque reanuda desde ellas. Luego escribe `estimacion_computo.json`
con horas de L4 por bloque, marcadas como estimadas. Con esa tabla se fijan `B` y `T` **antes**
de ver ningún resultado de permutación o de HPO. Tarda unos 6 minutos estimados.
''')
    code('''
correr('piloto')
print(json.dumps(json.loads((OUTPUT/'estimacion_computo.json').read_text()), indent=2, ensure_ascii=False))
''')
    md('''
## Bloque 1 — sobreajuste a un lote (10 redes, unos minutos estimados)
Lote fijo de 4 imágenes de train, 500 pasos con la receta de cada red. Pasa si el Dice de esas 4 en
modo evaluación supera 0,90. Si una red no pasa, se reporta tal cual: no se cambia su receta.
''')
    code('''
EJECUTAR_SOBREAJUSTE = False
if EJECUTAR_SOBREAJUSTE:
    correr('sobreajuste')
''')
    md('''
## Bloque 2 — ablación de TransUNet (15 corridas, unas 9 horas de L4 estimadas)
Tres variantes × semillas 11, 23, 37, 53, 71, con la receta del benchmark. `CONTROL=True` añade
TransUNet completo (5 corridas más, unas 3,5 h) como referencia en este mismo entorno; sin él, la
referencia son las corridas originales. Para repartir entre dos sesiones de Colab a la vez, usa
`FRAGMENTO='0/2'` en una y `'1/2'` en la otra (misma carpeta de Drive, tareas disjuntas).
''')
    code('''
EJECUTAR_ABLACION = False
CONTROL = False
FRAGMENTO = None   # p. ej. '0/2'
if EJECUTAR_ABLACION:
    variantes = ['sin_transformer','sin_saltos','sin_dice'] + (['completo'] if CONTROL else [])
    correr('ablacion', '--variantes', *variantes, *(['--fragmento', FRAGMENTO] if FRAGMENTO else []))
''')
    md('''
## Bloque 3 — permutación del pipeline con FCN (B = 200)
Mínimo de la guía para modelos costosos. Es el control del pipeline común a las 10 redes. Fija `B`
con la estimación del piloto y no lo cambies después de ver resultados. Se puede repartir con `FRAGMENTO`.
''')
    code('''
EJECUTAR_PERMUTACION_FCN = False
B_FCN = 200
FRAGMENTO = None
if EJECUTAR_PERMUTACION_FCN:
    correr('permutacion', '--permutacion', 'pipeline_fcn', '--B', B_FCN, *(['--fragmento', FRAGMENTO] if FRAGMENTO else []))
''')
    md('''
## Bloque 4 — búsqueda de hiperparámetros con igual presupuesto (T intentos × 10 redes)
Intento 0 = receta; intentos 1..T-1 = multiplicador de tasa 10^u y weight decay comunes a las 10.
Presupuesto por intento: semilla 11, hasta 10 épocas, paciencia 3. Las tareas van por intento y luego
por red, así que cualquier corte deja un prefijo completo igual para todas. Recomendado `T=4`.
''')
    code('''
EJECUTAR_HPO = False
T = 4
FRAGMENTO = None
if EJECUTAR_HPO:
    correr('hpo', '--T', T, *(['--fragmento', FRAGMENTO] if FRAGMENTO else []))
''')
    md('''
## Bloque 5 — permutación específica de TransUNet (B = 19)
Con B = 19 el p mínimo es 0,05: por debajo del mínimo de la guía (B ≥ 200), se declara así.
''')
    code('''
EJECUTAR_PERMUTACION_TRANSUNET = False
B_TRANSUNET = 19
FRAGMENTO = None
if EJECUTAR_PERMUTACION_TRANSUNET:
    correr('permutacion', '--permutacion', 'transunet', '--B', B_TRANSUNET, *(['--fragmento', FRAGMENTO] if FRAGMENTO else []))
''')
    md('''
## Resumen
Lee solo los JSON de la carpeta del complemento y, para la referencia de la ablación y el s_obs de la
permutación, los `result.json` y mejores checkpoints de validación del benchmark original. Escribe
`RESUMEN.json` y `RESUMEN.md` en Drive. Descarga esos dos archivos, `estimacion_computo.json`,
`environment.freeze.txt`, `preflight.json`, la carpeta `logs/` y los `result.json`/`history.json` de
cada corrida; los checkpoints de la ablación se quedan en Drive.
''')
    code('''
subprocess.run([PY, str(PROJECT/'scripts/resumir_complemento_entreno_e2.py'), '--output', str(OUTPUT),
                '--base-runs', str(BASE_RUNS)], check=True)
abiertas = sorted(p.parent.name for p in OUTPUT.glob('*/last.pt'))
print('Corridas empezadas sin cerrar:', abiertas or 'ninguna')
''')
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                                       'colab': {'name': 'Complemento_Entrenamiento_E2.ipynb', 'gpuType': 'L4'}},
          'nbformat': 4, 'nbformat_minor': 5}
    for i, c in enumerate(cells):
        c['id'] = f'e2-entreno-{i:02d}'
    raw = json.dumps(nb, ensure_ascii=False, indent=1)
    NOTEBOOK.write_text(raw, encoding='utf-8')
    verificacion = {'notebook_sha256': sha(raw.encode('utf-8')), 'zip_v2_sha256': ZIP_V2_SHA256,
                    'dependencias_protocolo': deps,
                    'incrustados': {k: v['sha256'] for k, v in archivos.items()},
                    'celdas_codigo_sintaxis_valida': True, 'ejecutado_en_colab': False}
    (NOTEBOOK.parent/'Complemento_Entrenamiento_E2.verificacion.json').write_text(
        json.dumps(verificacion, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(verificacion, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    crear()
