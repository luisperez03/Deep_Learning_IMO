"""Genera colab/Busqueda_Arquitectura_E2.ipynb. Solo escribe el cuaderno y su verificación.

Mismo paquete v2 del benchmark (paquete_colab_e2_v2.zip, mismo SHA-256) y mismo torch (2.5.1+cu121 en
un Python 3.11 aislado, requirements-e2.txt). Las compactas no usan Mamba ni transformers: esas ruedas
no se instalan. Encima del paquete escribe, verificados por SHA-256, los archivos nuevos de la
búsqueda que lleva incrustados. No sube nada ni ejecuta celdas.
"""
import ast
import base64
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT/'colab/Busqueda_Arquitectura_E2.ipynb'
ZIP_V2_SHA256 = '954f4fe5663d9fedd676dcee4f1f7fb9086a2a7f5c1907ce902b5df4c109cb49'
INCRUSTADOS = ['scripts/busqueda_arquitectura_e2.py', 'scripts/resumir_busqueda_arquitectura_e2.py',
               'scripts/complemento_entreno_e2.py', 'config/busqueda_arquitectura_e2.json',
               'config/arquitecturas_compactas_e2.csv']
# Del paquete v2; sus SHA-256 son los de config/protocolo_test_e2_cerrado.json. Si el ZIP trae otros, se detiene.
DEPENDENCIAS = ['scripts/experimento_e2_colab.py', 'scripts/modelos_e2.py', 'scripts/dataset_ready.py',
                'scripts/modelos_base.py']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hashes_protocolo():
    protocolo = json.loads((PROJECT/'config/protocolo_test_e2_cerrado.json').read_text(encoding='utf-8'))
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
        if encontrados.get(dep) != sha((PROJECT/dep).read_bytes()):
            raise RuntimeError(f'{dep} no coincide con el protocolo cerrado')
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
# Búsqueda de arquitectura de las redes compactas (sección 14 de la guía)

FCN, U-Net y SegNet compactas son diseño propio, así que recorren el espacio obligatorio de la guía:
3 bloques, anchos {16, 32, 64, 128, 256} en todas las combinaciones y dropout 0,2 en todas las
posiciones, **1000 arquitecturas por familia**. Protocolo fijado antes de correr en
`docs/PROTOCOLO_BUSQUEDA_ARQUITECTURA_E2.md` y `config/busqueda_arquitectura_e2.json`.

| paso | qué hace | orden de magnitud |
|---|---|---|
| piloto | 1 época de 15 configuraciones a 256 y a 128 px para medir tiempo | 15-30 min |
| fidelidad | elige la fidelidad de la fase 1 con la regla del protocolo y el presupuesto de horas | segundos |
| ajuste previo | tasa de Adam con la referencia (64,64,64)/(0,1,0), 3 tasas × 3 familias | minutos |
| fase 1 | 1000 × 3 familias, semilla 97, 6 épocas, a la fidelidad elegida; + 40 × 3 con la semilla 98 | la mayor parte |
| finalistas | ruido por semilla, k y finalistas (solo validación) | segundos |
| fase 2 | k finalistas + referencia + heredada, semillas 11, 23, 37, 53, 71, hasta 30 épocas | horas |
| resumen | `RESUMEN.json`, `RESUMEN.md`, `fase1.csv` | segundos |

**Todo es exploratorio y solo de validación**: el test del benchmark ya fue observado y aquí no se carga.
El cuaderno fue generado por `scripts/crear_colab_busqueda_arquitectura_e2.py` y probado en CPU con pocas
imágenes; **no se ha ejecutado en Colab** por quien lo preparó.

Cada corrida se reanuda sola: la fase 1 por corrida terminada (`result.json`), la fase 2 por época
(`last.pt`). Si la sesión se corta, vuelve a ejecutar las celdas 1 a 5 y la del paso en curso.
''')
    code('''
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, os, json, hashlib, zipfile, base64, datetime, time
subprocess.run(['nvidia-smi'], check=True)
subprocess.run(['mountpoint', '-q', '/content/drive'], check=True)
OUTPUT = Path('/content/drive/MyDrive/tesis_imo_e2_busqueda_arquitectura_v1')
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT/'logs').mkdir(exist_ok=True)
print('Salida:', OUTPUT, '| CPU:', os.cpu_count())
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
print('Paquete íntegro:', len(manifest['files']), 'archivos; dependencias verificadas')
''')
    code(f'''
# Archivos nuevos de la búsqueda, incrustados y verificados (generados desde el repositorio).
INCRUSTADOS = {json.dumps(archivos, indent=0)}
for rel, item in INCRUSTADOS.items():
    data = base64.b64decode(item['b64'])
    assert hashlib.sha256(data).hexdigest() == item['sha256'], rel
    (PROJECT/rel).parent.mkdir(parents=True, exist_ok=True)
    (PROJECT/rel).write_bytes(data)
    print('escrito', rel, item['sha256'][:12])
''')
    md('''
## Entorno aislado, el mismo torch del benchmark
Python 3.11 fuera del kernel, torch 2.5.1+cu121 y `requirements-e2.txt`. Si `/content` se borró
(sesión nueva), esta celda lo reconstruye. Al final guarda el `pip freeze` en Drive.
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
subprocess.run([PY,'-c','import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0))'],check=True)
freeze=subprocess.check_output([sys.executable,'-m','uv','pip','freeze','--python',PY],text=True)
(OUTPUT/'environment.freeze.txt').write_text(freeze)
''')
    md('''
## Lanzador
`correr` ejecuta un paso en un proceso. `correr_paralelo` lanza `n` procesos a la vez en la misma GPU,
cada uno con un fragmento disjunto de las tareas (`--fragmento i/n`); las redes compactas no llenan una
L4, así que varios procesos aprovechan el tiempo muerto sin cambiar nada del protocolo (cada corrida es
la misma que en serie). Cada proceso deja su log en `logs/` dentro de Drive. `progreso()` cuenta las
corridas terminadas.
''')
    code('''
RUNNER = str(PROJECT/'scripts/busqueda_arquitectura_e2.py')
def _comando(bloque, *argumentos):
    return [PY, '-u', RUNNER, bloque, *map(str, argumentos), '--output', str(OUTPUT)]
def correr(bloque, *argumentos):
    sello = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    log = OUTPUT/'logs'/f'{bloque}_{sello}.log'
    comando = ' '.join(_comando(bloque, *argumentos))
    print('$', comando, '\\n  log:', log)
    subprocess.run(['bash','-o','pipefail','-c',f'cd {PROJECT}/scripts && {comando} 2>&1 | tee -a {log}'],check=True)
def correr_paralelo(bloque, n, *argumentos):
    sello = datetime.datetime.now().strftime('%Y%m%dT%H%M%S'); procesos = []
    for i in range(n):
        log = OUTPUT/'logs'/f'{bloque}_{sello}_p{i}de{n}.log'
        f = open(log, 'a')
        procesos.append((subprocess.Popen(_comando(bloque, *argumentos, '--fragmento', f'{i}/{n}'),
                                          cwd=str(PROJECT/'scripts'), stdout=f, stderr=subprocess.STDOUT), f, log))
    print(f'{n} procesos de {bloque}; logs en', OUTPUT/'logs')
    try:
        while any(p.poll() is None for p, _, _ in procesos):
            time.sleep(120); progreso()
    finally:
        for p, f, _ in procesos:
            if p.poll() is None: p.terminate()
            f.close()
    fallidos = [str(log) for p, _, log in procesos if p.returncode != 0]
    if fallidos: raise RuntimeError(f'Procesos con error (ver logs): {fallidos}')
def progreso():
    t = {}
    for b in ('piloto','ajuste','fase1','fase2'):
        for r in (OUTPUT/b).glob('*/result.json'):
            fam = r.parent.name.split('_')[0]; t[(b, fam)] = t.get((b, fam), 0) + 1
    print(datetime.datetime.now().strftime('%H:%M'), {f'{b}:{f}': n for (b, f), n in sorted(t.items())}, flush=True)
''')
    md('''
## 1. Presupuesto, piloto y fidelidad de la fase 1
**Antes de correr el piloto**, escribe en `HORAS_FASE1` las horas de L4 que se aceptan para ajuste previo +
fase 1 + repetición (las acordadas al aprobar la búsqueda). El piloto corre una época de 5 anchos × 3
familias a 256 y a 128 px con la caché llena (unos 15-30 minutos estimados) y `estimar` deja
`estimacion_computo.json`. `fidelidad` aplica la regla fijada en el protocolo: la primera opción
(A: 256 px; B: 128 px; C: 128 px y mitad de train) cuya estimación cabe en `HORAS_FASE1`. Queda en
`fidelidad.json` y no se puede cambiar después de empezar el ajuste.
''')
    code('''
HORAS_FASE1 = None   # p. ej. 40; se fija ANTES de ver el piloto
assert HORAS_FASE1 is not None, 'Fija el presupuesto de horas antes del piloto'
correr('piloto')
correr('estimar')
correr('fidelidad', '--horas-fase1', HORAS_FASE1)
print(json.dumps(json.loads((OUTPUT/'fidelidad.json').read_text())['elegida'], indent=1, ensure_ascii=False))
''')
    md('''
## 2. Ajuste previo de la tasa (9 corridas cortas)
Referencia (64,64,64)/(0,1,0), tasas 0,0003, 0,001 y 0,003, semilla 97, 6 épocas, a la fidelidad de la fase 1. Regla fijada antes:
se queda la de mayor Dice de validación salvo que gane a 0,001 por menos de 0,005.
''')
    code('''
correr_paralelo('ajuste', 3)
correr('cerrar_ajuste')
''')
    md('''
## 3. Fase 1: las 1000 arquitecturas de cada familia
Semilla 97, 6 épocas, misma receta para todas. Las 40 primeras del orden se repiten con la semilla 98 para
estimar el ruido. El orden es una permutación fija: si hay que parar, lo hecho es una muestra aleatoria
del espacio, igual para las tres familias. `PROCESOS` es cuántos procesos comparten la GPU; mira
`nvidia-smi` en el primer progreso y súbelo si la utilización está baja y sobra memoria.
''')
    code('''
EJECUTAR_FASE1 = False
PROCESOS = 4
if EJECUTAR_FASE1:
    correr_paralelo('fase1', PROCESOS)
progreso()
''')
    md('''
## 4. Finalistas
Calcula σ por semilla (repetición) y σ entre arquitecturas; k = 10 si σ_semilla/σ_arquitectura ≤ 1 en las
tres familias, k = 20 si no. Añade la referencia y la heredada. Solo se corre con la fase 1 completa;
con `PERMITIR_INCOMPLETA=True` se elige del prefijo hecho y queda registrado como desviación.
''')
    code('''
PERMITIR_INCOMPLETA = False
correr('finalistas', *(['--permitir-incompleta'] if PERMITIR_INCOMPLETA else []))
print(json.dumps({f: v['fase2'] for f, v in json.loads((OUTPUT/'finalistas.json').read_text())['familias'].items()}, indent=1))
''')
    md('''
## 5. Fase 2: finalistas con 5 semillas y presupuesto completo
Semillas 11, 23, 37, 53, 71, hasta 30 épocas con paciencia 5; guarda el mejor checkpoint y mide la brecha
train-validación. Las tareas van por semilla: si se corta, todos los finalistas tienen las mismas semillas.
''')
    code('''
EJECUTAR_FASE2 = False
PROCESOS = 4
if EJECUTAR_FASE2:
    correr_paralelo('fase2', PROCESOS)
progreso()
''')
    md('''
## 6. Resumen
Escribe `RESUMEN.json`, `RESUMEN.md` y `fase1.csv` en la carpeta de Drive. Descarga la carpeta completa
**sin** los `best.pt` (se quedan en Drive) para el análisis local en `resultados/busqueda_arquitectura_e2/`.
''')
    code('''
subprocess.run([PY, str(PROJECT/'scripts/resumir_busqueda_arquitectura_e2.py'), '--output', str(OUTPUT)], check=True)
abiertas = sorted(p.parent.name for p in OUTPUT.glob('fase2/*/last.pt'))
print('Corridas de la fase 2 empezadas sin cerrar:', abiertas or 'ninguna')
# Paquete liviano para descargar (sin checkpoints):
import shutil
liviano = Path('/content/busqueda_arquitectura_e2_resultados.zip')
with zipfile.ZipFile(liviano, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in OUTPUT.rglob('*'):
        if p.is_file() and p.suffix != '.pt':
            z.write(p, p.relative_to(OUTPUT))
shutil.copy(liviano, OUTPUT/liviano.name)
print('Descarga', OUTPUT/liviano.name, round(liviano.stat().st_size/1e6, 1), 'MB')
''')
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                                       'colab': {'name': 'Busqueda_Arquitectura_E2.ipynb', 'gpuType': 'L4'}},
          'nbformat': 4, 'nbformat_minor': 5}
    for i, c in enumerate(cells):
        c['id'] = f'e2-busqueda-{i:02d}'
    raw = json.dumps(nb, ensure_ascii=False, indent=1)
    NOTEBOOK.write_text(raw, encoding='utf-8')
    verificacion = {'notebook_sha256': sha(raw.encode('utf-8')), 'zip_v2_sha256': ZIP_V2_SHA256,
                    'dependencias_protocolo': deps,
                    'incrustados': {k: v['sha256'] for k, v in archivos.items()},
                    'celdas_codigo_sintaxis_valida': True, 'ejecutado_en_colab': False}
    (NOTEBOOK.parent/'Busqueda_Arquitectura_E2.verificacion.json').write_text(
        json.dumps(verificacion, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(verificacion, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    crear()
