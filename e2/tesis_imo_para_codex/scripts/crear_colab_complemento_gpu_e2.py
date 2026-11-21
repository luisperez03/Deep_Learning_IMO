"""Genera colab/Complemento_GPU_E2.ipynb, autocontenido: lleva embebidos (base64 + SHA256) los archivos del
complemento listados en empaquetar_complemento_gpu_e2.COMPLEMENT, y en Colab arma el paquete desde el ZIP de test
cerrado y el ZIP de entrenamiento que están en Drive. Reejecutar este generador tras cambiar cualquiera de esos
archivos; el cuaderno imprime los hashes que lleva.
"""
import base64,hashlib,json
from pathlib import Path
from empaquetar_complemento_gpu_e2 import COMPLEMENT,CLOSED_ZIP_SHA256,TRAINVAL_ZIP_SHA256

PROJECT=Path(__file__).resolve().parents[1]
OUT=PROJECT/'colab/Complemento_GPU_E2.ipynb'

INTRO='''# Complemento GPU del E2: diagnósticos sobre los 50 checkpoints cerrados

**Exploratorio.** El test ya fue observado (E1 y evaluación cerrada E2). Nada de lo que salga aquí cambia modelos,
umbral (0,5) ni checkpoints; no hay optimizador ni entrenamiento. Todo se calibra en validación.

Qué produce, en `MyDrive/tesis_imo_e2_diagnosticos_v1/`:
- `clean`: train/val/test limpios con Dice, IoU, precisión, recall, HD95, ASSD, Brier, NLL, ECE fijo y adaptativo
  (15 bins), temperatura ajustada en validación y métricas recalibradas en test; latencia, memoria y FLOPs parciales;
  brecha train–val con el Dice de train en modo evaluación. 50 corridas.
- `robustness`: ruido gaussiano, desenfoque y brillo, 3 severidades cada uno, sobre test. 450 tareas.
- `interpret`: Integrated Gradients en 100 casos de validación para el top 3 por validación (TransUNet, U-Net, FCN),
  5 semillas; fidelidad (borrado/inserción contra orden aleatorio) y estabilidad entre semillas.
- Resumen: intervalos por bootstrap de grupos, curvas de robustez, ensembles de 5 semillas del top 3 con conjuntos de
  etiquetas y **predicción conforme por píxel con un umbral por clase (Mondrian) calibrada en validación**.

Necesitas en Drive: `tesis_imo_e2_arquitecturas_reales_v2/` (50 carpetas), `paquete_test_e2_cerrado.zip` y
`paquete_colab_e2_v2.zip` (o súbelos al panel de archivos). Entorno de ejecución con **GPU**, de preferencia la misma
en todas las sesiones: la carpeta de salida queda atada a la GPU de la primera corrida.

**Orden:** celdas 1 a 3 siempre (también tras reiniciar: `/content` se borra), luego el piloto, y después las etapas.
Cada etapa se puede volver a lanzar: lo terminado se verifica por firma y hash y se salta.
'''

PARAMS='''# 1. Parámetros, Drive y utilidades
from google.colab import drive
drive.mount('/content/drive')
import os, sys, subprocess, json, hashlib, zipfile, base64, time, shutil
from pathlib import Path
DRIVE = Path('/content/drive/MyDrive')
RUNS = DRIVE/'tesis_imo_e2_arquitecturas_reales_v2'
OUTPUT = DRIVE/'tesis_imo_e2_diagnosticos_v1'      # resultados definitivos (reanudables)
PILOT = Path('/content/piloto_diagnosticos')       # piloto de tiempos: disco local, se descarta
LOGS = OUTPUT/'logs'
def first(*paths):
    return next((Path(p) for p in paths if Path(p).exists()), None)
CLOSED_ZIP = first('/content/paquete_test_e2_cerrado.zip', DRIVE/'paquete_test_e2_cerrado.zip')
TRAINVAL_ZIP = first('/content/paquete_colab_e2_v2.zip', DRIVE/'paquete_colab_e2_v2.zip')
assert RUNS.is_dir(), f'Falta {RUNS}'
assert CLOSED_ZIP, 'Falta paquete_test_e2_cerrado.zip (raíz de Drive o panel de archivos)'
assert TRAINVAL_ZIP, 'Falta paquete_colab_e2_v2.zip (raíz de Drive o panel de archivos)'
LOGS.mkdir(parents=True, exist_ok=True)

def run(command, log=None, check=True):
    """Ejecuta mostrando la salida en vivo y, si se pide, copiándola a un log en Drive."""
    command = [str(c) for c in command]
    stream = open(log, 'a', encoding='utf-8') if log else None
    if stream: stream.write('\\n$ ' + ' '.join(command) + '\\n')
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=os.environ.copy()) as p:
        for line in p.stdout:
            print(line, end='', flush=True)
            if stream: stream.write(line); stream.flush()
        code = p.wait()
    if stream: stream.close()
    if check and code: raise RuntimeError(f'Comando fallido ({code}); revisa el error anterior: {command}')
    return code
def stamp(): return time.strftime('%Y%m%dT%H%M%S')
run(['nvidia-smi'])
'''

PACKAGE_HEAD='''# 2. Escribir los archivos del complemento, armar el paquete y extraerlo (verifica todos los hashes)
EMBEDDED = '''

PACKAGE_BODY='''
SOURCE = Path('/content/complemento_fuente/tesis_imo_para_codex')
for rel, (sha, b64) in EMBEDDED.items():
    data = base64.b64decode(b64)
    assert hashlib.sha256(data).hexdigest() == sha, rel
    (SOURCE/rel).parent.mkdir(parents=True, exist_ok=True); (SOURCE/rel).write_bytes(data)
print('Archivos del complemento:', {k: v[0][:12] for k, v in EMBEDDED.items()})
PACKAGE = Path('/content/paquete_complemento_gpu_e2.zip')
if not Path(str(PACKAGE) + '.sha256').exists():
    run([sys.executable, '-u', SOURCE/'scripts/empaquetar_complemento_gpu_e2.py', '--closed-zip', CLOSED_ZIP,
         '--trainval-zip', TRAINVAL_ZIP, '--output', PACKAGE], log=LOGS/f'paquete_{stamp()}.log')
def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
package_sha = sha_file(PACKAGE)
assert package_sha == Path(str(PACKAGE) + '.sha256').read_text().split()[0], 'Paquete distinto de su .sha256'
BASE = Path('/content/imo_e2_complemento')
if BASE.exists(): shutil.rmtree(BASE)
with zipfile.ZipFile(PACKAGE) as z:
    for name in z.namelist(): assert (BASE/name).resolve().is_relative_to(BASE), 'Ruta inválida'
    z.extractall(BASE)
manifest = json.loads((BASE/'PACKAGE_MANIFEST_COMPLEMENTO.json').read_text())
assert manifest['closed_zip_official'] and not manifest['local_test'], 'El paquete no salió del ZIP cerrado oficial'
for name, expected in manifest['files'].items(): assert sha_file(BASE/name) == expected, name
for name, expected in json.loads((BASE/'PACKAGE_MANIFEST.json').read_text()).items(): assert sha_file(BASE/name) == expected, name
PROJECT = BASE/'tesis_imo_para_codex'; SCRIPTS = PROJECT/'scripts'
record = OUTPUT/'paquete_usado.json'
info = {'package_sha256': package_sha, 'closed_zip_sha256': manifest['closed_zip_sha256'],
        'trainval_zip_sha256': manifest['trainval_zip_sha256'], 'embedded': {k: v[0] for k, v in EMBEDDED.items()}}
if record.exists() and json.loads(record.read_text())['embedded'] != info['embedded']:
    raise RuntimeError('OUTPUT tiene resultados de otra versión de los scripts: usa otra carpeta de salida')
record.write_text(json.dumps(info, indent=1))
print('Paquete verificado:', package_sha, '|', manifest['added_entries'], 'archivos agregados al ZIP cerrado')
'''

ENV='''# 3. Entorno aislado igual al del entrenamiento (Python 3.11, torch 2.5.1+cu121, Mamba). Repetir tras reiniciar.
run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.6.9'])
ENV = Path('/content/imo-e2-env'); PY = str(ENV/'bin/python')
if not Path(PY).exists(): run([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', str(ENV)])
def install(*items): run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PY, *items])
install('setuptools==75.8.0')
install('torch==2.5.1', 'torchvision==0.20.1', '--index-url', 'https://download.pytorch.org/whl/cu121')
install('-r', str(PROJECT/'requirements-e2.txt'))
install('transformers==4.46.3')
install('--no-deps', 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl')
install('--no-deps', 'https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl')
site = ENV/'lib/python3.11/site-packages'
dirs = [site/'torch/lib', *sorted((site/'nvidia').glob('*/lib'))]
paths = [str(p) for p in dirs if p.is_dir()] + os.environ.get('LD_LIBRARY_PATH', '').split(':')
os.environ['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(p for p in paths if p))
expected = json.loads((PROJECT/'config/protocolo_test_e2_cerrado.json').read_text())['training_environment']['packages']
check = (f"import importlib.metadata as m,json,torch,mamba_ssm,causal_conv1d; assert torch.cuda.is_available(); "
         f"e=json.loads({json.dumps(json.dumps(expected))}); a={{k:m.version(k) for k in e}}; "
         f"d={{k:(a[k],e[k]) for k in e if a[k]!=e[k]}}; assert not d, d; print('Entorno = protocolo;', torch.cuda.get_device_name(0))")
run([PY, '-u', '-c', check])
(OUTPUT/f'environment_{stamp()}.freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'uv', 'pip', 'freeze', '--python', PY], text=True))
def stage(name, *extra, output=OUTPUT, log_dir=LOGS):
    run([PY, '-u', SCRIPTS/'diagnosticos_gpu_e2.py', '--runs', RUNS, '--output', output, '--stage', name, *extra],
        log=log_dir/f'{name}_{stamp()}.log')
'''

PILOT_MD='''## 4. Piloto de tiempos (antes de las 450 tareas de robustez)
Corre las tres etapas con la semilla 11 de cada red sobre pocas imágenes (`--limit`), en disco local, y extrapola al
total con los tiempos que registra cada tarea. **Es una estimación**: no incluye esperas de Drive ni cambios de GPU.
La verificación de los 50 checkpoints se paga una vez por cada lanzamiento de etapa y se mide aparte.
'''

PILOT='''# 4. Piloto: ~10-20 min según la GPU (también estimado). Escribe OUTPUT/PILOTO_TIEMPOS.json
N = 8
if PILOT.exists(): shutil.rmtree(PILOT)
PILOT.mkdir(parents=True)
# --limit entra en la firma de la carpeta: interpret (limit 2) va en su propia subcarpeta
PILOT_IG = PILOT/'interpret'
for name, limit, out in [('clean', N, PILOT), ('robustness', N, PILOT), ('interpret', 2, PILOT_IG)]:
    stage(name, '--seeds', '11', '--limit', str(limit), output=out)
cfg = json.loads((PROJECT/'config/complementos_e2_20261120.json').read_text())
FULL = {'train': 1015, 'val': 218, 'test': 218}; SEEDS = 5
logs = {json.loads(p.read_text())['stage']: json.loads(p.read_text()) for p in sorted(PILOT.glob('**/tiempos/*.json'))}
load = {r['folder'].rsplit('_seed', 1)[0]: r['load_seconds'] for r in logs['clean']['runs']}
rows = []; total = {'clean': 0., 'robustness': 0., 'interpret': 0.}
for f in sorted(PILOT.glob('*_seed11_clean.json')):
    r = json.loads(f.read_text()); m = r['model']; s = r['seconds']
    per = {k: s[k] / r['splits'][k]['images'] for k in FULL}
    clean_s = SEEDS * (sum(per[k] * FULL[k] for k in FULL) + (s['total'] - sum(s[k] for k in FULL)) + load[m])
    rob = [json.loads(p.read_text()) for p in PILOT.glob(f'{m}_seed11_*_*.json') if '_clean' not in p.name and '_interpret' not in p.name]
    rob_s = SEEDS * (sum(x['seconds'] / x['images'] for x in rob) * FULL['test'] + load[m]) if rob else float('nan')
    ip = PILOT_IG/f'{m}_seed11_interpret.json'
    int_s = SEEDS * (json.loads(ip.read_text())['seconds'] / json.loads(ip.read_text())['cases'] * cfg['interpretability']['cases'] + load[m]) if ip.exists() else 0.
    total['clean'] += clean_s; total['robustness'] += rob_s; total['interpret'] += int_s
    rows.append({'model': m, 'clean_min': clean_s / 60, 'robustness_min': rob_s / 60, 'interpret_min': int_s / 60})
verif = {k: v['verification_seconds'] for k, v in logs.items()}
for k in total: total[k] += verif[k]
estimate = {'ESTIMADO': 'extrapolado de un piloto con --limit, semilla 11; no es una medición de la corrida completa',
            'gpu': logs['clean']['device'], 'per_model': rows, 'verification_seconds': verif,
            'total_hours': {k: v / 3600 for k, v in total.items()}}
(OUTPUT/'PILOTO_TIEMPOS.json').write_text(json.dumps(estimate, indent=1, ensure_ascii=False))
for r in rows: print(f"{r['model']:15s} clean {r['clean_min']:7.1f} min | robustez {r['robustness_min']:7.1f} min | IG {r['interpret_min']:6.1f} min")
print('ESTIMADO total (h):', {k: round(v, 2) for k, v in estimate['total_hours'].items()})
'''

STAGE_MD='''## {n}. Etapa `{name}`
{text}
Si la sesión se corta: vuelve a correr las celdas 1 a 3 y esta misma; lo terminado se salta.
'''

ROBUST='''# 6. Robustez: 450 tareas. Para repartirla entre sesiones, pon aquí una lista de redes, p. ej. ['transunet_2d','vmunet']
MODELS = None
stage('robustness', *(['--models', *MODELS] if MODELS else []))
'''

SUMMARY='''# 8. Resumen: verifica firmas y hashes (JSON y NPZ), calcula intervalos, ensembles y conformal en validación
run([PY, '-u', SCRIPTS/'resumir_diagnosticos_gpu_e2.py', '--output', OUTPUT], log=LOGS/f'resumen_{stamp()}.log')
summary = json.loads((OUTPUT/'RESUMEN_DIAGNOSTICOS.json').read_text())
print('Estado:', summary['status'], '| faltantes:', len(summary['missing']))
for e in summary['ensembles']:
    c = e['conformal']['mondrian_by_class']
    print(e['model'], 'cobertura conforme rótulo', round(c['label']['coverage'], 3), 'fondo', round(c['background']['coverage'], 3),
          '| marginal (un umbral) rótulo', round(e['conformal']['marginal_single_threshold_reference']['label']['coverage'], 3))
'''

EXPORT='''# 9. Descargar los JSON (sin NPZ ni pesos; los NPZ quedan en Drive, verificados por hash en el resumen)
EXPORT = Path('/content/diagnosticos_gpu_e2_json.zip')
with zipfile.ZipFile(EXPORT, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUTPUT.rglob('*')):
        if p.is_file() and p.suffix in {'.json', '.log', '.txt'}: z.write(p, p.relative_to(OUTPUT))
print(EXPORT, round(EXPORT.stat().st_size / 2**20, 1), 'MiB')
from google.colab import files
files.download(str(EXPORT))
'''

def main():
    embedded={rel:[hashlib.sha256((PROJECT/rel).read_bytes()).hexdigest(),base64.b64encode((PROJECT/rel).read_bytes()).decode()]
              for rel in COMPLEMENT}
    cells=[]
    def add(kind,text):
        cell={'cell_type':kind,'id':f'complemento-gpu-{len(cells):02d}','metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown',INTRO);add('code',PARAMS)
    add('code',PACKAGE_HEAD+json.dumps(embedded,indent=0)+'\n'+PACKAGE_BODY)
    add('code',ENV);add('markdown',PILOT_MD);add('code',PILOT)
    add('markdown',STAGE_MD.format(n=5,name='clean',text='50 corridas × (1015 train + 218 val + 218 test). Guarda probabilidades float16 de val y test del top 3 (NPZ en Drive).'))
    add('code',"# 5. Clean\nstage('clean')\n")
    add('markdown',STAGE_MD.format(n=6,name='robustness',text='10 redes × 5 semillas × 3 perturbaciones × 3 severidades sobre las 218 imágenes de test. Revisa antes `PILOTO_TIEMPOS.json`.'))
    add('code',ROBUST)
    add('markdown',STAGE_MD.format(n=7,name='interpret',text='Top 3 por validación × 5 semillas × 100 casos de validación; IG de 16 pasos más 6 pasadas de fidelidad por caso.'))
    add('code',"# 7. Explicabilidad\nstage('interpret')\n")
    add('markdown','## 8. Resumen y 9. Exportación\nEl resumen se puede correr con etapas incompletas: marca `partial` y lista lo que falta.\n')
    add('code',SUMMARY);add('code',EXPORT)
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'accelerator':'GPU','colab':{'name':'Complemento_GPU_E2.ipynb','gpuType':'L4'}},
        'nbformat':4,'nbformat_minor':5}
    OUT.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    for cell in cells:
        if cell['cell_type']=='code':compile(''.join(cell['source']),cell['id'],'exec')
    print(json.dumps({'notebook':str(OUT.relative_to(PROJECT)),'cells':len(cells),
                      'embedded':{k:v[0][:16] for k,v in embedded.items()},
                      'closed_zip_sha256':CLOSED_ZIP_SHA256,'trainval_zip_sha256':TRAINVAL_ZIP_SHA256},indent=1))

if __name__=='__main__':main()
