"""Construye ZIP v2 y notebook; sólo lee datos originales. No sube archivos."""
import csv
import hashlib
import json
import os
from pathlib import Path
import zipfile

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent

def sha_bytes(data): return hashlib.sha256(data).hexdigest()

def create_notebook(zip_hash):
    cells=[]
    def md(text): cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text): cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':text.splitlines(keepends=True)})
    md('''# Benchmark E2 corregido — arquitecturas reales, v2
Ejecuta las celdas en orden en **Colab con GPU**. Sube `paquete_colab_e2_v2.zip` al panel de archivos o a la raíz de Drive.
Se usan datos locales congelados train/val; no se evalúa test ni se descargan pesos preentrenados.
El preflight prueba cada arquitectura real y detiene el flujo si falla. VM-UNet necesita extensión CUDA: su ejecución completa se valida aquí, no en el portátil CPU.
Los checkpoints v2 usan una carpeta nueva: no reutilices resultados de los prototipos anteriores.
El benchmark es una adaptación binaria con recetas fijas; no es réplica de los experimentos originales ni un tuning exhaustivo.
''')
    code('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, os, json, hashlib, zipfile
subprocess.run(['nvidia-smi'], check=True)
subprocess.run(['mountpoint', '-q', '/content/drive'], check=True)
OUTPUT = Path('/content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2')
OUTPUT.mkdir(parents=True, exist_ok=True)
''')
    code(f'''EXPECTED_ZIP_SHA256 = '{zip_hash}'
ZIP = next((p for p in [Path('/content/paquete_colab_e2_v2.zip'), Path('/content/drive/MyDrive/paquete_colab_e2_v2.zip')] if p.exists()), None)
if ZIP is None: raise FileNotFoundError('Sube paquete_colab_e2_v2.zip al panel de archivos y repite esta celda.')
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
assert sha_file(ZIP)==EXPECTED_ZIP_SHA256, 'ZIP incorrecto o modificado; usa el archivo v2 entregado con este notebook'
BASE=Path('/content/imo_e2_v2')
BASE.mkdir(exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    for n in z.namelist():
        if not (BASE/n).resolve().is_relative_to(BASE): raise ValueError('Ruta no válida en ZIP')
    z.extractall(BASE)
manifest=json.loads((BASE/'PACKAGE_MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert sha_file(BASE/name)==expected, name
PROJECT=BASE/'tesis_imo_para_codex'
print('Paquete íntegro:',len(manifest['files']),'archivos')
''')
    md('''## Entorno aislado y fijado
Se crea Python 3.11 fuera del kernel de Colab. La instalación puede tardar varios minutos; no cambia el Python del notebook. No se compilan extensiones CUDA: se usan ruedas publicadas por sus autores para torch 2.5/CUDA 12.
''')
    code('''subprocess.run([sys.executable,'-m','pip','install','uv==0.6.9'],check=True)
ENV=Path('/content/imo-e2-env')
if not (ENV/'bin/python').exists():
    subprocess.run([sys.executable,'-m','uv','venv','--python','3.11',str(ENV)],check=True)
PY=str(ENV/'bin/python')
def pip_install(*items):
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',PY,*items],check=True)
pip_install('torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu121')
pip_install('-r',str(PROJECT/'requirements-e2.txt'))
pip_install('transformers==4.46.3')
pip_install('--no-deps','https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl')
pip_install('--no-deps','https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl')
subprocess.run([PY,'-c','import torch, torchvision, mamba_ssm; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0))'],check=True)
freeze=subprocess.check_output([sys.executable,'-m','uv','pip','freeze','--python',PY],text=True)
(OUTPUT/'environment.freeze.txt').write_text(freeze)
''')
    md('''## Comprobación previa obligatoria
Comprueba hashes de datos/código, separación de grupos, identidad arquitectónica, forward, gradientes, paso de optimizador y evaluación con datos sintéticos. Guarda memoria GPU y tiempo. Si falla un modelo, corrige el entorno o reduce **explícitamente** el alcance; no hay sustituciones automáticas. El lote nominal es 2; el último lote de una imagen se une al anterior (3) para BatchNorm.
''')
    code('''MODELS=['fcn','segnet','unet','deeplabv3plus','unetplusplus','pidnet_s','segnext_t','segformer_b0','transunet_2d','vmunet']
subprocess.run([PY,str(PROJECT/'scripts/preflight_e2.py'),'--output',str(OUTPUT),'--models',*MODELS],check=True)
report=json.loads((OUTPUT/'preflight.json').read_text())
for name,result in report['models'].items(): print(name,result)
''')
    md('''## Ejecutar o reanudar
La siguiente celda inicia las corridas. Por defecto son 10 modelos × 5 semillas, no una prueba rápida. Puedes empezar con `MODELS_RUN=['segformer_b0','pidnet_s']` y después ejecutar las restantes usando la misma configuración.
Conserva la carpeta de Drive y repite las celdas al reconectar. Se recupera la última época completa; el trabajo de una época interrumpida se repite. Si cambia la firma, se exige otra carpeta. No hay garantía de igualdad bit a bit al cambiar GPU/backend.
''')
    code('''MODELS_RUN=MODELS
SEEDS=[11,23,37,53,71]
subprocess.run([PY,str(PROJECT/'scripts/experimento_e2_colab.py'),
    '--output',str(OUTPUT),'--models',*MODELS_RUN,'--seeds',*[str(s) for s in SEEDS]],check=True)
''')
    code('''for p in sorted(OUTPUT.glob('*/result.json')):
    r=json.loads(p.read_text())
    print(r['model'],r['seed'],r['completed'],r['best_val_group_Dice'],r['best_file'])
print('Los resultados permanecen en',OUTPUT)
''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'Benchmark_E2_Colab_v2.ipynb'}},'nbformat':4,'nbformat_minor':5}
    for i,c in enumerate(cells): c['id']=f'e2-cell-{i:02d}'
    raw=json.dumps(notebook,ensure_ascii=False,indent=1)
    (PROJECT/'notebooks/benchmark_e2_colab.ipynb').write_text(raw,encoding='utf-8')
    (ROOT/'Benchmark_E2_Colab_v2.ipynb').write_text(raw,encoding='utf-8')

def main():
    selected=set((PROJECT/'scripts').glob('*.py'))
    selected.update(p for p in (PROJECT/'third_party/e2').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    selected.update([PROJECT/'config/experimento_e2.json',PROJECT/'requirements-e2.txt',PROJECT/'COLAB_E2_LEEME.md'])
    pre=PROJECT/'resultados/preentrenamiento'
    selected.update(pre/f for f in ['train.csv','val.csv','test.csv','normalization_train.json'])
    for split in ['train','val']:
        with (pre/f'{split}.csv').open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                selected.add(ROOT/row['image_path']);selected.add(PROJECT/row['binary_mask_path'])
    output=ROOT/'paquete_colab_e2_v2.zip'; temp=output.with_suffix('.tmp')
    manifest={'schema':2,'note':'Train/val images and masks only; test metadata for leakage checks','files':{}}
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(selected):
            data=path.read_bytes();name=path.relative_to(ROOT).as_posix()
            z.writestr(name,data);manifest['files'][name]=sha_bytes(data)
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2))
    os.replace(temp,output)
    # Releer ZIP completo: CRC, rutas y hashes de todos los miembros.
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for name,expected in manifest['files'].items(): assert sha_bytes(z.read(name))==expected,name
    checksum=sha_bytes(output.read_bytes())
    output.with_suffix('.zip.sha256').write_text(checksum+'  '+output.name+'\n',encoding='utf-8')
    create_notebook(checksum)
    print(json.dumps({'zip':str(output),'MiB':round(output.stat().st_size/2**20,2),'files':len(selected),'sha256':checksum},indent=2))

if __name__=='__main__': main()
