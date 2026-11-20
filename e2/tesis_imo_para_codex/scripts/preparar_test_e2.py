"""Congela evaluación y construye paquete Colab separado sin alterar entrenamiento."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1];ROOT=PROJECT.parent
RUNS=ROOT/'tesis_imo_e2_arquitecturas_reales_v2'
def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    original=ROOT/'paquete_colab_e2_v2.zip'
    payload={}
    with zipfile.ZipFile(original) as archive:
        for name in archive.namelist():
            relative=name.removeprefix('tesis_imo_para_codex/')
            if name.startswith('tesis_imo_para_codex/') and (
                relative.startswith(('scripts/','third_party/e2/')) or relative=='requirements-e2.txt'):
                payload[name]=archive.read(name)
    evaluator=PROJECT/'scripts/evaluar_e2_cerrado.py'
    payload['tesis_imo_para_codex/scripts/evaluar_e2_cerrado.py']=evaluator.read_bytes()
    metadata=PROJECT/'resultados/preentrenamiento'
    for filename in ['train.csv','val.csv','test.csv','normalization_train.json']:
        path=metadata/filename;payload[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    with (metadata/'test.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for row in rows:
        for path in [ROOT/row['image_path'],PROJECT/row['binary_mask_path']]:
            payload[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    records=[];env=None
    for path in sorted(RUNS.glob('*_seed*/result.json')):
        r=json.loads(path.read_text());c=json.loads((path.parent/'config.json').read_text())
        if not r['completed']:raise RuntimeError('Corrida incompleta')
        if env is not None and env!=c['environment']:raise RuntimeError('Entornos no uniformes')
        env=c['environment']
        records.append({'folder':path.parent.name,'model':r['model'],'seed':r['seed'],
            'best_file':r['best_file'],'best_sha256':r['best_sha256'],'fingerprint':r['fingerprint'],
            'config_sha256':sha((path.parent/'config.json').read_bytes()),'result_sha256':sha(path.read_bytes())})
    assert len(records)==50
    files={n.removeprefix('tesis_imo_para_codex/'):sha(data) for n,data in payload.items()
           if n.startswith('tesis_imo_para_codex/') and not '/masks_binary/' in n}
    protocol={'schema':1,'authorization':'Usuario: dale procede, tras propuesta de congelar y evaluar test de las 50 corridas',
        'threshold':.5,'size':256,'inference_batch':1,'inference_rng_seed':20261118,
        'seeds':[11,23,37,53,71],'test_order':[r['id'] for r in rows],
        'primary_metric':'Dice foreground: media por imagen dentro de grupo, después grupos y semillas',
        'primary_contrast':'TransUNet menos FCN; elegido por validación antes del test',
        'uncertainty':'5000 bootstrap de grupos, cinco semillas fijas; DE semillas por separado',
        'test_prior_exposure':'Test observado en E1. No ajustar hiperparámetros/umbral con esta evaluación.',
        'training_environment':env,'runs':records,'files':files}
    protocol_path=PROJECT/'config/protocolo_test_e2_cerrado.json'
    protocol_path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    payload['tesis_imo_para_codex/config/protocolo_test_e2_cerrado.json']=protocol_path.read_bytes()
    manifest={name:sha(data) for name,data in payload.items()}
    payload['PACKAGE_MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    zip_path=ROOT/'paquete_test_e2_cerrado.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(payload.items()):archive.writestr(name,data)
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        for name,expected in manifest.items():assert sha(archive.read(name))==expected
    zip_hash=sha(zip_path.read_bytes())
    cells=[]
    def add(kind,text):
        cell={'cell_type':kind,'id':f'test-e2-{len(cells)}','metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown','''# Evaluación final E2 — 50 checkpoints cerrados
Sólo inferencia de test, sin entrenamiento ni ajuste. El usuario autorizó este cierre tras revisar validación.
Selecciona **GPU**. Conserva en Drive `tesis_imo_e2_arquitecturas_reales_v2` con las 50 carpetas originales.
Sube **paquete_test_e2_cerrado.zip** al panel de archivos o a la raíz de Drive. Este ZIP contiene el test y código fijado, no los checkpoints.
No ejecutar a la vez el notebook de entrenamiento. El test tuvo exposición previa en E1; se declara esa limitación.
''')
    add('code','''from google.colab import drive
drive.mount('/content/drive')
import os, sys, subprocess, json, hashlib, zipfile
from pathlib import Path
def run(command, check=True):
    with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=os.environ.copy()) as process:
        for line in process.stdout: print(line,end='',flush=True)
        code=process.wait()
    if check and code: raise RuntimeError(f'Comando fallido ({code}); revisa el error anterior: {command}')
RUNS=Path('/content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2')
OUTPUT=Path('/content/drive/MyDrive/tesis_imo_e2_test_final_v1')
assert RUNS.is_dir(), 'Falta la carpeta de checkpoints en Drive'
OUTPUT.mkdir(parents=True,exist_ok=True)
run(['mountpoint','-q','/content/drive'])
run(['nvidia-smi'])
''')
    add('code',f'''EXPECTED_SHA256='{zip_hash}'
ZIP=next((p for p in [Path('/content/paquete_test_e2_cerrado.zip'),Path('/content/drive/MyDrive/paquete_test_e2_cerrado.zip')] if p.exists()),None)
if ZIP is None: raise FileNotFoundError('Sube paquete_test_e2_cerrado.zip y repite esta celda')
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
assert sha_file(ZIP)==EXPECTED_SHA256,'ZIP distinto del protocolo cerrado'
BASE=Path('/content/imo_e2_test_final');BASE.mkdir(exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    for name in z.namelist():
        assert (BASE/name).resolve().is_relative_to(BASE),'Ruta inválida'
    z.extractall(BASE)
for name,expected in json.loads((BASE/'PACKAGE_MANIFEST.json').read_text()).items():
    assert sha_file(BASE/name)==expected,name
PROJECT=BASE/'tesis_imo_para_codex'
print('Paquete verificado. Checkpoints se leerán de Drive.')
''')
    add('markdown','## Entorno completo — también funciona después de reiniciar la sesión\nIncluye setuptools, cuya ausencia causó el error anterior. No cambia los archivos de entrenamiento.')
    add('code','''run([sys.executable,'-m','pip','install','uv==0.6.9'])
ENV=Path('/content/imo-e2-env');PY=str(ENV/'bin/python')
if not Path(PY).exists():run([sys.executable,'-m','uv','venv','--python','3.11',str(ENV)])
def install(*items):run([sys.executable,'-m','uv','pip','install','--python',PY,*items])
install('setuptools==75.8.0')
install('torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu121')
install('-r',str(PROJECT/'requirements-e2.txt'))
install('transformers==4.46.3')
install('--no-deps','https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl')
install('--no-deps','https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl')
site=ENV/'lib/python3.11/site-packages'
dirs=[site/'torch/lib',*sorted((site/'nvidia').glob('*/lib'))]
paths=[str(p) for p in dirs if p.is_dir()]+os.environ.get('LD_LIBRARY_PATH','').split(':')
os.environ['LD_LIBRARY_PATH']=':'.join(dict.fromkeys(p for p in paths if p))
run([PY,'-u','-c','import torch,torchvision,triton,mamba_ssm; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0))'])
freeze=subprocess.check_output([sys.executable,'-m','uv','pip','freeze','--python',PY],text=True)
(OUTPUT/'environment.freeze.txt').write_text(freeze)
''')
    add('markdown','''## Evaluar o continuar
Verifica los 50 checkpoints antes de inferencia. Muestra progreso cada 50 imágenes y guarda cada corrida completa. Si se interrumpe una corrida, sólo esa corrida se repite; las completadas se verifican y omiten.
No hay optimizador, backward, búsqueda de umbral ni selección de checkpoint con test. Usa las mismas cinco semillas y umbral 0,5.
''')
    add('code',"run([PY,'-u',str(PROJECT/'scripts/evaluar_e2_cerrado.py'),'--runs',str(RUNS),'--output',str(OUTPUT)])\n")
    add('code','''summary=json.loads((OUTPUT/'RESUMEN_TEST.json').read_text())
for r in summary['summary']:
    print(r['model'], 'Dice:',round(r['Dice_mean'],4),'DE semillas:',round(r['Dice_sd_seeds'],4),'IC:',r['Dice_CI95_groups_fixed_seeds'])
print((OUTPUT/'INFORME_TEST.md').read_text())
''')
    add('markdown','## Descargar resultados para revisión\nIncluye métricas por imagen/grupo/semilla, intervalos, protocolo y máscaras de la semilla 11. No incluye pesos ni imágenes originales.')
    add('code','''EXPORT=Path('/content/resultados_test_e2_final.zip')
with zipfile.ZipFile(EXPORT,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUTPUT.rglob('*')):
        if p.is_file() and not p.name.endswith('.tmp'):z.write(p,p.relative_to(OUTPUT))
from google.colab import files
files.download(str(EXPORT))
''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    text=json.dumps(notebook,indent=1,ensure_ascii=False)
    (ROOT/'Evaluacion_Final_E2_Colab.ipynb').write_text(text,encoding='utf-8')
    (PROJECT/'notebooks/evaluacion_final_e2.ipynb').write_text(text,encoding='utf-8')
    print(json.dumps({'test_images':len(rows),'test_groups':len({r['group_id'] for r in rows}),
                      'checkpoints':len(records),'zip_MiB':zip_path.stat().st_size/2**20,'sha256':zip_hash},indent=2))

if __name__=='__main__':main()
