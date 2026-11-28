"""Una sesión de Colab, etapas reanudables. Cargado por el lanzador autocontenido."""
import hashlib
import json
import os
import time
import traceback
import ast
import contextlib
import subprocess
import sys
import threading
import uuid
import zipfile
from pathlib import Path

LANES = {'todo':['01','02','03','08','05','06','07','04','09','10'],
         'original':['01','02','03','04'], 'compactas':['05'],
         'rivales_a':['06'], 'rivales_b':['07'], 'rivales_c':['08'], 'cierre':['09','10'],
         'original_y_compactas':['01','02','03','04','05'], 'rivales':['08','06','07'],
         'sin_original':['08','05','06','07','09']}
ROOT = Path('/content/drive/MyDrive/tesis_imo_e3_secuencia_v4')


def source_digest(project):
    """La misma selección y serialización de source_signature() en Colab (rutas POSIX)."""
    paths=list((project/'scripts').glob('*.py'))+list((project/'third_party/e2').rglob('*.py'))
    paths += [project/'third_party/e2/sources.lock.json',project/'requirements-e2.txt']
    values={p.relative_to(project).as_posix():sha(p) for p in sorted(paths)}
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def supplemental_sources(notebooks):
    """Lee los bytes textuales congelados; nunca ejecuta el entrenamiento de la etapa 05."""
    nb=next(b for n,b in notebooks.items() if n.startswith('05_'))
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FUENTES_E2' for t in node.targets):
                return ast.literal_eval(node.value)
    raise RuntimeError('Falta FUENTES_E2 histórico; no reconstruir desde archivos actuales')


def prepare_sources(notebooks, scope, stage):
    """Restaura el perfil de fuentes histórico antes de preflight, con 05 completa o no."""
    project=Path(scope['PROJECT']);output=Path(scope['OUTPUT']);extra=supplemental_sources(notebooks)
    expected=set()
    candidates=list(output.glob('*/config.json'))
    preflight=output/'preflight.json'
    if preflight.exists():candidates.append(preflight)
    for path in candidates:
        record=json.loads(path.read_text())
        value=record.get('identity',{}).get('source')
        if value:expected.add(value)
    if len(expected)>1:raise RuntimeError('La etapa contiene varias firmas históricas; no se modifica ni reanuda')
    # Validar las dos variantes a partir del ZIP íntegro. Solo se restauran fuentes locales.
    archive=Path(scope['ZIP']);base=Path(scope['BASE'])
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist())
        signed=list((project/'scripts').glob('*.py'))+list((project/'third_party/e2').rglob('*.py'))
        signed += [project/'third_party/e2/sources.lock.json',project/'requirements-e2.txt']
        for path in signed:
            rel=path.relative_to(project).as_posix()
            if rel in extra:continue
            member=path.relative_to(base).as_posix()
            if member not in names or path.read_bytes()!=z.read(member):
                raise RuntimeError('Fuente ajena o modificada respecto al ZIP histórico: '+rel)
        for rel in extra:
            target=(project/rel).resolve()
            if not target.is_relative_to(project.resolve()):raise ValueError('Fuente fuera de PROJECT')
            member=(project/rel).relative_to(base).as_posix()
            if member in names:
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(member))
            elif target.exists():
                # Nunca borrar un archivo desconocido: solo el efecto exacto de FUENTES_E2.
                if target.read_bytes()!=extra[rel].encode('utf-8'):
                    raise RuntimeError('Fuente local desconocida: '+rel)
                target.unlink()
        base_signature=source_digest(project)
        for rel,text in extra.items():
            target=project/rel;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(text.encode('utf-8'))
        augmented_signature=source_digest(project)
        wanted=next(iter(expected),augmented_signature if stage in ['05','06','07','10'] else base_signature)
        if wanted==base_signature:
            for rel in extra:
                target=project/rel;member=target.relative_to(base).as_posix()
                if member in names:target.write_bytes(z.read(member))
                elif target.exists():target.unlink()
        elif wanted!=augmented_signature:
            raise RuntimeError(f'No se pudo recuperar la firma histórica {wanted}; base={base_signature}, ampliada={augmented_signature}. No se altera ningún checkpoint.')
    actual=source_digest(project)
    if actual!=wanted:raise RuntimeError('Firma de fuentes distinta después de reconstruir')
    print(f'Fuentes preparadas etapa {stage}: {actual} (histórica={bool(expected)})',flush=True)
    return {'source':actual,'base':base_signature,'ampliada':augmented_signature,'historica':bool(expected)}


class Tee:
    def __init__(self, stream, logfile):
        self.stream=stream;self.logfile=logfile;self.lock=threading.Lock()
    def write(self,text):
        with self.lock:
            self.stream.write(text);self.stream.flush();self.logfile.write(text);self.logfile.flush()
        return len(text)
    def flush(self):
        self.stream.flush();self.logfile.flush()


def stream_run(args, *, check=False, capture_output=False, text=False, env=None, timeout=None, **kwargs):
    """Muestra ambas tuberías mientras corren; conserva la semántica de captura y errores."""
    if 'stdout' in kwargs or 'stderr' in kwargs or 'input' in kwargs:
        raise ValueError('Redirección no soportada; usar capture_output para conservar la salida')
    process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                             env=dict(os.environ if env is None else env,PYTHONUNBUFFERED='1'),**kwargs)
    buffers=[bytearray(),bytearray()]
    def pump(pipe,index,display):
        import codecs
        decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
        while True:
            chunk=pipe.read1(4096)
            if not chunk:break
            if capture_output:buffers[index].extend(chunk)
            display.write(decoder.decode(chunk));display.flush()
        display.write(decoder.decode(b'',final=True));display.flush();pipe.close()
    threads=[threading.Thread(target=pump,args=(pipe,i,display),daemon=True)
             for i,(pipe,display) in enumerate([(process.stdout,sys.stdout),(process.stderr,sys.stderr)])]
    for thread in threads:thread.start()
    try:process.wait(timeout=timeout)
    except BaseException:
        process.kill();process.wait()
        raise
    finally:
        for thread in threads:thread.join()
    values=[bytes(b).decode('utf-8',errors='replace') if text else bytes(b) for b in buffers]
    result=subprocess.CompletedProcess(args,process.returncode,*(values if capture_output else [None,None]))
    if check:result.check_returncode()
    return result


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def targets(stage):
    if stage in ['01','02','03']:
        seeds={'01':[11],'02':[23,37,53,71],'03':[89,97,101,103,107]}[stage]
        return [(f'unetat_{a}',s,ROOT/'original'/f'unetat_{a}_seed{s}')
                for a in ['dbnet','dbnet_at'] for s in seeds]
    if stage=='04':
        return [('unetat_dbnet_at',11,ROOT/'hpo'/f'hpo_original_t{i}'/'unetat_dbnet_at_seed11') for i in range(4)]
    if stage=='05':
        return [(m,s,ROOT/'compactas/fase2'/f'{m}_a0498_s{s}')
                for m in ['fcn','unet','segnet'] for s in [89,97,101,103,107]]
    if stage in ['06','07','08']:
        models={'06':['transunet_2d','pidnet_s'],'07':['vmunet','segnext_t','deeplabv3plus','unetplusplus'],
                '08':['segformer_b0']}[stage]
        seeds=[89,97,101,103,107] if stage!='08' else [11,23,37,53,71,89,97,101,103,107]
        session={'06':'A','07':'B','08':'C'}[stage]
        return [(m,s,ROOT/f'rivales/sesion_{session}'/f'{m}_seed{s}') for m in models for s in seeds]
    return []


def completed(stage):
    expected=targets(stage)
    if not expected:return False
    for model,seed,folder in expected:
        if not (folder/'result.json').is_file():return False
        r=json.loads((folder/'result.json').read_text())
        c=json.loads((folder/'config.json').read_text())
        assert r['completed'] and r.get('seed',r.get('semilla'))==seed
        assert r.get('model',r.get('familia'))==model and r['fingerprint']==c['fingerprint']
        best=r['best_file']; assert Path(best).name==best
        assert sha(folder/best)==r['best_sha256'], f'Checkpoint alterado: {folder}'
        if model.startswith('unetat_'):assert c['identity']['opencv']=='4.11.0'
        if stage=='05':assert c['tarea']['arq']=='a0498' and c['tarea']['w']==[64,64,64]
        if stage=='08':assert abs(c['config']['models'][model]['lr']-.0005573638936676344)<1e-14
    return True


def transformed(source, stage):
    # Entornos separados en la misma VM: instalar Mamba no cambia la identidad del original.
    if stage in ['06','07','08','10']:
        source=source.replace("ENV=Path('/content/imo-e2-env')", "ENV=Path('/content/imo-e3-rivales-env')")
        # Fuera de las fuentes que participan en los fingerprints; antes de importar Mamba.
        source=source.replace("pip_install('torch==2.5.1'", "pip_install('setuptools==84.2.1')\npip_install('torch==2.5.1'")
    source='\n'.join(line for line in source.splitlines() if not line.strip().startswith('files.download('))+'\n'
    class Calls(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='subprocess' and node.func.attr=='run':
                if stage in ['06','07','08','10'] and node.args and isinstance(node.args[0],ast.List):
                    for argument in node.args[0].elts:
                        if isinstance(argument,ast.Constant) and isinstance(argument.value,str) and argument.value.startswith('import torch, torchvision, mamba_ssm;'):
                            argument.value="import importlib.metadata; assert importlib.metadata.version('setuptools')=='84.2.1'; "+argument.value
                node.func=ast.copy_location(ast.Name(id='_e3_run',ctx=ast.Load()),node.func)
            return node
    # Transformar llamadas, no cadenas CONTENIDO/FUENTES_E2 que fijan hashes históricos.
    return ast.unparse(ast.fix_missing_locations(Calls().visit(ast.parse(source))))+'\n'


def run(notebooks, lane='todo'):
    if lane not in LANES:raise ValueError(f'Carril desconocido: {lane}; opciones {list(LANES)}')
    from google.colab import drive, files
    drive.mount('/content/drive')
    ROOT.mkdir(parents=True,exist_ok=True)
    logdir=ROOT/'logs'/f"{time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())}_{lane}_{uuid.uuid4().hex[:8]}"
    logdir.mkdir(parents=True,exist_ok=True)
    status_path=ROOT/f'progreso_{lane}.json'
    status={'carril':lane,'etapas':{},'nota':'Solo entrenamiento y validación; no test ni despliegue'}

    def save():
        temp=status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
        os.replace(temp,status_path)

    for stage in LANES[lane]:
        stamp={'inicio_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'estado':'verificando'}
        status['etapas'][stage]=stamp; save()
        try:
            if completed(stage):
                print(f'Etapa {stage}: ya completa, checkpoints verificados. Se conserva.',flush=True)
                stamp['estado']='completa_previa_verificada';save();continue
            name,nb=next((n,b) for n,b in notebooks.items() if n.startswith(stage+'_'))
            print(f'\nEJECUTANDO {name} — misma máquina, checkpoints en Drive',flush=True)
            stamp['estado']='ejecutando';save()
            scope={'__name__':'__main__','_e3_run':stream_run}
            started=time.perf_counter()
            prepared=False
            stamp['logs']={'stdout':str(logdir/f'{stage}.stdout.log'),'stderr':str(logdir/f'{stage}.stderr.log')};save()
            with (logdir/f'{stage}.stdout.log').open('a',encoding='utf-8') as out, (logdir/f'{stage}.stderr.log').open('a',encoding='utf-8') as err:
                with contextlib.redirect_stdout(Tee(sys.stdout,out)),contextlib.redirect_stderr(Tee(sys.stderr,err)):
                    try:
                        for index,cell in enumerate(nb['cells']):
                            if cell['cell_type']!='code':continue
                            code=transformed(''.join(cell['source']),stage)
                            exec(compile(code,f'{name}:celda{index}','exec'),scope)
                            if not prepared and 'PROJECT' in scope:
                                stamp['fuentes']=prepare_sources(notebooks,scope,stage);prepared=True;save()
                    except BaseException:
                        traceback.print_exc();raise
            if targets(stage):assert completed(stage), f'Etapa {stage} no dejó todos los resultados'
            stamp.update(estado='completa',segundos=time.perf_counter()-started);save()
        except BaseException:
            stamp.update(estado='interrumpida_o_fallida',error=traceback.format_exc()[-5000:]);save()
            print(f'Se detuvo en {stage}. Los checkpoints escritos permanecen en Drive. '
                  'Corrige el error o reconecta y ejecuta el mismo lanzador; no borres resultados.',flush=True)
            raise
    print('Carril terminado. Progreso:',status_path,flush=True)
    if lane in ['todo','cierre','sin_original']:
        import zipfile
        archive=ROOT/'resultados_para_revision.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in ROOT.rglob('*'):
                if p.is_file() and p.suffix in ['.json','.csv','.txt']:
                    z.write(p,str(p.relative_to(ROOT)))
        print('Resultados también guardados en Drive:',archive,flush=True)
        files.download(str(archive))

