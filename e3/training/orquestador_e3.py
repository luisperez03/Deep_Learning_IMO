"""Una sesión de Colab, etapas reanudables. Cargado por el lanzador autocontenido."""
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

LANES = {'todo':['01','02','03','08','05','06','07','04','09','10'],
         'original':['01','02','03','04'], 'compactas':['05'],
         'rivales_a':['06'], 'rivales_b':['07'], 'rivales_c':['08'], 'cierre':['09','10'],
         'original_y_compactas':['01','02','03','04','05'], 'rivales':['08','06','07'],
         'sin_original':['08','05','06','07','09']}
ROOT = Path('/content/drive/MyDrive/tesis_imo_e3_secuencia_v4')


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
    return '\n'.join(line for line in source.splitlines() if not line.strip().startswith('files.download('))+'\n'


def run(notebooks, lane='todo'):
    if lane not in LANES:raise ValueError(f'Carril desconocido: {lane}; opciones {list(LANES)}')
    from google.colab import drive, files
    drive.mount('/content/drive')
    ROOT.mkdir(parents=True,exist_ok=True)
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
            scope={'__name__':'__main__'}
            started=time.perf_counter()
            for index,cell in enumerate(nb['cells']):
                if cell['cell_type']!='code':continue
                code=transformed(''.join(cell['source']),stage)
                exec(compile(code,f'{name}:celda{index}','exec'),scope)
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

