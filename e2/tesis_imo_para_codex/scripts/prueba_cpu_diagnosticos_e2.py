"""Prueba de humo en CPU del complemento GPU: mismas funciones de etapa, modelos SIN ENTRENAR, pocas imágenes.

Valida el flujo (lectura de datos, métricas, calibración, robustez, IG, NPZ, firmas, reanudación y resumidor),
no los números: con pesos aleatorios Dice, calibración y atribuciones no significan nada. No usa checkpoints,
no verifica el entorno del protocolo y deja VM-UNet fuera (exige CUDA y mamba-ssm).

Uso, dentro del paquete extraído o del proyecto con los datos al lado:
  python scripts/prueba_cpu_diagnosticos_e2.py --output <carpeta_nueva> --limit 3 --ig-steps 2
y después:
  python scripts/resumir_diagnosticos_gpu_e2.py --output <carpeta_nueva>
"""
import argparse,json,time
from pathlib import Path
import torch
import diagnosticos_gpu_e2 as D
from experimento_e2_colab import PROJECT,digest,atomic_json,environment,setup_seed
from modelos_e2 import MODELS_E2

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--limit',type=int,default=3)
    ap.add_argument('--ig-steps',type=int,default=2);ap.add_argument('--models',nargs='*')
    ap.add_argument('--seeds',nargs='*',type=int);ap.add_argument('--stages',nargs='*',default=['clean','robustness','interpret'])
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    D.DEVICE='cpu';D.LIMIT=args.limit;D.PROFILE_WARMUP,D.PROFILE_REPS=1,3
    protocol=D.read(PROJECT/'config/protocolo_test_e2_cerrado.json');cfg=D.read(PROJECT/'config/complementos_e2_20261120.json')
    cfg['interpretability']['steps']=args.ig_steps
    try:env=environment()
    except Exception as e:env={'error':f'{type(e).__name__}: {e}'}
    signature=digest({'smoke_cpu_untrained':True,'config':cfg,'limit':args.limit,'script':D.sha(Path(D.__file__)),
                      'metrics':D.sha(PROJECT/'scripts/metricas_diagnostico_e2.py')})
    lock=out/'PROTOCOLO_DIAGNOSTICOS.json'
    if lock.exists() and D.read(lock)['signature']!=signature:raise RuntimeError('Carpeta de otra prueba; usa una nueva')
    atomic_json(lock,{'signature':signature,'config':cfg,'environment':env,'closed_protocol':protocol,'limit':args.limit,
                      'WARNING':'PRUEBA DE HUMO EN CPU CON MODELOS SIN ENTRENAR: los números no tienen significado'})
    norm=D.read(PROJECT/'resultados/preentrenamiento/normalization_train.json')
    models=args.models or [m for m in MODELS_E2 if m!='vmunet'];skipped={}
    for m in list(models):
        try:MODELS_E2[m]()
        except Exception as e:skipped[m]=f'{type(e).__name__}: {e}';models.remove(m);print('Sin probar en este entorno:',m,skipped[m],flush=True)
    runs=[r for r in protocol['runs'] if r['model'] in models and (not args.seeds or r['seed'] in args.seeds)]
    times={}
    for stage in args.stages:
        t0=time.perf_counter()
        for run in runs:
            if stage=='interpret' and run['model'] not in cfg['top3_by_validation']:continue
            setup_seed(run['seed']);torch.set_num_threads(4)
            model=MODELS_E2[run['model']]();model.eval()
            D.STAGES[stage](model,run,out,signature,cfg,norm)
        times[stage]=time.perf_counter()-t0;print('Etapa',stage,'lista en',round(times[stage],1),'s',flush=True)
    atomic_json(out/'tiempos_prueba_cpu.json',{'limit':args.limit,'ig_steps':args.ig_steps,'runs':len(runs),'seconds':times,
                                                 'device':'cpu','models':models,'skipped_not_buildable':skipped})

if __name__=='__main__':main()
