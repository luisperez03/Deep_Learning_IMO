"""Evaluación final E2 autorizada: inferencia de test, sin optimizador ni ajuste.
Reanuda por corrida completada verificando firma, hashes y resultados previos.
"""
import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from dataset_ready import partition,load_example,metrics
from modelos_e2 import MODELS_E2,main_logits
from experimento_e2_colab import PROJECT,sha,digest,atomic_json,environment,setup_seed

def grouped_scores(details):
    groups=defaultdict(list)
    for row in details:groups[row['group_id']].append(row)
    values={key:{metric:float(np.mean([r[metric] for r in rows]))
                 for metric in ['Dice','IoU','mIoU','accuracy']} for key,rows in groups.items()}
    means={metric:float(np.mean([g[metric] for g in values.values()])) for metric in ['Dice','IoU','mIoU','accuracy']}
    return means,values

def verify_inputs(protocol,runs):
    for relative,expected in protocol['files'].items():
        if sha(PROJECT/relative)!=expected:raise RuntimeError(f'Archivo modificado: {relative}')
    rows=partition('test',allow_test=True)
    if [r['id'] for r in rows]!=protocol['test_order']:raise RuntimeError('Orden de test modificado')
    test_groups={r['group_id'] for r in rows}
    for part in ['train','val']:
        if test_groups & {r['group_id'] for r in partition(part)}:raise RuntimeError('Fuga de grupos')
    for row in rows:
        for path,expected in [(PROJECT.parent/row['image_path'],row['image_sha256']),
                              (PROJECT/row['binary_mask_path'],row['binary_mask_sha256'])]:
            if sha(path)!=expected:raise RuntimeError(f'Datos modificados: {path}')
    for record in protocol['runs']:
        folder=runs/record['folder']
        for name,expected in [('config.json',record['config_sha256']),('result.json',record['result_sha256']),
                              (record['best_file'],record['best_sha256'])]:
            if sha(folder/name)!=expected:raise RuntimeError(f'Corrida modificada: {folder/name}')
    return rows

def summarize(output,protocol,signature):
    records=[]
    for run in protocol['runs']:
        path=output/(run['folder']+'.json')
        if not path.exists():raise RuntimeError('Faltan corridas; no emitir ranking parcial como final')
        result=json.loads(path.read_text())
        payload={k:v for k,v in result.items() if k!='payload_sha256'}
        if digest(payload)!=result['payload_sha256'] or result['signature']!=signature:raise RuntimeError('Resultado inválido')
        records.append(result)
    group_names=sorted(records[0]['groups'])
    by_model=defaultdict(list)
    for r in records:by_model[r['model']].append(r)
    rng=np.random.default_rng(20261119)
    group_indices=rng.integers(0,len(group_names),size=(5000,len(group_names)))
    summary=[];bootstrap={};point={}
    for name,results in by_model.items():
        results.sort(key=lambda r:r['seed'])
        if [r['seed'] for r in results]!=protocol['seeds']:raise RuntimeError('Semillas incompletas')
        values=np.array([[r['groups'][g]['Dice'] for g in group_names] for r in results])
        group_average=values.mean(axis=0)
        bootstrap[name]=group_average[group_indices].mean(axis=1);point[name]=float(values.mean())
        lo,hi=np.quantile(bootstrap[name],[.025,.975])
        summary.append({'model':name,'Dice_mean':point[name],'Dice_sd_seeds':float(values.mean(axis=1).std(ddof=1)),
            'Dice_CI95_groups_fixed_seeds':[float(lo),float(hi)],
            'IoU_mean':float(np.mean([r['means']['IoU'] for r in results])),
            'mIoU_mean':float(np.mean([r['means']['mIoU'] for r in results])),
            'seconds_inference_total':sum(r['seconds_inference'] for r in results)})
    summary.sort(key=lambda r:-r['Dice_mean'])
    contrasts=[]
    for name in by_model:
        if name=='fcn':continue
        lo,hi=np.quantile(bootstrap[name]-bootstrap['fcn'],[.025,.975])
        contrasts.append({'model':name,'reference':'fcn','Dice_difference':point[name]-point['fcn'],
            'CI95_groups_fixed_seeds':[float(lo),float(hi)],'primary':name=='transunet_2d'})
    final={'signature':signature,'runs':len(records),'test_images':len(protocol['test_order']),
        'test_groups':len(group_names),'summary':summary,'contrasts':contrasts,
        'inference':'IC percentiles por grupos, cinco semillas fijas. Contrastes secundarios exploratorios sin ajuste de multiplicidad.',
        'prior_test_exposure':'El test tuvo exposición en E1; no es un holdout nunca observado.'}
    atomic_json(output/'RESUMEN_TEST.json',final)
    lines=['# Evaluación final de test E2','',f"{len(records)} corridas; {len(protocol['test_order'])} imágenes; {len(group_names)} grupos.",'',
        '| Modelo | Dice ± DE semillas | IC95 por grupos | IoU |','|---|---:|---:|---:|']
    for r in summary:
        lo,hi=r['Dice_CI95_groups_fixed_seeds']
        lines.append(f"| {r['model']} | {r['Dice_mean']:.4f} ± {r['Dice_sd_seeds']:.4f} | [{lo:.4f}, {hi:.4f}] | {r['IoU_mean']:.4f} |")
    lines+=['',final['inference'],final['prior_test_exposure'],
        'No se entrenó, ajustó umbral ni seleccionó checkpoint con test. Las diez recetas se cerraron antes de esta evaluación.',
        'Los tiempos incluyen carga de imágenes y transferencia; no equivalen a latencia pura del modelo.']
    (output/'INFORME_TEST.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return final

@torch.inference_mode()
def evaluate(protocol,runs,output):
    if not torch.cuda.is_available():raise RuntimeError('Las diez redes se evalúan juntas en CUDA, incluido VM-UNet')
    output.mkdir(parents=True,exist_ok=True)
    print('Verificando datos, código y los 50 checkpoints...',flush=True)
    rows=verify_inputs(protocol,runs)
    env=environment()
    expected_env=protocol['training_environment']
    # Mismo software principal; registrar GPU actual sin impedir inferencia si cambió el modelo GPU.
    if env['packages']!=expected_env['packages'] or env['cuda']!=expected_env['cuda']:
        raise RuntimeError(f'Entorno diferente al entrenamiento. Actual: {env}; esperado: {expected_env}')
    signature=digest({'protocol':protocol,'environment':env,'evaluator':sha(Path(__file__))})
    lock=output/'EVALUACION_CERRADA.json'
    if lock.exists() and json.loads(lock.read_text())['signature']!=signature:
        raise RuntimeError('No mezclar evaluaciones con firmas diferentes en una carpeta')
    atomic_json(lock,{'signature':signature,'environment':env,'protocol':protocol})
    device=torch.device('cuda')
    for i,run in enumerate(protocol['runs'],1):
        dest=output/(run['folder']+'.json')
        if dest.exists():
            old=json.loads(dest.read_text());payload={k:v for k,v in old.items() if k!='payload_sha256'}
            if old.get('signature')!=signature or digest(payload)!=old.get('payload_sha256'):raise RuntimeError('Resultado previo incompatible/corrupto')
            print(f"[{i}/50] Ya evaluado: {run['folder']}",flush=True);continue
        setup_seed(run['seed'])
        ck=torch.load(runs/run['folder']/run['best_file'],map_location='cpu',weights_only=True)
        if ck['fingerprint']!=run['fingerprint']:raise RuntimeError('Fingerprint de checkpoint incorrecto')
        model=MODELS_E2[run['model']]().to(device)
        model.load_state_dict(ck['model'],strict=True);model.eval();del ck
        assert not any(m.training for m in model.modules())
        torch.manual_seed(protocol['inference_rng_seed'])
        details=[];torch.cuda.synchronize();start=time.perf_counter()
        for j,row in enumerate(rows,1):
            x,y,valid=load_example(row)
            x=torch.from_numpy(x.copy())[None].to(device)
            probabilities=main_logits(model(x)).sigmoid()[0,0].cpu().numpy()
            if not np.isfinite(probabilities).all():raise RuntimeError('Predicción no finita')
            scores=metrics(probabilities,y,valid)
            details.append({'id':row['id'],'group_id':row['group_id'],**scores})
            if run['seed']==11:
                masks=output/'mascaras_seed11'/run['model'];masks.mkdir(parents=True,exist_ok=True)
                Image.fromarray(((probabilities>=.5)&valid).astype('uint8')*255).save(masks/(row['id']+'.png'))
            if j%50==0 or j==len(rows):print(f"[{i}/50] {run['folder']}: {j}/{len(rows)} imágenes",flush=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        means,groups=grouped_scores(details)
        result={'signature':signature,'model':run['model'],'seed':run['seed'],'checkpoint_sha256':run['best_sha256'],
            'seconds_inference':elapsed,'means':means,'groups':groups,'details':details}
        result['payload_sha256']=digest(result);atomic_json(dest,result)
        print(run['folder'],'Dice test',means['Dice'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    summarize(output,protocol,signature)
    print('Evaluación completa. RESUMEN_TEST.json e INFORME_TEST.md guardados.',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True)
    p.add_argument('--protocol',default=str(PROJECT/'config/protocolo_test_e2_cerrado.json'))
    args=p.parse_args()
    evaluate(json.loads(Path(args.protocol).read_text()),Path(args.runs),Path(args.output))

if __name__=='__main__':main()
