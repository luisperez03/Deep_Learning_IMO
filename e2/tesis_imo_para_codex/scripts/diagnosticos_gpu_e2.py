"""Complementos exploratorios sobre pesos cerrados: clean, robustez y explicaciones.
Reanudación por corrida/condición, firma y hashes. Ningún optimizador ni cambio de pesos.

Etapas (--stage): clean (train/val/test limpios, calibración, bordes, recursos), robustness
(3 perturbaciones x 3 severidades sobre test) e interpret (Integrated Gradients, top 3 por validación).
--limit N y --seeds sirven para el piloto de tiempos: con --limit la firma cambia y la carpeta de salida
debe ser distinta de la definitiva (el candado PROTOCOLO_DIAGNOSTICOS.json impide mezclarlas).
"""
import argparse,gc,json,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scipy.special import expit,logit
from dataset_ready import partition,load_example,metrics
from modelos_e2 import MODELS_E2,main_logits
from experimento_e2_colab import PROJECT,sha,digest,atomic_json,environment,setup_seed
from evaluar_e2_cerrado import verify_inputs
from metricas_diagnostico_e2 import diagnose,fit_temperature,perturb,aggregate,calibration_bins

DEVICE='cuda'   # La prueba en CPU (prueba_cpu_diagnosticos_e2.py) lo cambia a 'cpu'; main() exige CUDA.
LIMIT=None      # Imágenes por partición (piloto). None = todas.
PROFILE_WARMUP,PROFILE_REPS=10,50

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,r,signature):
    r={**r,'signature':signature};r['payload_sha256']=digest(r);atomic_json(p,r)
def valid(p,signature):
    if not p.exists():return False
    r=read(p);saved=r.pop('payload_sha256',None)
    if r.get('signature')!=signature or saved!=digest(r):raise RuntimeError(f'Incompatible/corrupto: {p}')
    return True
def rows_of(split):
    rows=partition(split,allow_test=split=='test')
    return rows[:LIMIT] if LIMIT else rows
def sync():
    if DEVICE=='cuda':torch.cuda.synchronize()
def infer(model,x):
    with torch.inference_mode():p=main_logits(model(torch.from_numpy(x.copy())[None].to(DEVICE))).sigmoid()[0,0].float().cpu().numpy()
    if not np.isfinite(p).all():raise RuntimeError('Probabilidades no finitas')
    return p
def atomic_npz(path,**arrays):
    temp=Path(str(path)+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)

_TEST_CACHE={}
def test_examples():
    """Test cargado una vez por proceso (218 x 3 x 256 x 256 float32, ~170 MB); lo reutilizan las 9 condiciones."""
    key=LIMIT
    if key not in _TEST_CACHE:
        _TEST_CACHE.clear();_TEST_CACHE[key]=[(row,*load_example(row)) for row in rows_of('test')]
    return _TEST_CACHE[key]

def profile(model,row):
    x,_,_=load_example(row);t=torch.from_numpy(x.copy())[None].to(DEVICE);cuda=DEVICE=='cuda'
    with torch.inference_mode():
        for _ in range(PROFILE_WARMUP):main_logits(model(t))
        sync();times=[]
        if cuda:torch.cuda.reset_peak_memory_stats()
        for _ in range(PROFILE_REPS):
            if cuda:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();main_logits(model(t));end.record();torch.cuda.synchronize();times.append(start.elapsed_time(end))
            else:
                t0=time.perf_counter();main_logits(model(t));times.append((time.perf_counter()-t0)*1000)
        peak=torch.cuda.max_memory_allocated() if cuda else None
        acts=[torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
        with torch.profiler.profile(activities=acts,with_flops=True) as prof:
            main_logits(model(t));sync()
    events=prof.key_averages()
    return {'latency_ms_median':float(np.median(times)),'latency_ms_p95':float(np.percentile(times,95)),
      'latency_ms_samples':times,'inference_peak_allocated_bytes':peak,'device':DEVICE,
      'parameters_total':sum(p.numel() for p in model.parameters()),
      'profile_counted_FLOPs':sum(e.flops for e in events),
      'operators_without_FLOP_count':sorted({e.key for e in events if not e.flops}),
      'FLOP_scope':'PyTorch profiler: conv/matmul contados; no total certificado de operadores custom/SSM/NMF',
      'timing_scope':f'batch1 FP32 256x256, dispositivo sincronizado, sin carga/transferencia/escritura; {PROFILE_WARMUP} warmup, {PROFILE_REPS} repeticiones'}

def clean(model,run,out,sig,cfg,norm):
    dest=out/(run['folder']+'_clean.json');probfile=out/(run['folder']+'_probabilities.npz')
    if valid(dest,sig):
        r=read(dest)
        if 'probabilities_sha256' in r and (not probfile.exists() or sha(probfile)!=r['probabilities_sha256']):raise RuntimeError('Probabilidades corruptas: '+probfile.name)
        return
    started=time.perf_counter()
    result={'model':run['model'],'seed':run['seed'],'checkpoint_sha256':run['best_sha256'],'splits':{},'seconds':{}}
    result['resources']=profile(model,rows_of('val')[0]);sample_z=[];sample_y=[];sample_w=[];saved={}
    keep=run['model'] in cfg['top3_by_validation']
    for split in ['train','val','test']:
        t0=time.perf_counter()
        rows=rows_of(split);counts=Counter(r['group_id'] for r in rows);details=[];probs=[]
        bins_raw=np.zeros((15,3));bins_cal=np.zeros((15,3))
        torch.manual_seed(20261118)
        for i,row in enumerate(rows):
            x,y,vm=load_example(row);p=infer(model,x)
            d={'id':row['id'],'group_id':row['group_id'],**diagnose(p,y,vm,with_border=split!='train')}
            weight=1/(len(counts)*counts[row['group_id']])
            bins_raw+=calibration_bins(p,y,vm)*weight
            if split=='val':
                ix=np.flatnonzero(vm);seed=int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)
                chosen=np.random.default_rng(seed).choice(ix,min(1024,len(ix)),replace=False)
                sample_z.append(logit(np.clip(p.ravel()[chosen],1e-7,1-1e-7)));sample_y.append(y.ravel()[chosen])
                sample_w.append(np.full(len(chosen),1/(counts[row['group_id']]*len(chosen))))
            if split=='test':
                pc=expit(logit(np.clip(p,1e-7,1-1e-7))/result['calibration']['temperature'])
                bins_cal+=calibration_bins(pc,y,vm)*weight
                calibrated=diagnose(pc,y,vm,with_border=False)
                d.update({k+'_calibrated':val for k,val in calibrated.items() if k not in ['Dice','IoU','mIoU','accuracy']})
            details.append(d)
            if keep and split!='train':probs.append(p.astype('float16'))
            if (i+1)%100==0 or i+1==len(rows):print(run['folder'],split,i+1,'/',len(rows),flush=True)
        result['splits'][split]={'details':details,'means':aggregate(details),'images':len(rows)}
        result['splits'][split]['reliability_bins_raw']=bins_raw.tolist()
        if split=='test':result['splits'][split]['reliability_bins_calibrated']=bins_cal.tolist()
        if probs:saved[split]=np.stack(probs);saved[split+'_ids']=np.array([r['id'] for r in rows])
        if split=='val':result['calibration']=fit_temperature(np.concatenate(sample_z),np.concatenate(sample_y),np.concatenate(sample_w))
        result['seconds'][split]=time.perf_counter()-t0
    result['gap_train_val_Dice']=result['splits']['train']['means']['Dice']-result['splits']['val']['means']['Dice']
    if saved:
        atomic_npz(probfile,**saved);result['probabilities_sha256']=sha(probfile)
    result['seconds']['total']=time.perf_counter()-started
    write(dest,result,sig)

def robust(model,run,out,sig,cfg,norm):
    conditions={'noise':cfg['robustness']['noise_sigma'],'blur':cfg['robustness']['gaussian_blur_sigma'],'brightness':cfg['robustness']['brightness_factor']}
    for kind,levels in conditions.items():
        for level in levels:
            dest=out/f'{run["folder"]}_{kind}_{level}.json'
            if valid(dest,sig):continue
            details=[];torch.manual_seed(20261118);t0=time.perf_counter();examples=test_examples()
            for i,(row,x,y,vm) in enumerate(examples):
                seed=int(hashlib.sha256(f'{row["id"]}:{kind}:{level}'.encode()).hexdigest()[:8],16)
                xp=perturb(x,vm,norm,kind,level,seed);p=infer(model,xp)
                details.append({'id':row['id'],'group_id':row['group_id'],**metrics(p,y,vm)})
                if (i+1)%100==0:print(run['folder'],kind,level,i+1,flush=True)
            write(dest,{'model':run['model'],'seed':run['seed'],'perturbation':kind,'level':level,'details':details,
                        'means':aggregate(details),'images':len(details),'seconds':time.perf_counter()-t0},sig)
            print(run['folder'],kind,level,'Dice',round(aggregate(details,['Dice'])['Dice'],4),flush=True)

def cases_for_interpret(out,sig,cfg):
    path=out/'casos_interpretabilidad.json'
    if valid(path,sig):return read(path)['ids']
    rows=partition('val');bygroup={}
    for row in rows:bygroup.setdefault(row['group_id'],row)
    candidates=[]
    for row in bygroup.values():
        _,y,vm=load_example(row);candidates.append((float(y[vm].mean()),row['id']))
    candidates.sort();rng=np.random.default_rng(20261120);ids=[];per=cfg['interpretability']['cases']//4
    for stratum in np.array_split(np.arange(len(candidates)),4):
        ids.extend(candidates[i][1] for i in rng.choice(stratum,size=per,replace=False))
    if LIMIT:ids=ids[::max(1,len(ids)//LIMIT)][:LIMIT]
    write(path,{'ids':ids,'selection':f'{len(ids)} grupos distintos de validación, estratificados por cuartil del área del rótulo; sin usar score/predicciones'},sig)
    return ids

def interpret(model,run,out,sig,cfg):
    if run['model'] not in cfg['top3_by_validation']:return
    dest=out/(run['folder']+'_interpret.json');mapsfile=out/(run['folder']+'_attributions.npz')
    if valid(dest,sig):
        if not mapsfile.exists() or sha(mapsfile)!=read(dest)['maps_sha256']:raise RuntimeError('Mapas corruptos: '+mapsfile.name)
        return
    selected=cases_for_interpret(out,sig,cfg);rows={r['id']:r for r in partition('val')};details=[];maps=[];t0=time.perf_counter()
    for p in model.parameters():p.requires_grad_(False)
    steps=cfg['interpretability']['steps']
    for j,key in enumerate(selected):
        row=rows[key];x,y,vm=load_example(row);t=torch.from_numpy(x.copy())[None].to(DEVICE);cleanp=infer(model,x)
        roi=(cleanp>=.5)&vm
        if not roi.any():roi=vm&(cleanp>=np.quantile(cleanp[vm],.99))
        rt=torch.from_numpy(roi.astype('float32')).to(DEVICE);denom=rt.sum()
        def score(a):return (main_logits(model(a))[0,0]*rt).sum()/denom
        gradient=torch.zeros_like(t)
        for alpha in (np.arange(steps)+.5)/steps:
            xi=(t*float(alpha)).detach().requires_grad_(True)
            gradient+=torch.autograd.grad(score(xi),xi)[0].detach()/steps
        attrib=t*gradient;importance=attrib.abs().sum(dim=1)[0].cpu().numpy();importance[~vm]=0
        with torch.no_grad():original=float(score(t));base=float(score(torch.zeros_like(t)))
        completeness=abs(float(attrib.sum())-(original-base))
        ix=np.flatnonzero(vm);ranked=ix[np.argsort(-importance.ravel()[ix],kind='stable')]
        random=np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8],16)).permutation(ix)
        fidelity=[]
        for fraction in cfg['interpretability']['fidelity_fractions']:
            n=max(1,int(len(ix)*fraction))
            for label,order in [('IG',ranked),('random',random)]:
                deletion=x.copy().reshape(3,-1);deletion[:,order[:n]]=0;deletion=deletion.reshape(x.shape)
                insertion=np.zeros_like(x).reshape(3,-1);insertion[:,order[:n]]=x.reshape(3,-1)[:,order[:n]];insertion=insertion.reshape(x.shape)
                with torch.no_grad():
                    ds=float(score(torch.from_numpy(deletion.copy())[None].to(DEVICE)));ins=float(score(torch.from_numpy(insertion.copy())[None].to(DEVICE)))
                fidelity.append({'fraction':fraction,'ordering':label,'deletion_logit_drop':original-ds,'insertion_logit_gain':ins-base})
        yb=y.astype(bool);pred=cleanp>=.5;tp=int((pred&yb&vm).sum());fp=int((pred&~yb&vm).sum());fn=int((~pred&yb&vm).sum())
        dice=metrics(cleanp,y,vm)['Dice']
        category='acuerdo_alto' if dice>=.9 else ('sin_solapamiento' if tp==0 else ('predominio_omision' if fn>2*fp else ('predominio_exceso' if fp>2*fn else 'error_mixto')))
        details.append({'id':key,'group_id':row['group_id'],'Dice':dice,'category_geometric':category,'completeness_abs_error':completeness,
                        'score_change_from_baseline':original-base,'roi_from_threshold':bool(((cleanp>=.5)&vm).any()),'fidelity':fidelity})
        maps.append((importance/max(float(importance.max()),1e-12)).astype('float16'))
        print(run['folder'],'IG',j+1,'/',len(selected),flush=True)
    atomic_npz(mapsfile,ids=np.array(selected),maps=np.stack(maps))
    write(dest,{'model':run['model'],'seed':run['seed'],'details':details,'maps_sha256':sha(mapsfile),'cases':len(selected),
      'seconds':time.perf_counter()-t0,
      'method':f'IG punto medio {steps} pasos, baseline RGB medio train (cero normalizado), score logit medio en ROI predicha fija; no explicación causal',
      'taxonomy':'Geométrica automática definida en complemento; no etiquetado semántico humano ni taxonomía anterior al benchmark'},sig)

STAGES={'clean':clean,'robustness':robust,'interpret':lambda m,r,o,s,c,n:interpret(m,r,o,s,c)}

def load_model(run,runs_dir):
    ck=torch.load(Path(runs_dir)/run['folder']/run['best_file'],map_location='cpu',weights_only=True)
    if ck['fingerprint']!=run['fingerprint']:raise RuntimeError('Fingerprint incorrecto: '+run['folder'])
    model=MODELS_E2[run['model']]().to(DEVICE);model.load_state_dict(ck['model'],strict=True);model.eval();del ck
    return model

def main():
    global LIMIT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=list(STAGES),required=True);p.add_argument('--models',nargs='*')
    p.add_argument('--seeds',nargs='*',type=int,help='Subconjunto de semillas (piloto)')
    p.add_argument('--limit',type=int,help='Imágenes por partición / casos IG (solo piloto; cambia la firma)')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);LIMIT=args.limit
    if not torch.cuda.is_available():raise RuntimeError('Se requiere GPU CUDA')
    protocol=read(PROJECT/'config/protocolo_test_e2_cerrado.json');cfg=read(PROJECT/'config/complementos_e2_20261120.json')
    t0=time.perf_counter();print('Verificando datos y 50 checkpoints...',flush=True);verify_inputs(protocol,Path(args.runs))
    for split in ['train','val']:
        for row in partition(split):
            if sha(PROJECT.parent/row['image_path'])!=row['image_sha256'] or sha(PROJECT/row['binary_mask_path'])!=row['binary_mask_sha256']:
                raise RuntimeError('Datos train/val alterados: '+row['id'])
    verification_seconds=time.perf_counter()-t0
    env=environment()
    if env['packages']!=protocol['training_environment']['packages']:
        raise RuntimeError(f"Software principal diferente al entrenamiento: {env['packages']}")
    signature=digest({'config':cfg,'closed_protocol':protocol,'environment':env,'script':sha(Path(__file__)),
                      'metrics':sha(PROJECT/'scripts/metricas_diagnostico_e2.py'),'limit':LIMIT})
    lock=out/'PROTOCOLO_DIAGNOSTICOS.json'
    if lock.exists():
        old=read(lock)
        if old['signature']!=signature:
            raise RuntimeError('No mezclar firmas en una carpeta. GPU registrada: %s; actual: %s. Si cambió la GPU, el '
                'paquete o --limit, usa la misma GPU o una carpeta de salida nueva.'%(old['environment'].get('device'),env['device']))
    else:atomic_json(lock,{'signature':signature,'config':cfg,'environment':env,'closed_protocol':protocol,'limit':LIMIT})
    norm=read(PROJECT/'resultados/preentrenamiento/normalization_train.json')
    runs=[r for r in protocol['runs'] if (not args.models or r['model'] in args.models) and (not args.seeds or r['seed'] in args.seeds)]
    log={'stage':args.stage,'limit':LIMIT,'device':env['device'],'verification_seconds':verification_seconds,'runs':[]}
    for run in runs:
        if args.stage=='interpret' and run['model'] not in cfg['top3_by_validation']:continue
        setup_seed(run['seed']);t1=time.perf_counter();model=load_model(run,args.runs);loaded=time.perf_counter()
        STAGES[args.stage](model,run,out,signature,cfg,norm);sync()
        log['runs'].append({'folder':run['folder'],'load_seconds':loaded-t1,'stage_seconds':time.perf_counter()-loaded})
        del model;gc.collect();torch.cuda.empty_cache()
    logs=out/'tiempos';logs.mkdir(exist_ok=True)
    atomic_json(logs/f'{args.stage}_{time.strftime("%Y%m%dT%H%M%S")}.json',log)
    print('Etapa finalizada:',args.stage,flush=True)

if __name__=='__main__':main()
