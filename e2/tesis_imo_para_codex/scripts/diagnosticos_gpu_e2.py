"""Complementos exploratorios sobre pesos cerrados: clean, robustez y explicaciones.
Reanudación por corrida/condición, firma y hashes. Ningún optimizador ni cambio de pesos.
"""
import argparse,gc,json,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scipy.special import expit,logit
from scipy.stats import spearmanr
from dataset_ready import partition,load_example,metrics
from modelos_e2 import MODELS_E2,main_logits
from experimento_e2_colab import PROJECT,sha,digest,atomic_json,environment,setup_seed
from evaluar_e2_cerrado import verify_inputs
from metricas_diagnostico_e2 import diagnose,fit_temperature,perturb,aggregate,label_set,calibration_bins

def read(p):return json.loads(p.read_text())
def write(p,r,signature):
    r={**r,'signature':signature};r['payload_sha256']=digest(r);atomic_json(p,r)
def valid(p,signature):
    if not p.exists():return False
    r=read(p);saved=r.pop('payload_sha256',None)
    if r.get('signature')!=signature or saved!=digest(r):raise RuntimeError(f'Incompatible/corrupto: {p}')
    return True
def infer(model,x):
    with torch.inference_mode():p=main_logits(model(torch.from_numpy(x.copy())[None].cuda())).sigmoid()[0,0].float().cpu().numpy()
    if not np.isfinite(p).all():raise RuntimeError('Probabilidades no finitas')
    return p
def atomic_npz(path,**arrays):
    temp=Path(str(path)+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)

def profile(model,row):
    x,_,_=load_example(row);t=torch.from_numpy(x.copy())[None].cuda()
    with torch.inference_mode():
        for _ in range(10):main_logits(model(t))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(50):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();main_logits(model(t));end.record();torch.cuda.synchronize();times.append(start.elapsed_time(end))
        peak=torch.cuda.max_memory_allocated()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True) as prof:
            main_logits(model(t));torch.cuda.synchronize()
    events=prof.key_averages()
    return {'latency_ms_median':float(np.median(times)),'latency_ms_p95':float(np.percentile(times,95)),
      'latency_ms_samples':times,'inference_peak_allocated_bytes':peak,
      'parameters_total':sum(p.numel() for p in model.parameters()),
      'profile_counted_FLOPs':sum(e.flops for e in events),
      'operators_without_FLOP_count':[e.key for e in events if not e.flops],
      'FLOP_scope':'PyTorch profiler: conv/matmul contados; no total certificado de operadores custom/SSM/NMF',
      'timing_scope':'batch1 FP32 256x256, GPU sincronizada, sin carga/transferencia/escritura; 10 warmup, 50 repeticiones'}

def clean(model,run,out,sig,cfg,norm):
    dest=out/(run['folder']+'_clean.json');probfile=out/(run['folder']+'_probabilities.npz')
    if valid(dest,sig):
        r=read(dest)
        if 'probabilities_sha256' in r and (not probfile.exists() or sha(probfile)!=r['probabilities_sha256']):raise RuntimeError('Probabilidades corruptas')
        return
    result={'model':run['model'],'seed':run['seed'],'checkpoint_sha256':run['best_sha256'],'splits':{}}
    result['resources']=profile(model,partition('val')[0]);sample_z=[];sample_y=[];sample_w=[];saved={}
    for split in ['train','val','test']:
        rows=partition(split,allow_test=split=='test');counts=Counter(r['group_id'] for r in rows);details=[];probs=[]
        bins_raw=np.zeros((15,3));bins_cal=np.zeros((15,3))
        torch.manual_seed(20261118)
        for i,row in enumerate(rows):
            x,y,v=load_example(row);p=infer(model,x)
            if not np.isfinite(p).all():raise RuntimeError('Probabilidad no finita')
            d={'id':row['id'],'group_id':row['group_id'],**diagnose(p,y,v,with_border=split!='train')}
            bins_raw+=calibration_bins(p,y,v)/(len(counts)*counts[row['group_id']])
            if split=='val':
                ix=np.flatnonzero(v);seed=int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)
                chosen=np.random.default_rng(seed).choice(ix,min(1024,len(ix)),replace=False)
                sample_z.append(logit(np.clip(p.ravel()[chosen],1e-7,1-1e-7)));sample_y.append(y.ravel()[chosen])
                sample_w.append(np.full(len(chosen),1/(counts[row['group_id']]*len(chosen))))
            if split=='test':
                pc=expit(logit(np.clip(p,1e-7,1-1e-7))/result['calibration']['temperature'])
                bins_cal+=calibration_bins(pc,y,v)/(len(counts)*counts[row['group_id']])
                d.update({k+'_calibrated':v for k,v in diagnose(pc,y,v,with_border=False).items() if k not in ['Dice','IoU','mIoU','accuracy']})
            details.append(d)
            if run['model'] in cfg['top3_by_validation'] and split!='train':probs.append(p.astype('float16'))
            if (i+1)%100==0 or i+1==len(rows):print(run['folder'],split,i+1,'/',len(rows),flush=True)
        result['splits'][split]={'details':details,'means':aggregate(details)}
        result['splits'][split]['reliability_bins_raw']=bins_raw.tolist()
        if split=='test':result['splits'][split]['reliability_bins_calibrated']=bins_cal.tolist()
        if probs:saved[split]=np.stack(probs);saved[split+'_ids']=np.array([r['id'] for r in rows])
        if split=='val':result['calibration']=fit_temperature(np.concatenate(sample_z),np.concatenate(sample_y),np.concatenate(sample_w))
    result['gap_train_val_Dice']=result['splits']['train']['means']['Dice']-result['splits']['val']['means']['Dice']
    if saved:
        atomic_npz(probfile,**saved);result['probabilities_sha256']=sha(probfile)
    write(dest,result,sig)

def robust(model,run,out,sig,cfg,norm):
    conditions={'noise':cfg['robustness']['noise_sigma'],'blur':cfg['robustness']['gaussian_blur_sigma'],'brightness':cfg['robustness']['brightness_factor']}
    for kind,levels in conditions.items():
        for level in levels:
            dest=out/f'{run["folder"]}_{kind}_{level}.json'
            if valid(dest,sig):continue
            details=[];torch.manual_seed(20261118)
            for i,row in enumerate(partition('test',allow_test=True)):
                x,y,v=load_example(row);seed=int(hashlib.sha256(f'{row["id"]}:{kind}:{level}'.encode()).hexdigest()[:8],16)
                xp=perturb(x,v,norm,kind,level,seed);p=infer(model,xp)
                details.append({'id':row['id'],'group_id':row['group_id'],**metrics(p,y,v)})
                if (i+1)%100==0:print(run['folder'],kind,level,i+1,flush=True)
            write(dest,{'model':run['model'],'seed':run['seed'],'perturbation':kind,'level':level,'details':details,'means':aggregate(details)},sig)

def cases_for_interpret(out,sig,cfg):
    path=out/'casos_interpretabilidad.json'
    if valid(path,sig):return read(path)['ids']
    rows=partition('val');bygroup={}
    for row in rows:bygroup.setdefault(row['group_id'],row)
    candidates=[]
    for row in bygroup.values():
        _,y,v=load_example(row);candidates.append((float(y[v].mean()),row['id']))
    candidates.sort();rng=np.random.default_rng(20261120);ids=[]
    for stratum in np.array_split(np.arange(len(candidates)),4):
        ids.extend(candidates[i][1] for i in rng.choice(stratum,size=25,replace=False))
    write(path,{'ids':ids,'selection':'100 grupos distintos de validación, 25 casos por cuartil del área de candidatos; sin usar score/predicciones'},sig)
    return ids

def interpret(model,run,out,sig,cfg):
    if run['model'] not in cfg['top3_by_validation']:return
    dest=out/(run['folder']+'_interpret.json');mapsfile=out/(run['folder']+'_attributions.npz')
    if valid(dest,sig):
        if sha(mapsfile)!=read(dest)['maps_sha256']:raise RuntimeError('Mapas corruptos')
        return
    selected=cases_for_interpret(out,sig,cfg);rows={r['id']:r for r in partition('val')};details=[];maps=[]
    for p in model.parameters():p.requires_grad_(False)
    for j,key in enumerate(selected):
        row=rows[key];x,y,v=load_example(row);t=torch.from_numpy(x.copy())[None].cuda();cleanp=infer(model,x)
        roi=(cleanp>=.5)&v
        if not roi.any():roi=v&(cleanp>=np.quantile(cleanp[v],.99))
        rt=torch.from_numpy(roi.astype('float32')).cuda();denom=rt.sum()
        def score(a):return (main_logits(model(a))[0,0]*rt).sum()/denom
        gradient=torch.zeros_like(t);steps=cfg['interpretability']['steps']
        for alpha in (np.arange(steps)+.5)/steps:
            xi=(t*float(alpha)).detach().requires_grad_(True)
            gradient+=torch.autograd.grad(score(xi),xi)[0].detach()/steps
        attrib=t*gradient;importance=attrib.abs().sum(dim=1)[0].cpu().numpy();importance[~v]=0
        with torch.no_grad():original=float(score(t));base=float(score(torch.zeros_like(t)))
        completeness=abs(float(attrib.sum())-(original-base))
        ix=np.flatnonzero(v);ranked=ix[np.argsort(-importance.ravel()[ix])]
        random=np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8],16)).permutation(ix)
        fidelity=[]
        for fraction in cfg['interpretability']['fidelity_fractions']:
            n=max(1,int(len(ix)*fraction))
            for label,order in [('IG',ranked),('random',random)]:
                deletion=x.copy().reshape(3,-1);deletion[:,order[:n]]=0;deletion=deletion.reshape(x.shape)
                insertion=np.zeros_like(x).reshape(3,-1);insertion[:,order[:n]]=x.reshape(3,-1)[:,order[:n]];insertion=insertion.reshape(x.shape)
                with torch.no_grad():
                    ds=float(score(torch.from_numpy(deletion.copy())[None].cuda()));ins=float(score(torch.from_numpy(insertion.copy())[None].cuda()))
                fidelity.append({'fraction':fraction,'ordering':label,'deletion_logit_drop':original-ds,'insertion_logit_gain':ins-base})
        pred=cleanp>=.5;tp=int((pred&y.astype(bool)&v).sum());fp=int((pred&~y.astype(bool)&v).sum());fn=int((~pred&y.astype(bool)&v).sum())
        dice=metrics(cleanp,y,v)['Dice']
        category='acuerdo_alto' if dice>=.9 else ('sin_solapamiento' if tp==0 else ('predominio_omision' if fn>2*fp else ('predominio_exceso' if fp>2*fn else 'error_mixto')))
        details.append({'id':key,'group_id':row['group_id'],'Dice':dice,'category_geometric':category,'completeness_abs_error':completeness,'score_change_from_baseline':original-base,'fidelity':fidelity})
        maps.append((importance/max(float(importance.max()),1e-12)).astype('float16'))
        print(run['folder'],'IG',j+1,'/100',flush=True)
    atomic_npz(mapsfile,ids=np.array(selected),maps=np.stack(maps))
    write(dest,{'model':run['model'],'seed':run['seed'],'details':details,'maps_sha256':sha(mapsfile),
      'method':'IG midpoint 16 pasos, baseline RGB medio train, score logit medio en ROI predicha fija; no explicación causal',
      'taxonomy':'Geométrica automática definida en complemento; no etiquetado semántico humano ni taxonomía anterior al benchmark'},sig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['clean','robustness','interpret'],required=True);p.add_argument('--models',nargs='*')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('Se requiere GPU CUDA')
    protocol=read(PROJECT/'config/protocolo_test_e2_cerrado.json');cfg=read(PROJECT/'config/complementos_e2_20261120.json')
    print('Verificando datos y 50 checkpoints...',flush=True);verify_inputs(protocol,Path(args.runs))
    for split in ['train','val']:
        for row in partition(split):
            if sha(PROJECT.parent/row['image_path'])!=row['image_sha256'] or sha(PROJECT/row['binary_mask_path'])!=row['binary_mask_sha256']:
                raise RuntimeError('Datos train/val alterados: '+row['id'])
    env=environment()
    if env['packages']!=protocol['training_environment']['packages']:raise RuntimeError('Software principal diferente al entrenamiento')
    signature=digest({'config':cfg,'closed_protocol':protocol,'environment':env,'script':sha(Path(__file__)),'metrics':sha(PROJECT/'scripts/metricas_diagnostico_e2.py')})
    lock=out/'PROTOCOLO_DIAGNOSTICOS.json'
    if lock.exists() and read(lock)['signature']!=signature:raise RuntimeError('No mezclar firmas/GPU en una carpeta')
    atomic_json(lock,{'signature':signature,'config':cfg,'environment':env,'closed_protocol':protocol})
    norm=read(PROJECT/'resultados/preentrenamiento/normalization_train.json')
    runs=[r for r in protocol['runs'] if not args.models or r['model'] in args.models]
    for run in runs:
        if args.stage=='interpret' and run['model'] not in cfg['top3_by_validation']:continue
        setup_seed(run['seed']);ck=torch.load(Path(args.runs)/run['folder']/run['best_file'],map_location='cpu',weights_only=True)
        if ck['fingerprint']!=run['fingerprint']:raise RuntimeError('Fingerprint incorrecto')
        model=MODELS_E2[run['model']]().cuda();model.load_state_dict(ck['model'],strict=True);model.eval();del ck
        {'clean':clean,'robustness':robust,'interpret':lambda m,r,o,s,c,n:interpret(m,r,o,s,c)}[args.stage](model,run,out,signature,cfg,norm)
        del model;gc.collect();torch.cuda.empty_cache()
    print('Etapa finalizada:',args.stage,flush=True)

if __name__=='__main__':main()
