"""Resumen reejecutable de diagnósticos, ensembles e interpretación sin nuevo ajuste."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.special import expit,logit
from scipy.stats import spearmanr
from dataset_ready import partition,load_example,metrics
from metricas_diagnostico_e2 import aggregate,diagnose,label_set
from experimento_e2_colab import PROJECT,atomic_json
from diagnosticos_gpu_e2 import valid,read

def grouped(rows,key):
    return {g:float(np.mean([r[key] for r in rows if r['group_id']==g])) for g in sorted({r['group_id'] for r in rows})}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args();out=Path(args.output)
    lock=read(out/'PROTOCOLO_DIAGNOSTICOS.json');cfg=lock['config'];sig=lock['signature'];protocol=lock['closed_protocol']
    rng=np.random.default_rng(20261120);models=list(dict.fromkeys(r['model'] for r in protocol['runs']));clean={};missing=[]
    for m in models:
        clean[m]=[]
        for seed in cfg['seeds']:
            path=out/f'{m}_seed{seed}_clean.json'
            if valid(path,sig):clean[m].append(read(path))
            else:missing.append(path.name)
    result={'signature':sig,'status':'partial' if missing else 'clean_complete','missing':missing,'clean':[],'robustness':[],'ensembles':[],'interpretability':[]}
    for m in models:
        if len(clean[m])!=5:continue
        runs=clean[m];entry={'model':m,'test':{},'train':{},'val':{},'resources':{},'temperature':[r['calibration']['temperature'] for r in runs]}
        for split in ['train','val','test']:
            keys=list(runs[0]['splits'][split]['means'])
            for key in keys:
                vals=np.array([list(grouped(r['splits'][split]['details'],key).values()) for r in runs])
                b=vals.mean(axis=0)[rng.integers(vals.shape[1],size=(5000,vals.shape[1]))].mean(axis=1)
                entry[split][key]={'mean':float(vals.mean()),'sd_seeds':float(vals.mean(axis=1).std(ddof=1)),'CI95_groups_fixed_seeds':np.quantile(b,[.025,.975]).tolist()}
        for key in ['latency_ms_median','latency_ms_p95','inference_peak_allocated_bytes','parameters_total','profile_counted_FLOPs']:
            entry['resources'][key]=float(np.mean([r['resources'][key] for r in runs]))
        entry['reliability_bins_raw']=np.mean([r['splits']['test']['reliability_bins_raw'] for r in runs],axis=0).tolist()
        entry['reliability_bins_calibrated']=np.mean([r['splits']['test']['reliability_bins_calibrated'] for r in runs],axis=0).tolist()
        train=np.array([list(grouped(r['splits']['train']['details'],'Dice').values()) for r in runs]).mean(axis=0)
        val=np.array([list(grouped(r['splits']['val']['details'],'Dice').values()) for r in runs]).mean(axis=0)
        gap=train[rng.integers(len(train),size=(5000,len(train)))].mean(axis=1)-val[rng.integers(len(val),size=(5000,len(val)))].mean(axis=1)
        ci=np.quantile(gap,[.025,.975]);entry['gap_train_val']={'difference':float(train.mean()-val.mean()),'CI95':ci.tolist(),
             'margin_exploratory':cfg['gap_margin_exploratory'],'upper_below_margin':bool(ci[1]<cfg['gap_margin_exploratory']),
             'scope':'Bootstrap independiente de grupos train y val, semillas fijas; mismo checkpoint eval. Diagnóstico exploratorio, val seleccionado.'}
        result['clean'].append(entry)
        for kind,levels in [('noise',cfg['robustness']['noise_sigma']),('blur',cfg['robustness']['gaussian_blur_sigma']),('brightness',cfg['robustness']['brightness_factor'])]:
            points=[];ready=True
            for level in levels:
                vals=[]
                for seed in cfg['seeds']:
                    path=out/f'{m}_seed{seed}_{kind}_{level}.json'
                    if not valid(path,sig):result['missing'].append(path.name);ready=False;continue
                    vals.append(list(grouped(read(path)['details'],'Dice').values()))
                if len(vals)==5:points.append(np.array(vals))
            if ready:
                baseline=np.array([list(grouped(r['splits']['test']['details'],'Dice').values()) for r in runs]);curve=np.stack([baseline,*points])
                # Areas of mean scores, normalized by clean mean, ordinal severity 0..3.
                ix=rng.integers(curve.shape[2],size=(5000,curve.shape[2]));bm=curve.mean(axis=1)[:,ix].mean(axis=2)
                ratio=np.trapz(bm,axis=0)/(3*np.maximum(bm[0],1e-9))
                result['robustness'].append({'model':m,'kind':kind,'levels':levels,'Dice_curve':curve.mean(axis=(1,2)).tolist(),
                     'normalized_area_ordinal':float(np.trapz(curve.mean(axis=(1,2)))/(3*baseline.mean())),
                     'CI95_groups':np.quantile(ratio,[.025,.975]).tolist(),'definition':'trapecios en severidad ordinal 0,1,2,3; normalizar por 3*Dice limpio'})
    # Ensembles are only computed for the three networks chosen by validation.
    rows=partition('test',allow_test=True)
    for m in cfg['top3_by_validation']:
        if len(clean[m])!=5:continue
        paths=[out/f'{m}_seed{s}_probabilities.npz' for s in cfg['seeds']]
        if not all(p.exists() for p in paths):result['missing'].extend(p.name for p in paths if not p.exists());continue
        loaded=[np.load(p) for p in paths]
        expected=[r['id'] for r in rows]
        if any(list(z['test_ids'])!=expected for z in loaded):raise RuntimeError('Orden de probabilidades diferente')
        maps=[z['test'] for z in loaded];temperatures=[r['calibration']['temperature'] for r in clean[m]];details=[]
        for i,row in enumerate(rows):
            _,y,v=load_example(row);ps=np.stack([a[i].astype('float32') for a in maps]);p=ps.mean(axis=0)
            pc=np.mean([expit(logit(np.clip(a,1e-7,1-1e-7))/t) for a,t in zip(ps,temperatures)],axis=0)
            entropy=-(np.clip(pc,1e-7,1-1e-7)*np.log(np.clip(pc,1e-7,1))+(1-np.clip(pc,1e-7,1-1e-7))*np.log(np.clip(1-pc,1e-7,1)))
            details.append({'id':row['id'],'group_id':row['group_id'],**diagnose(p,y,v),
                **label_set(pc,y,v,cfg['uncertainty']['nominal_set_mass']),
                'mean_entropy_calibrated':float(entropy[v].mean()),'mean_variance_between_seeds':float(ps.var(axis=0,ddof=1)[v].mean())})
        report={'model':m,'means':aggregate(details),'details':details,'nominal_mass':.9,
                'limits':'Probabilidades almacenadas float16. Conjuntos de masa mínima: cobertura empírica por píxel, no garantía conforme ni cobertura simultánea de imagen. Temperaturas ajustadas en val ya usado para checkpoint.'}
        for key in ['coverage','mean_set_size','Dice']:
            vals=np.array(list(grouped(details,key).values()));b=vals[rng.integers(len(vals),size=(5000,len(vals)))].mean(axis=1)
            report[key+'_CI95_groups']=np.quantile(b,[.025,.975]).tolist()
        baseline=np.mean([list(grouped(r['splits']['test']['details'],'Dice').values()) for r in clean[m]],axis=0)
        diff=np.array(list(grouped(details,'Dice').values()))-baseline
        report['gain_over_mean_member']={'difference':float(diff.mean()),'CI95_groups':np.quantile(diff[rng.integers(len(diff),size=(5000,len(diff)))].mean(axis=1),[.025,.975]).tolist()}
        result['ensembles'].append(report)
        for z in loaded:z.close()
        del maps,loaded
        ipaths=[out/f'{m}_seed{s}_interpret.json' for s in cfg['seeds']]
        if not all(valid(p,sig) for p in ipaths):result['missing'].extend(p.name for p in ipaths if not p.exists());continue
        ir=[read(p) for p in ipaths];imap=[np.load(out/f'{m}_seed{s}_attributions.npz')['maps'] for s in cfg['seeds']]
        stable=[];undefined=0
        for i in range(100):
            row=next(r for r in partition('val') if r['id']==ir[0]['details'][i]['id']);_,_,v=load_example(row)
            corr=[]
            for j in range(5):
                for k in range(j):
                    aa=imap[j][i][v];bb=imap[k][i][v]
                    if np.ptp(aa)==0 or np.ptp(bb)==0:undefined+=1;continue
                    corr.append(float(spearmanr(aa,bb).statistic))
            if corr:stable.append(float(np.mean(corr)))
        fidelity={}
        for frac in cfg['interpretability']['fidelity_fractions']:
            ds=[];ins=[]
            for r in ir:
                for d in r['details']:
                    pair={f['ordering']:f for f in d['fidelity'] if f['fraction']==frac}
                    ds.append(pair['IG']['deletion_logit_drop']-pair['random']['deletion_logit_drop'])
                    ins.append(pair['IG']['insertion_logit_gain']-pair['random']['insertion_logit_gain'])
            fidelity[str(frac)]={'deletion_advantage_over_random_mean':float(np.mean(ds)),'insertion_advantage_over_random_mean':float(np.mean(ins))}
        result['interpretability'].append({'model':m,'cases_per_seed':100,'seeds':5,'mean_pairwise_spearman':float(np.mean(stable)) if stable else None,
          'undefined_constant_map_pairs':undefined,'fidelity':fidelity,
          'max_IG_completeness_abs_error':max(d['completeness_abs_error'] for r in ir for d in r['details']),
          'category_counts_all_500_not_independent':dict(Counter(d['category_geometric'] for r in ir for d in r['details'])),
          'scope':'Validación, 100 grupos estratificados, 5 semillas correlacionadas; no explicación causal ni frecuencias poblacionales.'})
    result['status']='complete' if not result['missing'] and len(result['clean'])==10 and len(result['robustness'])==30 and len(result['interpretability'])==3 else 'partial'
    atomic_json(out/'RESUMEN_DIAGNOSTICOS.json',result)
    print('Estado:',result['status'],'Faltantes:',len(result['missing']),flush=True)
    for r in result['clean']:print(r['model'],'Dice',r['test']['Dice']['mean'],'Brier',r['test']['Brier']['mean'],'gap',r['gap_train_val']['difference'],flush=True)

if __name__=='__main__':main()
