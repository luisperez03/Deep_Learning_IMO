"""Resumen reejecutable de diagnósticos, ensembles e interpretación sin nuevo ajuste de los modelos.

Lee la carpeta de diagnosticos_gpu_e2.py, verifica firma y hashes (JSON y NPZ) y escribe RESUMEN_DIAGNOSTICOS.json.
La única cantidad calibrada aquí es el umbral conforme por clase, y se calibra en validación, nunca en test.
"""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.special import expit,logit
from scipy.stats import spearmanr
from dataset_ready import partition,load_example
from metricas_diagnostico_e2 import (aggregate,diagnose,label_set,conformal_scores_by_class,conformal_calibrate,
                                     conformal_sets)
from experimento_e2_colab import PROJECT,atomic_json,sha
from diagnosticos_gpu_e2 import valid,read

trapezoid=getattr(np,'trapezoid',None) or np.trapz
B=5000

def grouped(rows,key):
    return {g:float(np.mean([r[key] for r in rows if r['group_id']==g])) for g in sorted({r['group_id'] for r in rows})}

def boot_ci(values,rng):
    """IC95 percentil de la media; values: un valor por grupo (o caso). None si hay menos de 2."""
    v=np.asarray([x for x in values if x is not None and np.isfinite(x)],dtype='float64')
    if len(v)<2:return None
    b=v[rng.integers(len(v),size=(B,len(v)))].mean(axis=1)
    return np.quantile(b,[.025,.975]).tolist()

def checked_npz(path,expected):
    if not path.exists():return None
    if sha(path)!=expected:raise RuntimeError('NPZ alterado respecto al JSON firmado: '+path.name)
    return np.load(path)

def conformal_report(val_rows,test_rows,val_maps,test_maps,cfg,rng):
    """Mondrian por clase con la probabilidad media del ensamble SIN temperatura (la temperatura se ajustó en val:
    usarla haría que la función de puntaje dependiera de los datos de calibración)."""
    c=cfg['conformal'];store={0:{},1:{}}
    for i,row in enumerate(val_rows):
        _,y,vm=load_example(row);conformal_scores_by_class(val_maps[i],y,vm,row['group_id'],store)
    q,n=conformal_calibrate(store,c['alpha'],c['pixels_per_group_and_class'],c['seed'])
    area={};per_image=[]
    for i,row in enumerate(test_rows):
        _,y,vm=load_example(row);area.setdefault(row['group_id'],[]).append(float(y[vm].mean()))
        per_image.append((row['group_id'],conformal_sets(test_maps[i],y,vm,q['label'],q['background']),
                          conformal_sets(test_maps[i],y,vm,q['marginal'],q['marginal'])))
    area={g:float(np.mean(a)) for g,a in area.items()}
    cuts=np.quantile(list(area.values()),[1/3,2/3])
    def tally(which):
        acc={};groups={}
        for g,mond,marg in per_image:
            s=mond if which=='mondrian' else marg
            tercile=f'area_tercile_{int(np.digitize(area[g],cuts))+1}'
            for cls in ('all','label','background'):
                for key in (cls,f'{tercile}_{cls}'):
                    a=acc.setdefault(key,[0,0,0,0,0])
                    for j,f in enumerate(['covered','pixels','size_sum','empty','both']):a[j]+=s[cls][f]
                gg=groups.setdefault(g,{}).setdefault(cls,[0,0]);gg[0]+=s[cls]['covered'];gg[1]+=s[cls]['pixels']
        res={k:{'coverage':a[0]/a[1],'mean_set_size':a[2]/a[1],'empty_fraction':a[3]/a[1],'ambiguous_fraction':a[4]/a[1],'pixels':a[1]}
             for k,a in acc.items() if a[1]}
        for cls in ('all','label','background'):
            cov=[v[cls][0]/v[cls][1] for v in groups.values() if cls in v and v[cls][1]]
            res[cls]['by_group']={'groups':len(cov),'p5':float(np.percentile(cov,5)),'median':float(np.median(cov)),
                'fraction_below_nominal':float(np.mean(np.array(cov)<1-c['alpha'])),
                'mean_of_groups':float(np.mean(cov)),'CI95_mean_of_groups':boot_ci(cov,rng)}
        return res
    return {'alpha':c['alpha'],'nominal_coverage':1-c['alpha'],'q_hat':q,'calibration_pixels':n,
            'score':'s = 1 - p(clase verdadera), p = media de las 5 semillas sin temperatura',
            'mondrian_by_class':tally('mondrian'),'marginal_single_threshold_reference':tally('marginal'),
            'limits':('Calibrado en validación, que también eligió los checkpoints (parada temprana) y los top 3: la '
                      'intercambiabilidad val-test es aproximada. Los píxeles de una imagen no son intercambiables, así que la '
                      'garantía marginal por clase es aproximada; la distribución por grupo muestra cuánto se aparta. '
                      'No es cobertura simultánea de imagen.')}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);args=ap.parse_args();out=Path(args.output)
    lock=read(out/'PROTOCOLO_DIAGNOSTICOS.json');cfg=lock['config'];sig=lock['signature'];protocol=lock['closed_protocol'];limit=lock.get('limit')
    rng=np.random.default_rng(20261120);models=list(dict.fromkeys(r['model'] for r in protocol['runs']));clean={};missing=[]
    def lim(rows):return rows[:limit] if limit else rows
    test_rows=lim(partition('test',allow_test=True));val_rows=lim(partition('val'));val_by_id={r['id']:r for r in partition('val')}
    for m in models:
        clean[m]=[]
        for seed in cfg['seeds']:
            path=out/f'{m}_seed{seed}_clean.json'
            if valid(path,sig):clean[m].append(read(path))
            else:missing.append(path.name)
    result={'signature':sig,'limit':limit,'status':'partial' if missing else 'clean_complete','missing':missing,
            'clean':[],'robustness':[],'ensembles':[],'interpretability':[],
            'scope':'Exploratorio, posterior a la exposición del test. Ningún umbral, modelo ni checkpoint se cambia con estos resultados.'}
    for m in models:
        if len(clean[m])!=5:continue
        runs=clean[m];entry={'model':m,'test':{},'train':{},'val':{},'resources':{},'temperature':[r['calibration']['temperature'] for r in runs],
                             'temperature_at_bound':[r['calibration']['at_bound'] for r in runs]}
        for split in ['train','val','test']:
            for key in runs[0]['splits'][split]['means']:
                vals=np.array([list(grouped(r['splits'][split]['details'],key).values()) for r in runs])
                entry[split][key]={'mean':float(vals.mean()),'sd_seeds':float(vals.mean(axis=1).std(ddof=1)),
                                   'CI95_groups_fixed_seeds':boot_ci(vals.mean(axis=0),rng)}
        for key in ['latency_ms_median','latency_ms_p95','inference_peak_allocated_bytes','parameters_total','profile_counted_FLOPs']:
            xs=[r['resources'][key] for r in runs];entry['resources'][key]=None if None in xs else float(np.mean(xs))
        entry['resources']['device']=runs[0]['resources'].get('device')
        entry['reliability_bins_raw']=np.mean([r['splits']['test']['reliability_bins_raw'] for r in runs],axis=0).tolist()
        entry['reliability_bins_calibrated']=np.mean([r['splits']['test']['reliability_bins_calibrated'] for r in runs],axis=0).tolist()
        train=np.array([list(grouped(r['splits']['train']['details'],'Dice').values()) for r in runs]).mean(axis=0)
        val=np.array([list(grouped(r['splits']['val']['details'],'Dice').values()) for r in runs]).mean(axis=0)
        gap=train[rng.integers(len(train),size=(B,len(train)))].mean(axis=1)-val[rng.integers(len(val),size=(B,len(val)))].mean(axis=1)
        ci=np.quantile(gap,[.025,.975]);entry['gap_train_val']={'difference':float(train.mean()-val.mean()),'CI95':ci.tolist(),
             'margin_exploratory':cfg['gap_margin_exploratory'],'upper_below_margin':bool(ci[1]<cfg['gap_margin_exploratory']),
             'scope':'Bootstrap independiente de grupos train y val, semillas fijas; mismo checkpoint en modo evaluación. Diagnóstico exploratorio; val ya se usó para elegir checkpoints.'}
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
                # Área bajo la curva de Dice medio en severidad ordinal 0..3, normalizada por 3 x Dice limpio.
                ix=rng.integers(curve.shape[2],size=(B,curve.shape[2]));bm=curve.mean(axis=1)[:,ix].mean(axis=2)
                ratio=trapezoid(bm,axis=0)/(3*np.maximum(bm[0],1e-9))
                drop=[{'level':lv,'Dice_drop':float(baseline.mean()-pt.mean()),'CI95_groups':boot_ci(baseline.mean(axis=0)-pt.mean(axis=0),rng)}
                      for lv,pt in zip(levels,points)]
                result['robustness'].append({'model':m,'kind':kind,'levels':levels,'Dice_curve':curve.mean(axis=(1,2)).tolist(),
                     'normalized_area_ordinal':float(trapezoid(curve.mean(axis=(1,2)))/(3*baseline.mean())),
                     'CI95_groups':np.quantile(ratio,[.025,.975]).tolist(),'drop_by_level':drop,
                     'definition':'trapecios en severidad ordinal 0,1,2,3; normalizar por 3*Dice limpio'})
    # Ensembles, incertidumbre e interpretación: solo las tres redes elegidas por validación.
    for m in cfg['top3_by_validation']:
        if len(clean[m])!=5:continue
        loaded=[checked_npz(out/f'{m}_seed{r["seed"]}_probabilities.npz',r['probabilities_sha256']) for r in clean[m]]
        if any(z is None for z in loaded):result['missing'].extend(f'{m}_seed{r["seed"]}_probabilities.npz' for r,z in zip(clean[m],loaded) if z is None);continue
        if any(list(z['test_ids'])!=[r['id'] for r in test_rows] or list(z['val_ids'])!=[r['id'] for r in val_rows] for z in loaded):
            raise RuntimeError('Orden de probabilidades diferente')
        maps=[z['test'] for z in loaded];vmaps=[z['val'] for z in loaded];temperatures=[r['calibration']['temperature'] for r in clean[m]];details=[]
        ens_test=[];ens_val=[np.mean([a[i].astype('float32') for a in vmaps],axis=0) for i in range(len(val_rows))]
        for i,row in enumerate(test_rows):
            _,y,vm=load_example(row);ps=np.stack([a[i].astype('float32') for a in maps]);p=ps.mean(axis=0);ens_test.append(p)
            pc=np.mean([expit(logit(np.clip(a,1e-7,1-1e-7))/t) for a,t in zip(ps,temperatures)],axis=0)
            q=np.clip(pc,1e-7,1-1e-7);entropy=-(q*np.log(q)+(1-q)*np.log(1-q))
            details.append({'id':row['id'],'group_id':row['group_id'],**diagnose(p,y,vm),
                **label_set(pc,y,vm,cfg['uncertainty']['nominal_set_mass']),
                'mean_entropy_calibrated':float(entropy[vm].mean()),'mean_variance_between_seeds':float(ps.var(axis=0,ddof=1)[vm].mean())})
        report={'model':m,'means':aggregate(details),'details':details,'nominal_mass':cfg['uncertainty']['nominal_set_mass'],
                'limits':'Probabilidades almacenadas float16. Conjuntos de masa mínima: cobertura empírica por píxel, sin garantía; la garantía (aproximada) está en conformal. Temperaturas ajustadas en val ya usado para checkpoint.'}
        for key in ['coverage','mean_set_size','Dice','ECE15','Brier']:
            report[key+'_CI95_groups']=boot_ci(list(grouped(details,key).values()),rng)
        baseline=np.mean([list(grouped(r['splits']['test']['details'],'Dice').values()) for r in clean[m]],axis=0)
        diff=np.array(list(grouped(details,'Dice').values()))-baseline
        report['gain_over_mean_member']={'difference':float(diff.mean()),'CI95_groups':boot_ci(diff,rng)}
        report['conformal']=conformal_report(val_rows,test_rows,ens_val,ens_test,cfg,rng)
        result['ensembles'].append(report)
        for z in loaded:z.close()
        del maps,vmaps,loaded,ens_test,ens_val
        ipaths=[out/f'{m}_seed{s}_interpret.json' for s in cfg['seeds']]
        if not all(valid(p,sig) for p in ipaths):result['missing'].extend(p.name for p in ipaths if not p.exists());continue
        ir=[read(p) for p in ipaths]
        zs=[checked_npz(out/f'{m}_seed{s}_attributions.npz',r['maps_sha256']) for s,r in zip(cfg['seeds'],ir)]
        if any(z is None for z in zs):raise RuntimeError('Faltan mapas de atribución firmados')
        ids=[d['id'] for d in ir[0]['details']]
        if any([d['id'] for d in r['details']]!=ids or list(z['ids'])!=ids for r,z in zip(ir,zs)):raise RuntimeError('Casos IG distintos entre semillas')
        imap=[z['maps'] for z in zs];stable=[];undefined=0
        for i,key in enumerate(ids):
            _,_,vm=load_example(val_by_id[key]);corr=[]
            for j in range(5):
                for k in range(j):
                    aa=imap[j][i][vm].astype('float32');bb=imap[k][i][vm].astype('float32')
                    if np.ptp(aa)==0 or np.ptp(bb)==0:undefined+=1;continue
                    corr.append(float(spearmanr(aa,bb).statistic))
            stable.append(float(np.mean(corr)) if corr else None)
        fidelity={}
        for frac in cfg['interpretability']['fidelity_fractions']:
            ds=[];ins=[]
            for i in range(len(ids)):
                dcase=[];icase=[]
                for r in ir:
                    pair={f['ordering']:f for f in r['details'][i]['fidelity'] if f['fraction']==frac}
                    dcase.append(pair['IG']['deletion_logit_drop']-pair['random']['deletion_logit_drop'])
                    icase.append(pair['IG']['insertion_logit_gain']-pair['random']['insertion_logit_gain'])
                ds.append(float(np.mean(dcase)));ins.append(float(np.mean(icase)))
            fidelity[str(frac)]={'deletion_advantage_over_random_mean':float(np.mean(ds)),'deletion_CI95_cases':boot_ci(ds,rng),
                                 'deletion_fraction_cases_IG_better':float(np.mean(np.array(ds)>0)),
                                 'insertion_advantage_over_random_mean':float(np.mean(ins)),'insertion_CI95_cases':boot_ci(ins,rng),
                                 'insertion_fraction_cases_IG_better':float(np.mean(np.array(ins)>0))}
        ok=[s for s in stable if s is not None]
        result['interpretability'].append({'model':m,'cases_per_seed':len(ids),'seeds':5,
          'mean_pairwise_spearman':float(np.mean(ok)) if ok else None,'spearman_CI95_cases':boot_ci(ok,rng),
          'undefined_constant_map_pairs':undefined,'fidelity':fidelity,
          'max_IG_completeness_abs_error':max(d['completeness_abs_error'] for r in ir for d in r['details']),
          'category_counts_all_seeds_not_independent':dict(Counter(d['category_geometric'] for r in ir for d in r['details'])),
          'scope':'Validación, casos de grupos distintos estratificados por área; IC por bootstrap de casos con la media de las 5 semillas. '
                  'Semillas correlacionadas; no explicación causal ni frecuencias poblacionales.'})
    n_models=len(models);n_top=len(cfg['top3_by_validation'])
    result['status']='complete' if not result['missing'] and len(result['clean'])==n_models and len(result['robustness'])==3*n_models \
        and len(result['interpretability'])==n_top and len(result['ensembles'])==n_top else 'partial'
    if limit:result['status']+='_PILOTO_limit_%d'%limit
    atomic_json(out/'RESUMEN_DIAGNOSTICOS.json',result)
    print('Estado:',result['status'],'Faltantes:',len(result['missing']),flush=True)
    for r in result['clean']:print(r['model'],'Dice',round(r['test']['Dice']['mean'],4),'Brier',round(r['test']['Brier']['mean'],5),'gap',round(r['gap_train_val']['difference'],4),flush=True)
    for e in result['ensembles']:
        mc=e['conformal']['mondrian_by_class'];print(e['model'],'conformal rótulo',round(mc['label']['coverage'],3),'fondo',round(mc['background']['coverage'],3),flush=True)

if __name__=='__main__':main()
