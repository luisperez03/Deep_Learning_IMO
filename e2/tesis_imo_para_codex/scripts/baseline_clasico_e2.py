"""Baseline no profundo exploratorio. Tres configuraciones x cinco semillas.

Selección exclusivamente por Dice de validación; test se abre solo después del cierre.
No altera las redes cerradas. No iguala retroactivamente su búsqueda inexistente.
"""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from matplotlib.colors import rgb_to_hsv
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import joblib
from dataset_ready import partition,load_example,metrics

P=Path(__file__).resolve().parents[1]; OUT=P/'resultados/baseline_clasico_e2'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    temp=Path(str(p)+'.tmp');temp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(temp,p)

def descriptors(x,v,norm):
    rgb=np.clip(x.transpose(1,2,0)*np.array(norm['std'])+np.array(norm['mean']),0,1).astype('float32')
    hsv=rgb_to_hsv(rgb);gray=rgb.mean(axis=2);gy,gx=np.gradient(gray)
    yy,xx=np.indices(v.shape);ys,xs=np.where(v)
    yy=(yy-ys.min())/max(1,ys.max()-ys.min());xx=(xx-xs.min())/max(1,xs.max()-xs.min())
    extras=[xx,yy,np.hypot(gy,gx)]
    for size in [3,9]:
        avg=ndi.uniform_filter(gray,size=size);var=np.maximum(0,ndi.uniform_filter(gray*gray,size=size)-avg*avg)
        extras.extend([avg,np.sqrt(var)])
    return np.concatenate([rgb,hsv,np.stack(extras,axis=-1)],axis=-1).astype('float32')

def groupmean(details):
    groups=sorted({r['group_id'] for r in details})
    return {k:float(np.mean([np.mean([r[k] for r in details if r['group_id']==g]) for g in groups])) for k in ['Dice','IoU','mIoU','accuracy']}

def evaluate(model,rows,norm):
    details=[]
    for i,row in enumerate(rows):
        x,y,v=load_example(row);f=descriptors(x,v,norm);p=np.zeros_like(y,dtype='float32')
        p[v]=model.predict_proba(f[v])[:,1]
        details.append({'id':row['id'],'group_id':row['group_id'],**metrics(p,y,v),
                        'Brier':float(np.mean((p[v]-y[v])**2))})
        if (i+1)%100==0:print('Evaluadas',i+1,'/',len(rows),flush=True)
    return {'details':details,'means':groupmean(details)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=read(P/'config/complementos_e2_20261120.json');c=cfg['baseline'];seeds=cfg['seeds']
    norm=read(P/'resultados/preentrenamiento/normalization_train.json')
    signature=digest({'protocol':cfg,'script':sha(Path(__file__)),
      'data':{p:sha(P/'resultados/preentrenamiento'/p) for p in ['train.csv','val.csv','test.csv','normalization_train.json']}})
    lock=OUT/'PROTOCOLO.json'
    if lock.exists() and read(lock)['signature']!=signature:raise RuntimeError('Protocolo cambiado; usar carpeta nueva')
    save(lock,{'signature':signature,'protocol':cfg,'features':13,'framework':'scikit-learn 1.6.1',
               'training_scope':'exploratorio posterior a benchmark neuronal; todas las decisiones nuevas fijadas antes de entrenar este baseline'})
    samples=OUT/'muestras_train.npz'
    if not samples.exists():
        xs={s:[] for s in seeds};ys={s:[] for s in seeds};rng={s:np.random.default_rng(s) for s in seeds}
        for i,row in enumerate(partition('train')):
            x,y,v=load_example(row);f=descriptors(x,v,norm)[v];target=y[v]
            for s in seeds:
                ix=rng[s].choice(len(f),size=min(c['pixels_per_train_image'],len(f)),replace=False)
                xs[s].append(f[ix]);ys[s].append(target[ix])
            if (i+1)%200==0:print('Preparación train',i+1,flush=True)
        data={}
        for s in seeds:data[f'x{s}']=np.concatenate(xs[s]);data[f'y{s}']=np.concatenate(ys[s])
        with (OUT/'muestras_train.tmp').open('wb') as f:np.savez_compressed(f,**data)
        os.replace(OUT/'muestras_train.tmp',samples)
        del xs,ys,data
    data=np.load(samples);valrows=partition('val')
    validation=[]
    for leaves in c['max_leaf_nodes_candidates']:
        for seed in seeds:
            name=f'hgb_leaves{leaves}_seed{seed}';dest=OUT/(name+'_val.json');weights=OUT/(name+'.joblib')
            if dest.exists():
                old=read(dest)
                if old['signature']!=signature or old['weights_sha256']!=sha(weights):raise RuntimeError('Artefacto incompatible')
                validation.append(old);print('Reutilizado',name,flush=True);continue
            print('Entrenando',name,flush=True);start=time.perf_counter()
            model=HistGradientBoostingClassifier(max_iter=c['max_iter'],learning_rate=c['learning_rate'],
                 max_leaf_nodes=leaves,l2_regularization=c['l2_regularization'],early_stopping=False,
                 class_weight='balanced',random_state=seed)
            with threadpool_limits(limits=2):
                model.fit(data[f'x{seed}'],data[f'y{seed}']);train_seconds=time.perf_counter()-start
                result=evaluate(model,valrows,norm)
            temp=Path(str(weights)+'.tmp');joblib.dump(model,temp);os.replace(temp,weights)
            result.update(model='hgb_pixels',seed=seed,max_leaf_nodes=leaves,signature=signature,
                          weights_sha256=sha(weights),train_seconds=train_seconds)
            save(dest,result);validation.append(result);print(name,result['means'],flush=True)
    means={leaves:float(np.mean([r['means']['Dice'] for r in validation if r['max_leaf_nodes']==leaves])) for leaves in c['max_leaf_nodes_candidates']}
    chosen=max(c['max_leaf_nodes_candidates'],key=lambda n:(means[n],-n))
    selection={'signature':signature,'criterion':'Dice val medio sobre cinco semillas; menor complejidad ante empate','validation_means':means,'chosen_max_leaf_nodes':chosen}
    if (OUT/'SELECCION_ANTES_TEST.json').exists() and read(OUT/'SELECCION_ANTES_TEST.json')!=json.loads(json.dumps(selection)):
        raise RuntimeError('Selección cambió; no volver a seleccionar tras test')
    save(OUT/'SELECCION_ANTES_TEST.json',selection)
    results=[]
    for seed in seeds:
        dest=OUT/f'hgb_seed{seed}_test.json'
        if dest.exists():
            r=read(dest)
            if r['signature']!=signature or r['max_leaf_nodes']!=chosen:raise RuntimeError('Test incompatible')
        else:
            model=joblib.load(OUT/f'hgb_leaves{chosen}_seed{seed}.joblib')
            with threadpool_limits(limits=2):r=evaluate(model,partition('test',allow_test=True),norm)
            r.update(model='hgb_pixels',seed=seed,signature=signature,max_leaf_nodes=chosen);save(dest,r)
        results.append(r)
    groups=sorted({d['group_id'] for d in results[0]['details']})
    vals=np.array([[np.mean([d['Dice'] for d in r['details'] if d['group_id']==g]) for g in groups] for r in results])
    rng=np.random.default_rng(20261120);ix=rng.integers(len(groups),size=(5000,len(groups)))
    summary={'signature':signature,'selection':selection,'test_Dice':float(vals.mean()),'DE_seeds':float(vals.mean(axis=1).std(ddof=1)),
             'IC95_groups_fixed_seeds':np.quantile(vals.mean(axis=0)[ix].mean(axis=1),[.025,.975]).tolist(),
             'limit':'Clásico ajustado en 3 configuraciones; comparación exploratoria. No igualdad retroactiva de tuning con redes de una receta.'}
    save(OUT/'RESUMEN.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
