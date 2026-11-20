"""EDA, shift, bordes guardados y sensibilidad estadística; sin inferencia neuronal."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import ndimage as ndi, stats
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from dataset_ready import load_example, partition
from analisis_integracion_e2 import holm, NAMES

P=Path(__file__).resolve().parents[1]; OUT=P/'resultados/complementos_locales_e2'
CFG=P/'config/complementos_e2_20261120.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def border(p,y,v):
    p=np.asarray(p,bool)&v;y=np.asarray(y,bool)&v
    tp=int((p&y).sum());fp=int((p&~y&v).sum());fn=int((~p&y).sum())
    prec=tp/(tp+fp) if tp+fp else float(not y.any())
    rec=tp/(tp+fn) if tp+fn else float(not p.any())
    if not p.any() or not y.any():
        penalty=float(np.hypot(*y.shape)) if p.any()!=y.any() else 0.
        return {'precision':prec,'recall':rec,'HD95':penalty,'ASSD':penalty,'empty_prediction':not bool(p.any())}
    bp=p&~ndi.binary_erosion(p);by=y&~ndi.binary_erosion(y)
    d1=ndi.distance_transform_edt(~by)[bp];d2=ndi.distance_transform_edt(~bp)[by]
    return {'precision':prec,'recall':rec,'HD95':float(max(np.percentile(d1,95),np.percentile(d2,95))),
            'ASSD':float((d1.sum()+d2.sum())/(len(d1)+len(d2))),'empty_prediction':False}

def summarize_group(rows,keys):
    gs=sorted({r['group_id'] for r in rows})
    return {k:float(np.mean([np.mean([r[k] for r in rows if r['group_id']==g]) for g in gs])) for k in keys}

def main():
    OUT.mkdir(parents=True,exist_ok=True); cfg=read(CFG)
    save(OUT/'PROTOCOLO.json',{'protocol':cfg,'sha256':hashlib.sha256(CFG.read_bytes()).hexdigest()})
    norm=read(P/'resultados/preentrenamiento/normalization_train.json')
    mean=np.array(norm['mean']);std=np.array(norm['std']);features=[];eda=[]
    hist=defaultdict(lambda:np.zeros((2,3,32),dtype=np.int64)); truths={}
    for split in ['train','val','test']:
        for i,row in enumerate(partition(split,allow_test=split=='test')):
            x,y,v=load_example(row);rgb=np.clip(x.transpose(1,2,0)*std+mean,0,1)
            pixels=rgb[v];fg=float(y[v].mean());h=-sum(z*np.log2(z) for z in [fg,1-fg] if z>0)
            labels,n=ndi.label(y&v); sizes=np.bincount(labels.ravel())[1:]
            gray=rgb.mean(axis=2);gradient=np.hypot(ndi.sobel(gray,0),ndi.sobel(gray,1))
            hist_features=[]
            for c in range(3):
                hist_features.extend((np.histogram(pixels[:,c],bins=16,range=(0,1))[0]/len(pixels)).tolist())
                for k in [0,1]:hist[split][k,c]+=np.histogram(rgb[...,c][v&(y==k)],bins=32,range=(0,1))[0]
            yy,xx=np.where(v);aspect=(xx.max()-xx.min()+1)/(yy.max()-yy.min()+1)
            desc=[*pixels.mean(axis=0),*pixels.std(axis=0),*hist_features,float(gradient[v].mean()),float(gray[v].std()),float(aspect)]
            features.append({'id':row['id'],'group_id':row['group_id'],'split':split,'features':desc})
            perimeter=int((y.astype(bool)&~ndi.binary_erosion(y.astype(bool))&v).sum())
            eda.append({'id':row['id'],'group_id':row['group_id'],'split':split,'foreground_fraction':fg,
              'binary_entropy_bits':float(h),'regions':int(n),'smallest_region_pixels':int(sizes.min()) if len(sizes) else 0,
              'perimeter_pixels':perimeter,'valid_pixels':int(v.sum())})
            if split=='test':truths[row['id']]=(y,v)
            if (i+1)%200==0:print(split,i+1,flush=True)
    save(OUT/'eda_por_imagen.json',eda);save(OUT/'descriptores_imagen.json',features)
    save(OUT/'histogramas_rgb_por_clase.json',{k:v.tolist() for k,v in hist.items()})
    summary={split:summarize_group([r for r in eda if r['split']==split],['foreground_fraction','binary_entropy_bits','regions','perimeter_pixels']) for split in ['train','val','test']}
    # One row per group; descriptors do not use masks or model predictions.
    fs=[r for r in features if r['split']!='val'];groups=sorted({r['group_id'] for r in fs})
    xx=np.array([np.mean([r['features'] for r in fs if r['group_id']==g],axis=0) for g in groups])
    yy=np.array([int(next(r['split'] for r in fs if r['group_id']==g)=='test') for g in groups])
    def auc(labels):
        pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=20261120))
        cv=StratifiedKFold(5,shuffle=True,random_state=20261120)
        prob=cross_val_predict(pipe,xx,labels,cv=cv,method='predict_proba',n_jobs=1)[:,1]
        return float(roc_auc_score(labels,prob)),prob
    with threadpool_limits(limits=2):
        observed,prob=auc(yy);rng=np.random.default_rng(20261120);null=[]
        for i in range(cfg['adversarial']['permutations']):
            null.append(auc(rng.permutation(yy))[0])
            if (i+1)%50==0:print('Permutación de dominio',i+1,flush=True)
    bs=[]
    for _ in range(5000):
        ix=rng.integers(len(yy),size=len(yy))
        if len(np.unique(yy[ix]))==2:bs.append(roc_auc_score(yy[ix],prob[ix]))
    adv={'auc_crossfit':observed,'CI95_conditional_predictions':np.quantile(bs,[.025,.975]).tolist(),
         'p_permutation_MC':float((1+sum(v>=observed for v in null))/(len(null)+1)),
         'null_auc':null,'groups':groups,'domain':yy.tolist(),'probability':prob.tolist(),
         'limit':'Sin rechazo no prueba ausencia de shift; IC condicional a predicciones crossfit. No es permutación de etiquetas de segmentación.'}
    save(OUT/'validacion_adversarial.json',adv)
    # Border evaluation is limited to the ten available seed-11 mask sets.
    border_summary=[];all_border=[]
    for model in NAMES:
        rows=[]
        for row in partition('test',allow_test=True):
            y,v=truths[row['id']]
            with Image.open(P.parent/'resultados_test_e2_final/mascaras_seed11'/model/(row['id']+'.png')) as im:pred=np.asarray(im)>0
            rows.append({'model':model,'seed':11,'id':row['id'],'group_id':row['group_id'],**border(pred,y,v)})
        all_border.extend(rows);border_summary.append({'model':model,'seed':11,**summarize_group(rows,['precision','recall','HD95','ASSD','empty_prediction'])})
        print('Bordes',model,flush=True)
    save(OUT/'bordes_seed11_por_imagen.json',all_border);save(OUT/'bordes_seed11_resumen.json',border_summary)
    # Exact orthogonal finite-grid SS decomposition, not population variance components.
    models=list(NAMES);rs={m:[read(P.parent/'resultados_test_e2_final'/f'{m}_seed{seed}.json') for seed in cfg['seeds']] for m in models}
    gs=sorted(rs[models[0]][0]['groups']);a=np.array([[[r['groups'][g]['Dice'] for g in gs] for r in rs[m]] for m in models])
    grand=a.mean();effects={};remainder=a-grand
    for name,axis in [('modelo',0),('semilla',1),('grupo',2)]:
        e=a.mean(axis=tuple(j for j in range(3) if j!=axis),keepdims=True)-grand
        effects[name]=np.broadcast_to(e,a.shape);remainder=remainder-effects[name]
    interactions={}
    for i,j in [(0,1),(0,2),(1,2)]:
        k=next(k for k in range(3) if k not in (i,j))
        e=a.mean(axis=k,keepdims=True)-grand-list(effects.values())[i]-list(effects.values())[j]
        interactions[f'{i}_{j}']=e;remainder=remainder-e
    total=float(((a-grand)**2).sum());ss={k:float((v*v).sum()) for k,v in {**effects,**interactions,'interaccion_triple_residual':remainder}.items()}
    assert abs(sum(ss.values())-total)<1e-8
    d=a[models.index('transunet_2d')].mean(axis=0)-a[models.index('fcn')].mean(axis=0)
    delta=cfg['delta'];p1=float(stats.ttest_1samp(d,-delta,alternative='greater').pvalue);p2=float(stats.ttest_1samp(d,delta,alternative='less').pvalue)
    sem=stats.sem(d);ci90=stats.t.interval(.9,len(d)-1,loc=d.mean(),scale=sem)
    stat={'SS_total':total,'SS_components':ss,'fractions_descriptive':{k:v/total for k,v in ss.items()},
      'interpretation':'Descomposición descriptiva ortogonal de cuadrícula modelo×semilla×grupo; no modelo causal ni componentes de varianza poblacionales.',
      'tost_exploratory':{'delta':delta,'difference':float(d.mean()),'p_lower':p1,'p_upper':p2,'p_TOST':max(p1,p2),
           'CI90_t':list(map(float,ci90)),'equivalent_alpha05':max(p1,p2)<.05,
           'assumptions':'Grupos aproximadamente independientes; aproximación t a la media, sensibilidad bootstrap ya disponible; cinco semillas fijas.'}}
    save(OUT/'sensibilidad_estadistica.json',stat)
    save(OUT/'RESUMEN.json',{'EDA':summary,'adversarial':{k:v for k,v in adv.items() if k not in ['null_auc','groups','domain','probability']},'statistics':stat,'borders':border_summary})
    print('Complementos locales finalizados',flush=True)

if __name__=='__main__':main()
