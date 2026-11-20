"""Métricas numéricas para diagnósticos E2; sin framework ni ajuste con test."""
import numpy as np
from scipy import ndimage as ndi
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
from dataset_ready import metrics

def reliability(p,y,bins=15,adaptive=False):
    p=np.asarray(p).ravel();y=np.asarray(y).ravel()
    if adaptive:
        edges=np.unique(np.quantile(p,np.linspace(0,1,bins+1)))
        b=np.searchsorted(edges[1:-1],p,side='right')
    else:b=np.minimum((p*bins).astype(int),bins-1)
    n=np.bincount(b,minlength=bins);a=np.bincount(b,weights=p,minlength=bins);t=np.bincount(b,weights=y,minlength=bins)
    return float(np.abs(a-t).sum()/len(p))

def probability_metrics(p,y,v):
    p=np.asarray(p)[v].astype('float64');t=np.asarray(y)[v].astype('float64')
    z=np.clip(p,1e-7,1-1e-7)
    return {'Brier':float(np.mean((p-t)**2)), 'NLL':float(-np.mean(t*np.log(z)+(1-t)*np.log1p(-z))),
            'ECE15':reliability(p,t), 'ECE_adaptive15':reliability(p,t,adaptive=True)}

def calibration_bins(p,y,v,bins=15):
    z=np.asarray(p)[v].astype('float64');t=np.asarray(y)[v];b=np.minimum((z*bins).astype(int),bins-1)
    return np.stack([np.bincount(b,minlength=bins),np.bincount(b,weights=z,minlength=bins),
                     np.bincount(b,weights=t,minlength=bins)],axis=1)/len(z)

def fit_temperature(logits,labels,weights):
    z=np.asarray(logits,dtype='float64');y=np.asarray(labels,dtype='float64');w=np.asarray(weights,dtype='float64');w=w/w.sum()
    def nll(logt):
        zz=z/np.exp(logt);return float(np.sum(w*(np.logaddexp(0,zz)-y*zz)))
    fit=minimize_scalar(nll,bounds=(np.log(.05),np.log(10)),method='bounded')
    if not fit.success:raise RuntimeError('Fallo temperature scaling')
    return {'temperature':float(np.exp(fit.x)),'validation_NLL_before':nll(0),'validation_NLL_after':float(fit.fun),
            'bounds':[.05,10.], 'at_bound':bool(abs(fit.x-np.log(.05))<1e-3 or abs(fit.x-np.log(10))<1e-3)}

def border_metrics(p,y,v):
    p=(np.asarray(p)>=.5)&v;y=np.asarray(y,bool)&v
    tp=int((p&y).sum());fp=int((p&~y&v).sum());fn=int((~p&y).sum())
    result={'precision':tp/(tp+fp) if tp+fp else float(not y.any()),
            'recall':tp/(tp+fn) if tp+fn else float(not p.any()),
            'TP':tp,'FP':fp,'FN':fn,'empty_prediction':int(not p.any())}
    if not p.any() or not y.any():
        dist=float(np.hypot(*y.shape)) if p.any()!=y.any() else 0.
        return {**result,'HD95':dist,'ASSD':dist}
    bp=p&~ndi.binary_erosion(p);by=y&~ndi.binary_erosion(y)
    d1=ndi.distance_transform_edt(~by)[bp];d2=ndi.distance_transform_edt(~bp)[by]
    return {**result,'HD95':float(max(np.percentile(d1,95),np.percentile(d2,95))),
            'ASSD':float((d1.sum()+d2.sum())/(len(d1)+len(d2)))}

def diagnose(p,y,v,with_border=True):
    d={**metrics(p,y,v),**probability_metrics(p,y,v)}
    if with_border:d.update(border_metrics(p,y,v))
    return d

def perturb(x,v,norm,kind,level,seed):
    """Transforma contenido RGB, sin mezclar padding ni cambiar anotación."""
    mean=np.asarray(norm['mean'],dtype='float32');std=np.asarray(norm['std'],dtype='float32')
    rgb=x.transpose(1,2,0)*std+mean;yy,xx=np.where(v)
    sl=(slice(yy.min(),yy.max()+1),slice(xx.min(),xx.max()+1))
    patch=rgb[sl].copy()
    if kind=='noise':patch+=np.random.default_rng(seed).normal(0,level,patch.shape).astype('float32')
    elif kind=='blur':patch=ndi.gaussian_filter(patch,sigma=(level,level,0))
    elif kind=='brightness':patch*=level
    else:raise ValueError(kind)
    rgb[sl]=np.clip(patch,0,1);out=(rgb-mean)/std;out[~v]=0
    return out.transpose(2,0,1).astype('float32')

def aggregate(rows,keys=None):
    if keys is None:keys=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k not in ['seed']]
    gs=sorted({r['group_id'] for r in rows})
    return {k:float(np.mean([np.mean([r[k] for r in rows if r['group_id']==g]) for g in gs])) for k in keys}

def label_set(p,y,v,mass=.9):
    """Conjunto binario de mínima cardinalidad con masa >= objetivo; empírico, no conforme."""
    pp=p[v];yy=y[v];single=(np.maximum(pp,1-pp)>=mass)
    correct=((pp>=.5)==yy)
    return {'coverage':float(np.mean(~single|correct)),'mean_set_size':float(np.mean(np.where(single,1,2))),
            'singleton_fraction':float(single.mean())}

# --- Predicción conforme por píxel, condicionada a la clase (Mondrian) ---
# Puntaje de no conformidad s = 1 - p(clase verdadera). Un umbral por clase: con uno solo, el fondo
# (~97 % de los píxeles válidos, casi todos fáciles) se lleva la garantía y los píxeles de rótulo quedan
# por debajo de la nominal. Garantía aproximada: los píxeles de una imagen no son intercambiables.

def conformal_quantile(scores,alpha):
    """k-ésimo menor puntaje con k = ceil((n+1)(1-alpha)); inf si k > n (muestra demasiado pequeña).
    Estadístico de orden exacto: np.quantile(..., 'higher') en k/n puede tomar el k+1."""
    s=np.asarray(scores,dtype='float64').ravel();n=len(s)
    if n==0:raise ValueError('Sin puntajes de calibración')
    k=int(np.ceil((n+1)*(1-alpha)-1e-12))
    return float('inf') if k>n else float(np.partition(s,k-1)[k-1])

def conformal_scores_by_class(p,y,v,group,store):
    """Acumula en store[clase][grupo] hasta per_group puntajes por grupo y clase (muestreo sin reemplazo)."""
    p=np.asarray(p,dtype='float32');y=np.asarray(y).astype(bool);v=np.asarray(v,bool)
    for c,s in ((1,(1-p)[y&v]),(0,p[~y&v])):
        if len(s):store[c].setdefault(group,[]).append(s)
    return store

def conformal_calibrate(store,alpha,per_group,seed):
    """store[c][grupo] -> listas de puntajes. Cada grupo aporta como máximo per_group píxeles por clase."""
    rng=np.random.default_rng(seed);cal={}
    for c in (0,1):
        chunks=[]
        for g in sorted(store[c]):
            s=np.concatenate(store[c][g]);chunks.append(rng.choice(s,min(per_group,len(s)),replace=False) if len(s)>per_group else s)
        cal[c]=np.concatenate(chunks)
    q={'background':conformal_quantile(cal[0],alpha),'label':conformal_quantile(cal[1],alpha),
       'marginal':conformal_quantile(np.concatenate([cal[0],cal[1]]),alpha)}
    return q,{'background':len(cal[0]),'label':len(cal[1])}

def conformal_sets(p,y,v,q_label,q_background):
    """Conjunto por píxel {c: 1-p_c <= q_c}. Devuelve conteos de cobertura y tamaño por clase verdadera."""
    p=np.asarray(p,dtype='float32')[v];t=np.asarray(y).astype(bool)[v]
    in_label=(1-p)<=q_label;in_bg=p<=q_background
    covered=np.where(t,in_label,in_bg);size=in_label.astype(int)+in_bg.astype(int)
    out={}
    for name,sel in (('all',np.ones_like(t)),('label',t),('background',~t)):
        out[name]={'covered':int(covered[sel].sum()),'pixels':int(sel.sum()),'size_sum':int(size[sel].sum()),
                   'empty':int((size[sel]==0).sum()),'both':int((size[sel]==2).sum())}
    return out
