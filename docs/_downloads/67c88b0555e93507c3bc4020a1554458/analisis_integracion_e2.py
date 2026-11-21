"""Análisis exploratorio posterior al test; conserva intactas las 50 corridas.

Unidad estadística: grupo de similitud, promediando las cinco semillas fijas.
No ejecuta modelos ni transforma estos análisis en confirmatorios retroactivos.
"""
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset_ready import partition, load_example, metrics

P = Path(__file__).resolve().parents[1]
OUT = P/'resultados/integracion_e2'
FIG = P/'libro/figuras/e2'
SEEDS = [11,23,37,53,71]
NAMES = {'fcn':'FCN compacta','segnet':'SegNet compacta','unet':'U-Net compacta',
 'deeplabv3plus':'DeepLabv3+ R18','unetplusplus':'U-Net++ R18','pidnet_s':'PIDNet-S',
 'segnext_t':'SegNeXt-T','segformer_b0':'SegFormer-B0','transunet_2d':'TransUNet','vmunet':'VM-UNet'}

def read(path): return json.loads(path.read_text(encoding='utf-8'))

def holm(p):
    p=np.asarray(p); order=np.argsort(p); result=np.empty(len(p))
    result[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return result

def signflip(d, seed=20261120, draws=49999):
    """Bilateral sobre media; exacta <=16 pares; Monte Carlo +1 en otro caso."""
    d=np.asarray(d,float); observed=abs(d.mean()); eps=1e-14
    if len(d)<=16:
        null=np.asarray(list(itertools.product([-1,1],repeat=len(d))))@d/len(d)
        return float(np.mean(abs(null)>=observed-eps))
    rng=np.random.default_rng(seed); count=0
    for start in range(0,draws,1000):
        signs=rng.integers(0,2,size=(min(1000,draws-start),len(d)))*2-1
        count+=int(np.count_nonzero(abs(signs@d/len(d))>=observed-eps))
    return (count+1)/(draws+1)

def aggregate(details, key):
    groups=sorted({r['group_id'] for r in details})
    return {g:float(np.mean([r[key] for r in details if r['group_id']==g])) for g in groups}

def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    audit=read(P/'resultados/auditoria_test_e2/AUDITORIA.json')
    assert not audit['issues']
    models=[r['model'] for r in audit['table']]
    raw={m:[read(P.parent/'resultados_test_e2_final'/f'{m}_seed{s}.json') for s in SEEDS] for m in models}
    groups=sorted(raw[models[0]][0]['groups']); ng=len(groups)
    scores={m:np.array([[r['groups'][g]['Dice'] for g in groups] for r in raw[m]]) for m in models}
    groupmean={m:a.mean(axis=0) for m,a in scores.items()}
    rng=np.random.default_rng(20261120); ix=rng.integers(ng,size=(10000,ng))
    contrasts=[]
    for m in models:
        if m=='fcn':continue
        d=groupmean[m]-groupmean['fcn']; boot=d[ix].mean(axis=1)
        sd=d.std(ddof=1); db=d[ix].mean(axis=1)/d[ix].std(axis=1,ddof=1)
        seed_d=scores[m].mean(axis=1)-scores['fcn'].mean(axis=1)
        # Normal approximation for planning only, not retrospective achieved power.
        z=stats.norm.ppf(1-.05/(2*9))+stats.norm.ppf(.8)
        contrasts.append({'model':m,'difference':float(d.mean()),'CI95':np.quantile(boot,[.025,.975]).tolist(),
          'd_z_group':float(d.mean()/sd),'d_z_CI95':np.quantile(db,[.025,.975]).tolist(),
          'P_group_superior':float(np.mean(d>0)+.5*np.mean(d==0)),
          'p_signflip_MC':signflip(d),'p_wilcoxon_asymptotic':float(stats.wilcoxon(d,method='approx').pvalue),
          'shapiro_difference_p':float(stats.shapiro(d).pvalue),
          'paired_seed_difference_sd':float(seed_d.std(ddof=1)),
          'planning_MDD_R5_9contrasts':float(z*seed_d.std(ddof=1)/np.sqrt(5)),
          'planning_R_delta002_9contrasts':int(math.ceil((z*seed_d.std(ddof=1)/.02)**2))})
    for key in ['p_signflip_MC','p_wilcoxon_asymptotic']:
        for row,adj in zip(contrasts,holm([c[key] for c in contrasts])):row[key+'_Holm9']=float(adj)
    friedman=stats.friedmanchisquare(*[groupmean[m] for m in models])
    matrix=np.stack([groupmean[m] for m in models],axis=1)
    ranks=stats.rankdata(-matrix,axis=1).mean(axis=0)
    seed_ranks=stats.rankdata(-np.stack([scores[m].mean(axis=1) for m in models],axis=1),axis=1).mean(axis=0)
    q=float(stats.studentized_range.ppf(.95,len(models),np.inf)/np.sqrt(2))
    cd=q*np.sqrt(len(models)*(len(models)+1)/(6*ng))
    baseline=[]; areas={}; idgroup={}
    for row in partition('test',allow_test=True):
        _,y,v=load_example(row); areas[row['id']]=float(y[v].mean()); idgroup[row['id']]=row['group_id']
        baseline.append({'id':row['id'],'group_id':row['group_id'],**metrics(np.zeros_like(y),y,v)})
    baseline_means={k:float(np.mean(list(aggregate(baseline,k).values()))) for k in ['Dice','IoU','mIoU','accuracy']}
    # Training-defined cutpoints, held fixed when describing test.
    train_areas=[]
    for row in partition('train'):
        _,y,v=load_example(row);train_areas.append(float(y[v].mean()))
    cut=np.quantile(train_areas,[.25,.5,.75]); strata=[]
    for m in models:
        byid={r['id']:float(np.mean([next(d['Dice'] for d in run['details'] if d['id']==r['id']) for run in raw[m]])) for r in raw[m][0]['details']}
        for k in range(4):
            ids=[i for i,a in areas.items() if np.searchsorted(cut,a,side='right')==k]
            gs=sorted({idgroup[i] for i in ids})
            vals=np.array([np.mean([byid[i] for i in ids if idgroup[i]==g]) for g in gs])
            b=vals[rng.integers(len(vals),size=(5000,len(vals)))].mean(axis=1)
            strata.append({'model':m,'quartile_train_area':k+1,'images':len(ids),'groups':len(gs),
                           'Dice':float(vals.mean()),'CI95':np.quantile(b,[.025,.975]).tolist()})
    val_audit={r['model']:r for r in read(P/'resultados/auditoria_e2_recibida/auditoria.json')['summary']}
    cost=[]
    for m in models:
        r=val_audit[m]; a=next(x for x in audit['table'] if x['model']==m)
        dominated=any(x['parameters']<=a['parameters'] and x['Dice_test']>=a['Dice_test'] and
                      (x['parameters']<a['parameters'] or x['Dice_test']>a['Dice_test']) for x in audit['table'])
        cost.append({'model':m,'parameters':r['parameters'],'hours_recorded':r['hours_recorded'],
                     'rank_groups':float(ranks[models.index(m)]),'rank_seeds':float(seed_ranks[models.index(m)]),
                     'pareto_point_estimates':not dominated})
    table_discrepancy=max(abs(float(scores[r['model']].mean())-r['Dice_test']) for r in audit['table'])
    result={'analysis_status':'Exploratory after test; fixed five seeds; approximate independent similarity groups',
      'alpha':.05,'delta_inherited_E1':.02,'bootstrap_draws':10000,'signflip_draws':49999,
      'contrasts':contrasts,'friedman':{'statistic':float(friedman.statistic),'p':float(friedman.pvalue),'blocks':ng},
      'nemenyi_CD':float(cd),'baseline_empty':baseline_means,'strata_cutpoints_train':cut.tolist(),
      'strata':strata,'cost':cost,'error_correlation':np.corrcoef(matrix.T).tolist(),'model_order':models,
      'max_table_discrepancy':table_discrepancy}
    (OUT/'analisis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Análisis complementario E2','',
      '**Exploratorio, especificado después de observar test.** Unidad: 122 grupos; cinco semillas fijas. No hay 610 réplicas independientes.',
      'Se conserva el contraste cerrado previo. Esta ampliación no sustituye un preregistro anterior al entrenamiento.',
      '',f'Friedman por grupos: estadístico {friedman.statistic:.5f}; p = {friedman.pvalue:.8g}.',
      'El contraste global detecta heterogeneidad del conjunto; no demuestra que TransUNet supere a FCN.',
      '', '| Modelo − FCN | Diferencia | IC95 | d_z (IC95) | p intercambio | p intercambio Holm(9) | p Wilcoxon Holm(9) |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for c in contrasts:
        lines.append(f"| {NAMES[c['model']]} | {c['difference']:.5f} | [{c['CI95'][0]:.5f}, {c['CI95'][1]:.5f}] | {c['d_z_group']:.3f} [{c['d_z_CI95'][0]:.3f}, {c['d_z_CI95'][1]:.3f}] | {c['p_signflip_MC']:.6g} | {c['p_signflip_MC_Holm9']:.6g} | {c['p_wilcoxon_asymptotic_Holm9']:.6g} |")
    lines += ['', '## Supuestos y alcance', '',
      '- Intercambio pareado: 49.999 cambios de signo Monte Carlo, p con corrección +1, bilateral sobre la media. Requiere intercambiabilidad del par bajo la nula; no es una prueba exacta exhaustiva ni una permutación de etiquetas con reentrenamiento.',
      '- Wilcoxon usa aproximación asintótica con corrección de empates y cero excluido; responde a localización/rangos bajo simetría, no exactamente a la misma hipótesis de media. Shapiro por diferencia se conserva en JSON; normalidad no acredita independencia.',
      '- Friedman/Nemenyi usan grupos como bloques, condicionales a modelos ya entrenados. La independencia de grupos es aproximada; la similitud no identifica escenas certificadas. No extrapolar a otros datasets.',
      '- Holm se aplica por separado a los nueve contrastes contra FCN para cada procedimiento. Los IC95 son marginales, no simultáneos. No elegir la prueba que dé el p más favorable.',
      '- d_z estandariza diferencias por grupo; no es d entre semillas. La probabilidad de superioridad por grupo y su dispersión se conservan en JSON.',
      '- δ heredado = 0,02 Dice. El IC del contraste principal incluye cero y se extiende fuera de ±δ; no acredita superioridad relevante ni equivalencia.',
      '', '## Referencia trivial y recursos', '',
      f"Máscara siempre vacía: Dice={baseline_means['Dice']:.4f}, IoU={baseline_means['IoU']:.4f}, mIoU={baseline_means['mIoU']:.4f}, exactitud={baseline_means['accuracy']:.4f}. No sustituye un clásico fuerte ajustado.",
      'El contraste frente a cero no prueba ausencia de fuga ni reemplaza la permutación completa de etiquetas.',
      '', '| Modelo | Parámetros | Horas registradas (5 semillas) | Rango por grupos | Rango por semillas | Pareto Dice/parámetros |',
      '|---|---:|---:|---:|---:|---|']
    for c in cost:lines.append(f"| {NAMES[c['model']]} | {c['parameters']:,} | {c['hours_recorded']:.3f} | {c['rank_groups']:.3f} | {c['rank_seeds']:.2f} | {'Sí' if c['pareto_point_estimates'] else 'No'} |")
    lines+=['','El tiempo suma épocas de entrenamiento y validación; excluye parte de E/S, interrupciones e instalación. No es tiempo puro de entrenamiento ni latencia. Pareto se refiere a estimaciones puntuales, sin afirmar dominancia estadística.',
      '', '## Potencia y sobreajuste', '',
      'El JSON guarda DE pareada entre semillas, MDD con R=5 y R aproximado para δ=0,02, α/9 y potencia 0,80. Es planificación futura normal basada en un piloto pequeño, no potencia prospectiva ni potencia observada certificada.',
      'Los historiales contienen pérdida de entrenamiento y Dice de validación: son escalas distintas. No restarlas para calcular una brecha ni afirmar ausencia de sobreajuste. Falta Dice de train en modo evaluación en el checkpoint elegido y un margen δg fijado previamente.',
      '',f'Discrepancia máxima al reconstruir la media principal desde los registros agrupados: {table_discrepancy:.3g}. Esta es reproducción aritmética; no nueva inferencia en entorno limpio.',
      '', '## Segmentos y errores', '',
      f'Cuartiles de fracción de área positiva definidos en train: {cut.tolist()}. Resultados por estrato e IC remuestreando grupos dentro del estrato en analisis.json. Un grupo puede aparecer en varios estratos; no tratarlos como independientes.',
      'La correlación de 1−Dice coincide con la de Dice y describe dificultad compartida. No es una medida de acuerdo humano ni demuestra ruido irreducible. Los subgrupos son descriptivos, no contrastes confirmatorios.']
    (OUT/'ANALISIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # All histories shown individually; no survivor-biased averaged learning curve.
    fig,axes=plt.subplots(5,2,figsize=(12,16),sharex=True)
    for ax,m in zip(axes.flat,models):
        for s in SEEDS:
            history=read(P.parent/'tesis_imo_e2_arquitecturas_reales_v2'/f'{m}_seed{s}'/'history.json')
            ax.plot([h['epoch'] for h in history],[h['val_group_Dice'] for h in history],label=str(s),lw=1)
        ax.set_title(NAMES[m]);ax.set_ylim(0,1);ax.grid(alpha=.2)
    axes[0,0].legend(title='Semilla',ncol=5,fontsize=7)
    fig.supxlabel('Época');fig.supylabel('Dice validación por grupos');fig.tight_layout()
    fig.savefig(FIG/'curvas_validacion.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(12,16),sharex=True)
    for ax,m in zip(axes.flat,models):
        for s in SEEDS:
            h=read(P.parent/'tesis_imo_e2_arquitecturas_reales_v2'/f'{m}_seed{s}'/'history.json')
            ax.plot([x['epoch'] for x in h],[x['train_loss'] for x in h],label=str(s),lw=1)
        ax.set_title(NAMES[m]);ax.grid(alpha=.2)
    axes[0,0].legend(title='Semilla',ncol=5,fontsize=7)
    fig.supxlabel('Época');fig.supylabel('Pérdida de entrenamiento (PIDNet incluye auxiliares)');fig.tight_layout()
    fig.savefig(FIG/'curvas_perdida.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for r in audit['table']:
        ax.scatter(r['parameters'],r['Dice_test'],s=45)
        ax.annotate(NAMES[r['model']],(r['parameters'],r['Dice_test']),xytext=(5,5),textcoords='offset points',fontsize=8)
    frontier=sorted([r for r in audit['table'] if next(c['pareto_point_estimates'] for c in cost if c['model']==r['model'])],key=lambda r:r['parameters'])
    ax.plot([r['parameters'] for r in frontier],[r['Dice_test'] for r in frontier],'--',alpha=.5)
    ax.set_xscale('log');ax.set_xlim(45000,3e8);ax.set_ylim(.68,.9)
    ax.set_xlabel('Parámetros (escala logarítmica)');ax.set_ylabel('Dice test');ax.grid(alpha=.2)
    ax.set_title('Eficiencia por tamaño · frontera de Pareto puntual');fig.tight_layout()
    fig.savefig(FIG/'pareto.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    order=np.argsort(ranks)
    ax.scatter(ranks[order],np.arange(len(models)),color='#176B87')
    ax.set_yticks(np.arange(len(models)),[NAMES[models[i]] for i in order]);ax.invert_yaxis()
    ax.set_xlim(.5,10.5);ax.set_xlabel('Rango promedio por grupo (menor es mejor)')
    ax.plot([1,1+cd],[9.6,9.6],lw=3);ax.text(1,9.4,f'CD Nemenyi = {cd:.3f}',fontsize=9)
    ax.set_ylim(10,-.7);ax.grid(axis='x',alpha=.2)
    ax.set_title('Rangos y diferencia crítica · análisis exploratorio por grupos')
    fig.tight_layout();fig.savefig(FIG/'rangos_cd.png',dpi=160);plt.close(fig)
    print(json.dumps({'friedman':result['friedman'],'baseline':baseline_means,'discrepancy':table_discrepancy},indent=2))

if __name__=='__main__':main()
