"""Audita resultados de test recibidos, sin ejecutar nuevas inferencias."""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from dataset_ready import partition,load_example,metrics
from experimento_e2_colab import sha,digest
from evaluar_e2_cerrado import grouped_scores

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent/'resultados_test_e2_final'
OUT=PROJECT/'resultados/auditoria_test_e2'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    closed=read(ROOT/'EVALUACION_CERRADA.json');summary=read(ROOT/'RESUMEN_TEST.json')
    protocol=read(PROJECT/'config/protocolo_test_e2_cerrado.json')
    issues=[]
    def check(ok,description):
        if not ok:issues.append(description)
    check(closed['protocol']==protocol,'protocolo diferente al congelado')
    expected_signature=digest({'protocol':protocol,'environment':closed['environment'],
                               'evaluator':sha(PROJECT/'scripts/evaluar_e2_cerrado.py')})
    check(expected_signature==closed['signature']==summary['signature'],'firma global')
    rows=partition('test',allow_test=True)
    check([r['id'] for r in rows]==protocol['test_order'],'orden test')
    truths={};negative_count=0
    for row in rows:
        check(sha(PROJECT.parent/row['image_path'])==row['image_sha256'],'imagen '+row['id'])
        check(sha(PROJECT/row['binary_mask_path'])==row['binary_mask_sha256'],'mascara '+row['id'])
        _,y,v=load_example(row);truths[row['id']]=(y,v)
        negative_count+=int(y[v].sum()==0)
    test_ids={r['id']:r['group_id'] for r in rows};group_names=sorted(set(test_ids.values()))
    by_model=defaultdict(list);mask_count=0;image_records=0
    expected_files={run['folder']+'.json' for run in protocol['runs']}
    check({p.name for p in ROOT.glob('*_seed*.json')}==expected_files,'archivos de corridas diferentes')
    for run in protocol['runs']:
        r=read(ROOT/(run['folder']+'.json'));tag=run['folder']
        check(digest({k:v for k,v in r.items() if k!='payload_sha256'})==r['payload_sha256'],tag+' payload')
        check(r['signature']==expected_signature,tag+' firma')
        check((r['model'],r['seed'])==(run['model'],run['seed']),tag+' identidad')
        check(r['checkpoint_sha256']==run['best_sha256'],tag+' checkpoint')
        check(len(r['details'])==len(rows) and {d['id']:d['group_id'] for d in r['details']}==test_ids,tag+' membresia')
        check([d['id'] for d in r['details']]==protocol['test_order'],tag+' orden')
        means,groups=grouped_scores(r['details'])
        check(means==r['means'] and groups==r['groups'],tag+' agregacion')
        for d in r['details']:
            image_records+=1
            check(all(np.isfinite(d[k]) and 0<=d[k]<=1 for k in ['Dice','IoU','mIoU','accuracy']),tag+' rango')
            if run['seed']==11:
                path=ROOT/'mascaras_seed11'/run['model']/(d['id']+'.png')
                with Image.open(path) as im:mask=np.array(im)
                check(set(np.unique(mask)).issubset({0,255}),tag+' mascara no binaria')
                y,v=truths[d['id']];recomputed=metrics(mask>0,y,v)
                check(all(abs(recomputed[k]-d[k])<1e-12 for k in recomputed),tag+' mascara/metricas '+d['id'])
                mask_count+=1
        by_model[r['model']].append(r)
        print(tag,'verificado',flush=True)
    validation=read(PROJECT/'resultados/auditoria_e2_recibida/auditoria.json')
    val={r['model']:r for r in validation['summary']}
    rng=np.random.default_rng(20261119);indices=rng.integers(0,len(group_names),size=(5000,len(group_names)))
    boots={};values_by_model={};table=[]
    for name,results in by_model.items():
        results.sort(key=lambda r:r['seed'])
        check([r['seed'] for r in results]==protocol['seeds'],name+' semillas')
        values=np.array([[r['groups'][g]['Dice'] for g in group_names] for r in results])
        values_by_model[name]=values
        boots[name]=values.mean(axis=0)[indices].mean(axis=1)
        lo,hi=np.quantile(boots[name],[.025,.975]);mean=float(values.mean());sd=float(values.mean(axis=1).std(ddof=1))
        original=next(r for r in summary['summary'] if r['model']==name)
        check(abs(mean-original['Dice_mean'])<1e-12 and abs(sd-original['Dice_sd_seeds'])<1e-12,name+' media/DE')
        check(np.allclose([lo,hi],original['Dice_CI95_groups_fixed_seeds'],atol=1e-12,rtol=0),name+' IC')
        table.append({'model':name,'Dice_val':val[name]['Dice_mean'],'Dice_test':mean,'Dice_sd_seeds':sd,
            'CI95':[float(lo),float(hi)],'IoU_test':st.mean(r['means']['IoU'] for r in results),
            'delta_test_val':mean-val[name]['Dice_mean'],'parameters':val[name]['parameters']})
    table.sort(key=lambda r:-r['Dice_test'])
    for contrast in summary['contrasts']:
        name=contrast['model'];ci=np.quantile(boots[name]-boots['fcn'],[.025,.975])
        check(np.allclose(ci,contrast['CI95_groups_fixed_seeds'],atol=1e-12,rtol=0),name+' contraste IC')
        check(abs(float(values_by_model[name].mean()-values_by_model['fcn'].mean())-contrast['Dice_difference'])<1e-12,name+' contraste media')
    check(len(by_model)==10 and len(expected_files)==50,'completitud')
    primary=next(c for c in summary['contrasts'] if c['primary'])
    report={'issues':issues,'runs':len(expected_files),'image_records':image_records,'masks_recomputed':mask_count,
        'test_images':len(rows),'test_groups':len(group_names),'negative_test_images':negative_count,
        'evaluation_device':closed['environment']['device'],'training_device':protocol['training_environment']['device'],
        'table':table,'primary_contrast':primary,'scope':'Audit only; no new inference or training.'}
    (OUT/'AUDITORIA.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Resultados finales E2: auditoría y conclusiones','',
        f"Se verificaron {len(expected_files)} corridas, {image_records} registros por imagen y {mask_count} máscaras de semilla 11. Incidencias: {len(issues)}.",
        f"Test: {len(rows)} imágenes, {len(group_names)} grupos; máscaras vacías: {negative_count}.",
        f"Entrenamiento: {report['training_device']}; evaluación: {report['evaluation_device']}. No comparar tiempos entre ambas GPUs como si fueran el mismo dispositivo.",'',
        '| Red | Dice validación | Dice test ± DE semillas | IC95 test por grupos | IoU test | Parámetros |',
        '|---|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['model']} | {r['Dice_val']:.4f} | {r['Dice_test']:.4f} ± {r['Dice_sd_seeds']:.4f} | [{r['CI95'][0]:.4f}, {r['CI95'][1]:.4f}] | {r['IoU_test']:.4f} | {r['parameters']:,} |")
    lines+=['','## Conclusión principal','',
        'TransUNet conserva el mayor promedio de Dice en test, mientras FCN pasa al segundo lugar y U-Net al tercero.',
        f"La diferencia principal TransUNet–FCN es {primary['Dice_difference']:.5f} (aproximadamente {primary['Dice_difference']*100:.2f} puntos porcentuales). IC95 pareado por grupos: [{primary['CI95_groups_fixed_seeds'][0]:.5f}, {primary['CI95_groups_fixed_seeds'][1]:.5f}].",
        'El intervalo incluye cero: esta evaluación no establece una mejora clara de TransUNet sobre FCN con este procedimiento. Tampoco demuestra equivalencia entre ambas.',
        'FCN compacta es una referencia favorable en coste/tamaño: 72.145 parámetros frente a 105.322.001 de TransUNet. Una recomendación basada en parsimonia es defendible; no confundirla con declarar a FCN ganadora en Dice.',
        'U-Net permanece competitiva. VM-UNet presenta mayor variación entre semillas. Los resultados bajos de SegFormer corresponden a esta adaptación desde cero y receta fija; no justifican inferioridad universal de los Transformers.',
        '', '## Límites de interpretación','',
        '- Los IC remuestrean grupos con cinco semillas fijas; la DE entre semillas se presenta aparte. Los grupos de similitud no equivalen necesariamente a escenas de captura independientes.',
        '- El contraste principal se fijó tras validación y antes de test. Los contrastes restantes son exploratorios y no tienen ajuste de multiplicidad.',
        '- Test tuvo exposición en E1. Se conservaron los checkpoints y umbral 0,5 fijados antes de esta evaluación; no describirlo como holdout nunca observado.',
        '- No atribuir diferencias sólo a arquitectura: capacidad, optimizador y auxiliares PIDNet también difieren; no hubo búsqueda exhaustiva de hiperparámetros.',
        '- Ausencia de imágenes negativas limita la evaluación de falsas alarmas en escenas sin etiquetas, si el conteo de máscaras vacías es cero.',
        '- Los tiempos registrados incluyen carga, transferencia y escritura de máscaras de semilla 11; no son una medición de latencia pura ni un benchmark de FPS.',
        '', '## Qué procede','',
        'Conservar estos resultados como cierre del benchmark E2. Integrar tabla, contraste principal, curvas de validación y ejemplos de errores en el entregable. Mantener la réplica del paper guía y el cierre bibliográfico/PRISMA como requisitos separados. Cualquier nuevo tuning tras ver este test sería exploratorio y requeriría una evaluación independiente futura.',
        '', '## Incidencias detectadas','',json.dumps(issues,ensure_ascii=False,indent=2)]
    (OUT/'CONCLUSIONES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'issues':issues,'runs':len(expected_files),'mask_checks':mask_count,'negative_test_images':negative_count,'primary':primary},indent=2))

if __name__=='__main__':main()
