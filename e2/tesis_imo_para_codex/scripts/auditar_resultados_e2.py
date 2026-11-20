"""Auditoría de resultados recibidos: lectura de hashes y validación, sin inferencia."""
import csv
import hashlib
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
import torch

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent/'tesis_imo_e2_arquitecturas_reales_v2'
OUT=PROJECT/'resultados/auditoria_e2_recibida'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pref=read(ROOT/'preflight.json')
    with (PROJECT/'resultados/preentrenamiento/val.csv').open(encoding='utf-8-sig') as f:
        val={r['id']:r['group_id'] for r in csv.DictReader(f)}
    metadata=PROJECT/'resultados/preentrenamiento'
    local_data=digest({f:sha(metadata/f) for f in ['train.csv','val.csv','test.csv','normalization_train.json']})
    errors=[];runs=[];configs=set();identities=set();envs=set();per_model=defaultdict(list)
    for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        r=read(folder/'result.json');c=read(folder/'config.json');history=read(folder/'history.json')
        bad=[]
        def check(ok,message):
            if not ok:bad.append(message)
        spec={k:v for k,v in c.items() if k!='fingerprint'}
        check(digest(spec)==c['fingerprint']==r['fingerprint'],'fingerprint')
        check(r['completed'] is True,'not_completed')
        check(r['model']==c['name'] and r['seed']==c['seed'],'name_seed')
        check(c['identity']==pref['identity'],'preflight_identity')
        check(c['identity']['data']==local_data,'local_data_metadata_identity')
        check(c['environment']==pref['environment'],'environment_vs_preflight')
        check(digest(c['config'])==pref['config_hash'],'config_vs_preflight')
        check(r['parameters']==pref['models'][r['model']]['parameters'],'parameter_count')
        check(pref['models'][r['model']]['status']=='passed','preflight_not_passed')
        check(len(history)==r['epochs_completed'] and [h['epoch'] for h in history]==list(range(1,len(history)+1)),'history_sequence')
        check(all(math.isfinite(h[k]) for h in history for k in ['train_loss','val_group_Dice','seconds']),'nonfinite_history')
        best=max(h['val_group_Dice'] for h in history)
        best_epoch=next(h['epoch'] for h in history if h['val_group_Dice']==best)
        check(abs(best-r['best_val_group_Dice'])<1e-12,'history_best')
        check(r['best_file']==f'best_epoch_{best_epoch:03d}.pt','best_filename')
        check(len(history)==c['config']['epochs'] or len(history)-best_epoch>=c['config']['patience'],'premature_completion')
        hashes={}
        for key,file in [('last_sha256','last.pt'),('best_sha256',r['best_file'])]:
            path=folder/file
            hashes[file]=sha(path) if path.exists() else None
            check(hashes[file]==r[key],f'hash:{file}')
        # weights_only: no carga de clases del modelo ni pickle arbitrario de last.pt.
        ck=torch.load(folder/r['best_file'],map_location='cpu',weights_only=True,mmap=True)
        check(ck['fingerprint']==r['fingerprint'] and ck['epoch']+1==best_epoch,'best_checkpoint_metadata')
        details=ck['validation'];grouped=defaultdict(list)
        check(len(details)==len(val) and len({d['id'] for d in details})==len(val),'validation_count_duplicates')
        check({d['id']:d['group_id'] for d in details}==val,'validation_membership')
        for d in details:
            grouped[d['group_id']].append(d)
            check(all(math.isfinite(d[k]) and 0<=d[k]<=1 for k in ['Dice','IoU','mIoU','accuracy']),'validation_metrics_range')
        scores={k:st.mean(st.mean(d[k] for d in ds) for ds in grouped.values()) for k in ['Dice','IoU','mIoU','accuracy']}
        check(abs(scores['Dice']-best)<1e-12,'checkpoint_group_Dice')
        row={'model':r['model'],'seed':r['seed'],'Dice':scores['Dice'],'IoU':scores['IoU'],
             'mIoU':scores['mIoU'],'epochs':len(history),'best_epoch':best_epoch,
             'parameters':r['parameters'],'seconds':sum(h['seconds'] for h in history),
             'at_epoch_limit':len(history)==c['config']['epochs'],'issues':bad}
        runs.append(row);per_model[r['model']].append(row)
        configs.add(digest(c['config']));identities.add(digest(c['identity']));envs.add(digest(c['environment']))
        errors.extend({'run':folder.name,'issue':x} for x in bad)
        print(folder.name,'OK' if not bad else bad,flush=True)
        del ck
    for model,rows in per_model.items():
        if sorted(r['seed'] for r in rows)!=[11,23,37,53,71]:errors.append({'model':model,'issue':'seed_set'})
    if len(runs)!=50 or len(per_model)!=10:errors.append({'issue':'expected_50_runs_10_models'})
    summary=[]
    for model,rows in per_model.items():
        summary.append({'model':model,'n':len(rows),'Dice_mean':st.mean(r['Dice'] for r in rows),
            'Dice_sd':st.stdev(r['Dice'] for r in rows),'IoU_mean':st.mean(r['IoU'] for r in rows),
            'parameters':rows[0]['parameters'],'hours_recorded':sum(r['seconds'] for r in rows)/3600,
            'limit_runs':sum(r['at_epoch_limit'] for r in rows),
            'best_epoch_range':[min(r['best_epoch'] for r in rows),max(r['best_epoch'] for r in rows)]})
    summary.sort(key=lambda r:-r['Dice_mean'])
    result={'runs':len(runs),'models':len(summary),'validation_images':len(val),
        'validation_groups':len(set(val.values())),'config_variants':len(configs),
        'identity_variants':len(identities),'environment_variants':len(envs),
        'checkpoint_hashes_checked':2*len(runs),'issues':errors,'summary':summary,
        'per_run':runs,'scope':'validation only; no test inference; best checkpoints inspected weights_only; last checked by hash'}
    (OUT/'auditoria.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Auditoría de las 50 corridas E2 recibidas','',
        f'Carpeta leída: `{ROOT}`. No se modificaron corridas ni se ejecutó inferencia de test.',
        f'Corridas: {len(runs)}; modelos: {len(summary)}; incidencias: {len(errors)}.',
        f'Validación: {len(val)} imágenes, {len(set(val.values()))} grupos. Hashes de checkpoints comprobados: {2*len(runs)}.',
        f'Variantes de configuración/identidad/entorno: {len(configs)}/{len(identities)}/{len(envs)}.','',
        '| Modelo | Dice medio ± DE | IoU medio | Parámetros | Horas registradas | Corridas a 30 épocas |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['model']} | {r['Dice_mean']:.4f} ± {r['Dice_sd']:.4f} | {r['IoU_mean']:.4f} | {r['parameters']:,} | {r['hours_recorded']:.2f} | {r['limit_runs']}/5 |")
    lines+=['','DE: desviación estándar muestral entre cinco semillas, no intervalo de confianza. Cada puntuación corresponde al mejor checkpoint seleccionado por validación agrupada. Las horas suman entrenamiento + validación por época; no incluyen necesariamente guardado, instalación, interrupciones o cobro total.',
        '', '## Interpretación y pasos siguientes', '',
        '- TransUNet lidera esta validación; no se afirma superioridad general ni en test. U-Net y FCN compactas ofrecen una relación tamaño/rendimiento favorable.',
        '- No seleccionar la mejor semilla para presentar el modelo: mantener las cinco y reportar dispersión.',
        '- No concluir que Transformer/SSM son inferiores por los resultados de una receta desde cero y presupuesto limitado. Optimizadores, capacidades y pérdidas auxiliares difieren.',
        '- Congelar los 50 checkpoints y recetas. Revisar errores cualitativos en validación (casos comunes, etiquetas pequeñas, ausencia de etiqueta y bordes).',
        '- Preparar evaluación final de test de las diez redes con las cinco semillas, mismos grupos/métricas y umbral 0,5; ejecutar tras autorización explícita del cierre. Declarar exposición previa al test del E1.',
        '- Después: comparación pareada por grupos/semillas, incertidumbre y coste, figuras y actualización del libro. No usar test para nuevas decisiones de hiperparámetros.',
        '- La réplica del paper guía y el cierre de literatura/PRISMA siguen siendo tareas separadas.',
        '', '## Incidencias', '', json.dumps(errors,ensure_ascii=False,indent=2)]
    (OUT/'INFORME.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_run']},indent=2),flush=True)

if __name__=='__main__':main()
