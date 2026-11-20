"""Casos de validación seleccionados explícitamente, sin modificar modelos."""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from dataset_ready import partition, load_example, metrics
from preparar_base import letterbox
from modelos_e2 import MODELS_E2, main_logits

PROJECT=Path(__file__).resolve().parents[1]
RUNS=PROJECT.parent/'tesis_imo_e2_arquitecturas_reales_v2'
OUT=PROJECT/'resultados/revision_visual_e2'

@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    rows={r['id']:r for r in partition('val')}
    scores=defaultdict(lambda:defaultdict(list))
    for p in sorted(RUNS.glob('*_seed*/result.json')):
        result=json.loads(p.read_text())
        ck=torch.load(p.parent/result['best_file'],weights_only=True,map_location='cpu',mmap=True)
        for d in ck['validation']:scores[d['id']][result['model']].append(d['Dice'])
        del ck
    info=[]
    for id,row in rows.items():
        _,y,v=load_example(row)
        means={m:st.mean(a) for m,a in scores[id].items()}
        info.append({'id':id,'group_id':row['group_id'],'area_fraction':float(y[v].mean()),
                     'mean_all_models':st.mean(means.values()),'model_means':means,
                     'range_models':max(means.values())-min(means.values())})
    selected=[];used_groups=set()
    def choose(candidates,reason):
        for r in candidates:
            if r['group_id'] not in used_groups:
                selected.append({**r,'reason':reason});used_groups.add(r['group_id']);return
    choose(sorted(info,key=lambda r:r['mean_all_models']),'Menor Dice medio de los diez modelos')
    choose(sorted(info,key=lambda r:r['mean_all_models']),'Segundo grupo difícil')
    choose(sorted((r for r in info if r['area_fraction']>0),key=lambda r:r['area_fraction']),'Menor fracción de etiqueta positiva disponible')
    negatives=[r for r in info if r['area_fraction']==0]
    if negatives:choose(sorted(negatives,key=lambda r:r['mean_all_models']),'Negativo más difícil')
    else:choose(sorted(info,key=lambda r:-r['range_models']),'Mayor desacuerdo entre modelos; no hay máscaras vacías en val')
    median=st.median(r['mean_all_models'] for r in info)
    choose(sorted(info,key=lambda r:abs(r['mean_all_models']-median)),'Caso próximo a la mediana')
    choose(sorted(info,key=lambda r:-r['mean_all_models']),'Caso de alto acuerdo')
    manifest={'split':'val','selection':'descriptive, not representative sample; distinct similarity groups',
              'illustration_seed':11,'negative_images':len(negatives),'cases':selected}
    (OUT/'casos.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    models=['fcn','unet','transunet_2d'];predictions={}
    for name in models:
        folder=RUNS/f'{name}_seed11';r=json.loads((folder/'result.json').read_text())
        ck=torch.load(folder/r['best_file'],weights_only=True,map_location='cpu',mmap=True)
        model=MODELS_E2[name]();model.load_state_dict(ck['model']);model.eval()
        for case in selected:
            x,y,v=load_example(rows[case['id']]);p=main_logits(model(torch.from_numpy(x.copy())[None]))[0,0].sigmoid().numpy()
            predictions[(name,case['id'])]=(p>=.5,metrics(p,y,v)['Dice'])
        del model,ck
        print(name,'paneles inferidos',flush=True)
    panels=[]
    for case in selected:
        id=case['id'];row=rows[id];_,y,v=load_example(row)
        with Image.open(PROJECT.parent/row['image_path']) as original:im,_=letterbox(original.convert('RGB'),256)
        canvas=Image.new('RGB',(5*256,304),'#ffffff');draw=ImageDraw.Draw(canvas)
        canvas.paste(im,(0,48));draw.text((5,5),id+' | '+case['reason'],fill='black')
        draw.text((5,29),'Original',fill='black')
        truth=Image.fromarray((y*255).astype('uint8')).convert('RGB');canvas.paste(truth,(256,48));draw.text((261,29),'Mascara de referencia',fill='black')
        for col,name in enumerate(models,2):
            p,dice=predictions[(name,id)];colors=np.zeros((256,256,3),dtype='uint8')
            colors[p & y.astype(bool) & v]=(40,180,70)
            colors[p & ~y.astype(bool) & v]=(230,70,70)
            colors[~p & y.astype(bool) & v]=(70,130,240)
            canvas.paste(Image.fromarray(colors),(col*256,48));draw.text((col*256+5,29),f'{name} Dice={dice:.3f}',fill='black')
        canvas.save(OUT/f'{id}.png');panels.append(canvas)
    mosaic=Image.new('RGB',(1280,304*len(panels)+35),'white')
    ImageDraw.Draw(mosaic).text((8,10),'VALIDACION, semilla 11. Verde: acierto | Rojo: falso positivo | Azul: etiqueta omitida. Casos dirigidos, no muestra aleatoria.',fill='black')
    for i,panel in enumerate(panels):mosaic.paste(panel,(0,35+304*i))
    mosaic.save(OUT/'panel_validacion.png')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
