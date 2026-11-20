"""Manifiestos agrupados y máscaras binarias; nunca inicia entrenamiento.

Todos los enlaces perceptuales se conservan juntos por precaución. Los componentes
resultantes NO equivalen a escenas certificadas. Falsos positivos pueden reducir
la muestra efectiva, pero no se separan pares conocidos entre entrenamiento/test.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,json,random,sys
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,__version__ as pillow_version
from inicio_voc import PROJECT,sha,csv_write,palette

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def make_groups(ids,edges):
    parent={k:k for k in ids}
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    for edge in edges:
        a,b=find(edge['id_a']),find(edge['id_b'])
        if a!=b:parent[max(a,b)]=min(a,b)
    groups=defaultdict(list)
    for key in sorted(ids):groups['g_'+find(key)].append(key)
    return dict(groups)

def grouped_split(groups,fractions,seed):
    """Asignar grupos enteros por déficit de cuota; sin optimizar métricas del test."""
    order=sorted(groups);random.Random(seed).shuffle(order)
    order.sort(key=lambda k:-len(groups[k]))
    total=sum(map(len,groups.values()));target={s:total*v for s,v in fractions.items()}
    counts=Counter();result={}
    for group in order:
        split=max(fractions,key=lambda s:(target[s]-counts[s])/max(target[s],1))
        result[group]=split;counts[split]+=len(groups[group])
    return result

def binary_mask(mask,colors):
    rgb=np.asarray(mask.convert('RGB'))
    packed=rgb[:,:,0].astype(np.uint32)*65536+rgb[:,:,1].astype(np.uint32)*256+rgb[:,:,2]
    allowed=[r*65536+g*256+b for r,g,b in colors]
    if not np.isin(packed,allowed).all():raise ValueError('Color desconocido; no convertir automáticamente')
    return (packed!=0).astype(np.uint8)

def letterbox(image,size,is_mask=False):
    """Preservar aspecto; misma geometría RGB/máscara y vecino más cercano en máscara."""
    scale=min(size/image.width,size/image.height)
    shape=(max(1,round(image.width*scale)),max(1,round(image.height*scale)))
    resized=image.resize(shape,Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR)
    canvas=Image.new('L' if is_mask else 'RGB',(size,size),0)
    x=(size-shape[0])//2;y=(size-shape[1])//2;canvas.paste(resized,(x,y))
    valid=np.zeros((size,size),dtype=bool);valid[y:y+shape[1],x:x+shape[0]]=True
    return canvas,valid

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-root',type=Path,default=PROJECT.parent)
    args=parser.parse_args(argv);data=args.data_root/'VOC2012'
    cfgpath=PROJECT/'config/preentrenamiento.json';cfg=json.loads(cfgpath.read_text())
    out=PROJECT/'resultados/preentrenamiento';out.mkdir(parents=True,exist_ok=True)
    maskdir=out/'masks_binary';maskdir.mkdir(exist_ok=True)
    inv=PROJECT/'resultados/inicio_voc/inventario.csv';summ=PROJECT/'resultados/inicio_voc/resumen.json'
    pairpath=PROJECT/'resultados/revision_voc/candidatos_similares.csv'
    rows=read_csv(inv);base={r['id']:r for r in rows};edges=read_csv(pairpath)
    # Una preparación existente no puede cambiar de dataset/política en silencio.
    # Para otra versión, conservar primero esta carpeta y usar una versión nueva.
    if (out/'freeze_manifest.json').exists():
        previous=json.loads((out/'freeze_manifest.json').read_text())
        current_inputs={str(p.relative_to(PROJECT)).replace('\\','/'):sha(p) for p in [cfgpath,inv,summ,pairpath]}
        if previous['inputs']!=current_inputs:
            raise ValueError('Entradas/config cambiaron respecto al manifiesto congelado. Crear una nueva versión; no sobrescribir el test existente.')
    summary=json.loads(summ.read_text(encoding='utf-8'))
    original_run=json.loads((PROJECT/'resultados/inicio_voc/ejecucion.json').read_text(encoding='utf-8'))
    for item in original_run['source_metadata']:
        if sha(data/item['path'])!=item['sha256']:raise ValueError('Metadato cambió: '+item['path'])
    groups=make_groups(base,edges);id_group={k:g for g,ids in groups.items() for k in ids}
    reasons=defaultdict(list);checks=[];colors=palette(data/'labelmap.txt')
    if cfg['quarantine_semantic_disagreements']:
        for issue in summary['xml_mask_class_set_disagreements']:reasons[issue['id']].append('semantic_disagreement_pending_review')
    for key,reason in cfg['manual_quarantine'].items():reasons[key].append(reason)
    for i,row in enumerate(rows):
        key=row['id']
        if sha(data/'JPEGImages'/row['image'])!=row['image_sha256']:raise ValueError('Imagen cambió: '+key)
        if row['status']!='ok':reasons[key].append(row['status']);continue
        path=data/'SegmentationClass'/row['mask']
        if sha(path)!=row['mask_sha256']:raise ValueError('Máscara cambió: '+key)
        with Image.open(path) as im:binary=binary_mask(im,colors)
        xml=data/'bboxes/annotations'/f'{key}.xml'
        if xml.exists():
            for n,obj in enumerate(ET.parse(xml).getroot().findall('object')):
                box=obj.find('bndbox')
                if box is None:continue
                x1,y1,x2,y2=[int(float(box.findtext(k))) for k in ['xmin','ymin','xmax','ymax']]
                # Tolerancia de un píxel ante convenciones de índices VOC.
                crop=binary[max(0,y1-1):min(binary.shape[0],y2+1),max(0,x1-1):min(binary.shape[1],x2+1)]
                fraction=float(crop.mean()) if crop.size else 0.0
                if fraction<cfg['minimum_bbox_coverage']:
                    reasons[key].append('bbox_with_very_low_mask_coverage')
                    checks.append({'id':key,'object_index':n,'foreground_fraction_in_bbox':fraction})
        if (i+1)%300==0:print(f'Calidad y hashes: {i+1}/{len(rows)}',flush=True)
    # Sin máscara no equivale a anotación errónea: permite usar copias anotadas
    # del mismo grupo. Las alertas de calidad sí se propagan conservadoramente.
    quarantined={id_group[k] for k,v in reasons.items() if any(r!='missing_mask' for r in v)}
    eligible={g:[k for k in ids if base[k]['status']=='ok'] for g,ids in groups.items() if g not in quarantined}
    eligible={g:ids for g,ids in eligible.items() if ids}
    assignment=grouped_split(eligible,cfg['fractions'],cfg['seed'])
    if not eligible:raise ValueError('No quedan pares elegibles')
    if (out/'test.csv').exists():
        previous_test={r['id'] for r in read_csv(out/'test.csv')}
        planned_test={key for group,keys in eligible.items() if assignment[group]=='test' for key in keys}
        if planned_test!=previous_test:raise ValueError('El código cambiaría el test reservado; crear una versión explícita')
    manifest=[];stats=defaultdict(list);resolution_rows=[]
    channel_sum=np.zeros(3,dtype=np.float64);square_sum=np.zeros(3,dtype=np.float64);pixel_n=0
    for row in rows:
        key=row['id'];group=id_group[key];status='eligible_automatic_qc';reason=''
        if row['status']!='ok':status='excluded_no_usable_mask';reason=row['status']
        elif group in quarantined:status='quarantine';reason=';'.join(reasons[key]) or 'related_group_contains_flagged_annotation'
        split=assignment[group] if status=='eligible_automatic_qc' else ''
        record={'id':key,'group_id':group,'status':status,'reason':reason,'split':split,
          'image_path':f'VOC2012/JPEGImages/{row["image"]}',
          'original_mask_path':f'VOC2012/SegmentationClass/{row["mask"]}' if row['mask'] else '',
          'binary_mask_path':'','image_sha256':row['image_sha256'],'original_mask_sha256':row['mask_sha256'],
          'binary_mask_sha256':'','visual_approval':'not_individually_certified'}
        if split:
            with Image.open(data/'SegmentationClass'/row['mask']) as mask:binary=binary_mask(mask,colors)
            dest=maskdir/f'{key}.png';Image.fromarray(binary).save(dest)
            record['binary_mask_path']=str(dest.relative_to(PROJECT)).replace('\\','/');record['binary_mask_sha256']=sha(dest)
            stats[split].append(float(binary.mean()))
            # Solo train: ni escoger resolución ni ajustar normalización con test.
            if split=='train':
                with Image.open(data/'JPEGImages'/row['image']) as original:
                    boxed,valid=letterbox(original.convert('RGB'),cfg['resolution'])
                    arr=np.asarray(boxed,dtype=np.float64)[valid]/255.0
                    channel_sum+=arr.sum(axis=0);square_sum+=(arr**2).sum(axis=0);pixel_n+=len(arr)
                for size in cfg['compare_resolutions']:
                    boxed,_=letterbox(Image.fromarray(binary),size,True);fg=int(np.asarray(boxed).sum())
                    resolution_rows.append({'id':key,'size':size,'foreground_pixels_total':fg,'empty_after_resize':fg==0,'less_than_16_pixels_total':fg<16})
        manifest.append(record)
    csv_write(out/'manifest.csv',manifest,list(manifest[0]))
    csv_write(out/'bbox_coverage_flags.csv',checks,['id','object_index','foreground_fraction_in_bbox'])
    csv_write(out/'resolution_train.csv',resolution_rows,['id','size','foreground_pixels_total','empty_after_resize','less_than_16_pixels_total'])
    for split in cfg['fractions']:csv_write(out/f'{split}.csv',[r for r in manifest if r['split']==split],list(manifest[0]))
    assigned={r['id']:r['split'] for r in manifest if r['split']}
    crossings=[e for e in edges if e['id_a'] in assigned and e['id_b'] in assigned and assigned[e['id_a']]!=assigned[e['id_b']]]
    assert not crossings,'Fuga conocida entre particiones'
    mean=channel_sum/pixel_n;std=np.sqrt(np.maximum(square_sum/pixel_n-mean**2,0))
    normalization={'source':'train only, letterbox content excluding padding','size':cfg['resolution'],'mean':mean.tolist(),'std':std.tolist(),'pixels':pixel_n}
    (out/'normalization_train.json').write_text(json.dumps(normalization,indent=2),encoding='utf-8')
    result={'status_counts':dict(Counter(r['status'] for r in manifest)),
      'all_similarity_groups':len(groups),'largest_similarity_group':max(map(len,groups.values())),
      'quarantined_groups':len(quarantined),'splits':{s:{'images':len(v),'groups':sum(assignment[g]==s for g in assignment),'foreground_fraction_median':float(np.median(v))} for s,v in stats.items()},
      'known_candidate_edges_crossing_splits':len(crossings),
      'limits':['Conservative similarity groups are not certified independent scenes','Automatic QC does not certify every annotation','Quarantine is not proof that every flagged mask is wrong','Training disabled until explicit hardware/protocol review'],
      'resolution_train':{str(size):{'empty':sum(r['empty_after_resize'] for r in resolution_rows if r['size']==size),'less_than_16_pixels_total':sum(r['less_than_16_pixels_total'] for r in resolution_rows if r['size']==size)} for size in cfg['compare_resolutions']}}
    (out/'resumen.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    frozen={'script_sha256':sha(Path(__file__)),'helper_sha256':sha(PROJECT/'scripts/inicio_voc.py'),'python':sys.version,'packages':{'numpy':np.__version__,'Pillow':pillow_version},'config':cfg,
      'inputs':{str(p.relative_to(PROJECT)).replace('\\','/'):sha(p) for p in [cfgpath,inv,summ,pairpath]},
      'artifacts':{p.name:sha(p) for p in [out/'manifest.csv',out/'train.csv',out/'val.csv',out/'test.csv',out/'normalization_train.json']}}
    (out/'freeze_manifest.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
