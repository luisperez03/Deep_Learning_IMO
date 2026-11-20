"""Segunda etapa: similitud visual, metadatos de origen y revisión ampliada.

No elimina imágenes, corrige máscaras ni crea splits. Los hashes perceptuales
resumen apariencia y pueden confundir fondos uniformes. Todas las coincidencias
se registran como candidatos; no son duplicados confirmados ni escenas verificadas.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,json,hashlib,sys,platform
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageOps,ImageDraw,__version__ as pillow_version
from inicio_voc import PROJECT,sha,csv_write

def visual_hashes(im):
    """dHash: cambios horizontales de brillo; aHash: brillo respecto a la media.

    Ambos tienen 64 bits. Distancia Hamming = número de bits diferentes (0..64).
    No son invariantes a recortes, rotaciones ni perspectivas fuertes.
    """
    gray=im.convert('L')
    a=np.asarray(gray.resize((9,8),Image.Resampling.LANCZOS),dtype=np.int16)
    b=np.asarray(gray.resize((8,8),Image.Resampling.LANCZOS),dtype=np.float32)
    def integer(bits):
        value=0
        for bit in bits.ravel():value=(value<<1)|int(bit)
        return value
    return integer(a[:,1:]>a[:,:-1]),integer(b>b.mean())

def candidate_pairs(records,dmax,amax):
    """Comparación exhaustiva de pares, no muestreo: O(n²), viable para este VOC."""
    pairs=[]
    for i,a in enumerate(records):
        for b in records[i+1:]:
            dd=(a['dhash']^b['dhash']).bit_count()
            if dd>dmax:continue
            da=(a['ahash']^b['ahash']).bit_count()
            if da>amax:continue
            pairs.append({'id_a':a['id'],'id_b':b['id'],'dhash_distance':dd,'ahash_distance':da,
              'split_a':a['original_split'],'split_b':b['original_split'],
              'cross_train_val':{a['original_split'],b['original_split']}=={'train','val'},
              'source_folder_a':a['source_folder_xml'],'source_folder_b':b['source_folder_xml']})
    return sorted(pairs,key=lambda p:(p['dhash_distance']+p['ahash_distance'],p['id_a'],p['id_b']))

def pixel_hash(im):
    """Igualdad RGB exacta tras decodificar, incluyendo dimensiones.

    Ignora diferencias de metadatos del JPEG; no corrige orientación EXIF.
    Solo igualdad de este hash permite llamar idénticos a los píxeles.
    """
    rgb=im.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode()+b'\0'+rgb.tobytes()).hexdigest()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=PROJECT.parent)
    args=parser.parse_args(argv);data=args.data_root/'VOC2012'
    source=PROJECT/'resultados/inicio_voc';out=PROJECT/'resultados/revision_voc';out.mkdir(parents=True,exist_ok=True)
    config_path=PROJECT/'config/revision_voc.json';cfg=json.loads(config_path.read_text())
    with (source/'inventario.csv').open(encoding='utf-8-sig') as f:inventory=list(csv.DictReader(f))
    summary=json.loads((source/'resumen.json').read_text(encoding='utf-8'))
    oldrun=json.loads((source/'ejecucion.json').read_text(encoding='utf-8'))
    # Verificar los XML contra la auditoría previa evita mezclar versiones silenciosamente.
    for item in oldrun['source_metadata']:
        if sha(data/item['path'])!=item['sha256']:raise ValueError(f"Fuente cambió: {item['path']}; repetir inicio_voc.py")
    records=[];source_hashes=[]
    for i,row in enumerate(inventory):
        path=data/'JPEGImages'/row['image']
        if not row['image']:continue
        digest=sha(path)
        if digest!=row['image_sha256']:raise ValueError(f'{path.name} cambió; repetir auditoría')
        with Image.open(path) as f:
            im=f.convert('RGB');dh,ah=visual_hashes(im);ph=pixel_hash(im)
            # Estadísticos por imagen sobre miniatura: diagnóstico de iluminación,
            # NO estadísticas de normalización para entrenar un modelo.
            thumb=np.asarray(im.resize((cfg['thumbnail_size'],cfg['thumbnail_size']),Image.Resampling.BILINEAR),dtype=np.float32)/255
        xml=data/'bboxes/annotations'/f"{row['id']}.xml"
        folder=ET.parse(xml).getroot().findtext('folder','unknown') if xml.exists() else 'unknown'
        luminance=thumb@np.array([.2126,.7152,.0722])
        records.append({'id':row['id'],'original_split':row['original_split'],'source_folder_xml':folder,
          'dhash':dh,'ahash':ah,'rgb_pixel_sha256':ph,'brightness_thumb':float(luminance.mean()),'contrast_thumb':float(luminance.std()),
          'red_mean_thumb':float(thumb[:,:,0].mean()),'green_mean_thumb':float(thumb[:,:,1].mean()),'blue_mean_thumb':float(thumb[:,:,2].mean())})
        if (i+1)%300==0:print(f'Firmas visuales: {i+1}/{len(inventory)}',flush=True)
    pairs=candidate_pairs(records,cfg['dhash_max'],cfg['ahash_max'])
    pixel_groups=defaultdict(list)
    for record in records:pixel_groups[record['rgb_pixel_sha256']].append(record)
    identical=[[r['id'] for r in group] for group in pixel_groups.values() if len(group)>1]
    identical_cross=[[r['id'] for r in group] for group in pixel_groups.values()
                     if {'train','val'}.issubset({r['original_split'] for r in group})]
    fields=['id_a','id_b','dhash_distance','ahash_distance','split_a','split_b','cross_train_val','source_folder_a','source_folder_b']
    csv_write(out/'candidatos_similares.csv',pairs,fields)
    csv_write(out/'caracteristicas_imagen.csv',records,list(records[0]))
    groups=defaultdict(list)
    for r in records:groups[r['source_folder_xml']].append(r)
    group_rows=[{'source_folder_xml':k,'images':len(v),'train':sum(r['original_split']=='train' for r in v),'val':sum(r['original_split']=='val' for r in v),'unassigned':sum(r['original_split']=='unassigned' for r in v)} for k,v in sorted(groups.items())]
    csv_write(out/'grupos_origen_xml.csv',group_rows,list(group_rows[0]))
    # Priorizar el riesgo entre splits para mostrar, sin modificar las particiones.
    selected=sorted(pairs,key=lambda p:(not p['cross_train_val'],p['dhash_distance']+p['ahash_distance'],p['id_a'],p['id_b']))[:cfg['pair_panel_limit']]
    for start in range(0,len(selected),4):
        subset=selected[start:start+4];canvas=Image.new('RGB',(1400,360*len(subset)),'white');d=ImageDraw.Draw(canvas)
        for j,pair in enumerate(subset):
            title=f"{pair['id_a']} ({pair['split_a']}) / {pair['id_b']} ({pair['split_b']}) | dHash={pair['dhash_distance']}, aHash={pair['ahash_distance']} | CANDIDATO"
            d.text((10,j*360+5),title,fill='black')
            for col,key in enumerate(['id_a','id_b']):
                with Image.open(data/'JPEGImages'/f'{pair[key]}.jpg') as im:
                    canvas.paste(ImageOps.contain(im.convert('RGB'),(680,320)),(col*700+10,j*360+30))
        canvas.save(out/f'pares_{start//4+1:02d}.jpg',quality=92)
    disagreements={r['id']:r for r in summary['xml_mask_class_set_disagreements']}
    priorities=list(dict.fromkeys(cfg['priority_mask_ids']+sorted(disagreements)))[:cfg['mask_review_limit']]
    review=[]
    for key in priorities:
        row=next(r for r in inventory if r['id']==key)
        mp=data/'SegmentationClass'/f'{key}.png'
        if sha(mp)!=row['mask_sha256']:raise ValueError('Máscara cambió: '+key)
        with Image.open(data/'JPEGImages'/f'{key}.jpg') as f:im=f.convert('RGB')
        with Image.open(mp) as f:mask=f.convert('RGB')
        # Se usa la resolución original; no hay reducción dentro de cada panel.
        a=np.asarray(mask);fg=np.any(a!=0,axis=2)
        overlay=np.asarray(im).copy();overlay[fg]=(.6*overlay[fg]+.4*np.array([0,255,0])).astype(np.uint8)
        canvas=Image.new('RGB',(im.width*3,im.height+70),'white');d=ImageDraw.Draw(canvas)
        canvas.paste(im,(0,70));canvas.paste(mask,(im.width,70));canvas.paste(Image.fromarray(overlay),(2*im.width,70))
        d.text((10,8),key+' | Original / Mascara / Superposicion | Pixeles originales',fill='black')
        xml=data/'bboxes/annotations'/f'{key}.xml'
        boxclasses=sorted({o.findtext('name') for o in ET.parse(xml).getroot().findall('object')}) if xml.exists() else ['sin XML']
        d.text((10,28),'XML: '+', '.join(boxclasses)+' | Mascara: '+row['classes'].replace('|',', '),fill='black')
        canvas.save(out/f'mascara_{key}.png')
        review.append({'id':key,'class_disagreement':key in disagreements,'review_status':'pending','notes':''})
    csv_write(out/'revision_mascaras_plantilla.csv',review,list(review[0]))
    result={'images_compared':len(records),'all_pairs_examined':len(records)*(len(records)-1)//2,'candidate_pairs':len(pairs),
      'identical_rgb_pixel_groups':identical,'identical_rgb_pixel_groups_cross_train_val':identical_cross,
      'candidate_pairs_cross_train_val':sum(p['cross_train_val'] for p in pairs),
      'source_folders':group_rows,'mask_review_ids':priorities,'pair_panel_ids':selected,
      'limitations':['Coincidencias visuales son candidatos, no duplicados confirmados','folder XML no identifica necesariamente una escena: puede ser persona o lote','No se modifican splits ni originales','Hash perceptual puede omitir rotaciones, recortes o cambios de perspectiva','Estadisticos de miniatura no deben usarse como normalizacion de entrenamiento']}
    (out/'resumen.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    inputs={str(p.relative_to(PROJECT)).replace('\\','/'):sha(p) for p in [config_path,source/'inventario.csv',source/'resumen.json',source/'ejecucion.json']}
    run={'time_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'packages':{'numpy':np.__version__,'Pillow':pillow_version},'config':cfg,'inputs':inputs,'script_sha256':sha(Path(__file__)),'helper_sha256':sha(PROJECT/'scripts/inicio_voc.py'),'original_image_hashes_verified':len(records),'original_metadata_hashes_verified':len(oldrun['source_metadata'])}
    (out/'ejecucion.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"{len(pairs)} pares candidatos; {result['candidate_pairs_cross_train_val']} entre train/val originales; {len(groups)} carpetas de origen XML.",flush=True)
    print(f'Grupos RGB idénticos: {len(identical)}; entre train/val: {len(identical_cross)}',flush=True)

if __name__=='__main__':main()
