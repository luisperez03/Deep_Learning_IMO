"""Auditoría reproducible de VOC, solo lectura de originales. Python 3.10+."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageOps, ImageDraw, __version__ as pillow_version

PROJECT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def palette(path):
    colors = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.startswith('#'): continue
        name, rgb, *_ = line.split(':')
        colors[tuple(map(int, rgb.split(',')))] = name
    return colors

def mask_stats(mask, colors):
    rgb = mask.convert('RGB')
    counts = rgb.getcolors(maxcolors=rgb.width*rgb.height) or []
    known = Counter(); unknown = Counter()
    for count, color in counts:
        if color in colors: known[colors[color]] += count
        else: unknown[str(color)] += count
    total = rgb.width * rgb.height
    foreground = sum(n for k,n in known.items() if k != 'background')
    return known, unknown, foreground / total

def csv_write(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

def make_panels(dataset, rows, output, seed, count):
    eligible = [r for r in rows if r['status'] == 'ok']
    selected = random.Random(seed).sample(eligible, min(count,len(eligible)))
    csv_write(output/'muestra_visual.csv', [{'id':r['id'],'revision':'pendiente','observacion':''} for r in selected], ['id','revision','observacion'])
    for start in range(0,len(selected),10):
        page=selected[start:start+10]
        canvas=Image.new('RGB',(1200, len(page)*180+35),'white'); draw=ImageDraw.Draw(canvas)
        draw.text((10,10),'Original | Mascara semantica | Superposicion binaria (verde) - revision humana pendiente',fill='black')
        for j,r in enumerate(page):
            with Image.open(dataset/'JPEGImages'/r['image']) as f: im=f.convert('RGB')
            with Image.open(dataset/'SegmentationClass'/r['mask']) as f: mask=f.convert('RGB')
            a=np.asarray(mask); foreground=np.any(a != 0,axis=2)
            overlay=np.asarray(im).copy()
            overlay[foreground]=(0.55*overlay[foreground]+0.45*np.array([0,255,0])).astype(np.uint8)
            for col, img in enumerate([im,mask,Image.fromarray(overlay)]):
                thumb=ImageOps.contain(img,(390,150),method=Image.Resampling.NEAREST if col==1 else Image.Resampling.LANCZOS)
                canvas.paste(thumb,(col*400+5, j*180+55))
            draw.text((5,j*180+36),f"{r['id']} | etiqueta: {float(r['foreground_fraction']):.2%}",fill='black')
        canvas.save(output/f'panel_{start//10+1:02d}.jpg',quality=90)
    return [r['id'] for r in selected]

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=PROJECT.parent)
    args=parser.parse_args(argv)
    config_path=PROJECT/'config/inicio_voc.json'
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    dataset=(args.data_root/cfg['dataset']).resolve()
    output=PROJECT/cfg['output_directory']; output.mkdir(parents=True,exist_ok=True)
    colors=palette(dataset/'labelmap.txt')
    images={p.stem:p for p in sorted((dataset/'JPEGImages').glob('*.jpg'))}
    masks={p.stem:p for p in sorted((dataset/cfg['mask_directory']).glob('*.png'))}
    splits={name:set((dataset/'ImageSets'/f'{name}.txt').read_text().split()) for name in ['train','val','trainval']}
    rows=[]; pixels=Counter(); presence=Counter(); issues=[]; groups=defaultdict(list)
    label_mismatches=[]
    for idx,key in enumerate(sorted(images.keys()|masks.keys())):
        r={'id':key,'image':images[key].name if key in images else '', 'mask':masks[key].name if key in masks else '',
           'original_split':'train' if key in splits['train'] else 'val' if key in splits['val'] else 'unassigned',
           'width':'','height':'','image_mode':'','mask_mode':'','foreground_fraction':'','classes':'','unknown_pixels':0,
           'image_sha256':'','mask_sha256':'','status':'ok'}
        try:
            if key in images:
                r['image_sha256']=sha(images[key]);groups[r['image_sha256']].append(key)
            if key in masks:r['mask_sha256']=sha(masks[key])
            if key not in images: r['status']='missing_image'
            elif key not in masks:r['status']='missing_mask'
            else:
                with Image.open(images[key]) as im, Image.open(masks[key]) as mask:
                    im.load(); mask.load()
                    r.update(width=im.width,height=im.height,image_mode=im.mode,mask_mode=mask.mode)
                    if im.size!=mask.size:r['status']='size_mismatch'
                    known,unknown,frac=mask_stats(mask,colors)
                    r.update(foreground_fraction=frac,classes='|'.join(sorted(k for k in known if k!='background')),unknown_pixels=sum(unknown.values()))
                    if unknown:r['status']='unknown_colors'
                    if frac==0 and not unknown:r['status']='empty_mask'
                    if r['status']=='ok':pixels.update(known);presence.update(k for k in known if k!='background')
                    if unknown:issues.append({'id':key,'issue':'unknown_colors','detail':dict(unknown)})
                xml=dataset/'bboxes/annotations'/f'{key}.xml'
                if xml.exists():
                    boxclasses={o.findtext('name') for o in ET.parse(xml).getroot().findall('object')}
                    maskclasses={k for k in known if k!='background'}
                    if boxclasses!=maskclasses:
                        label_mismatches.append({'id':key,'xml_classes':sorted(boxclasses),'mask_classes':sorted(maskclasses)})
                else:issues.append({'id':key,'issue':'missing_xml','detail':str(xml.relative_to(dataset))})
        except (OSError,ValueError,ET.ParseError) as e:
            r['status']='read_error';issues.append({'id':key,'issue':'read_error','detail':str(e)})
        rows.append(r)
        if (idx+1)%100==0:print(f'Pares revisados: {idx+1}/{len(images.keys()|masks.keys())}',flush=True)
    csv_write(output/'inventario.csv',rows,list(rows[0]))
    csv_write(output/'distribucion_clases.csv',[{'class':k,'pixels':v,'images':sum(r['status']=='ok' and float(r['foreground_fraction'])<1 for r in rows) if k=='background' else presence.get(k,0)} for k,v in sorted(pixels.items())],['class','pixels','images'])
    duplicates=[v for v in groups.values() if len(v)>1]
    cross=[g for g in duplicates if any(k in splits['train'] for k in g) and any(k in splits['val'] for k in g)]
    summary={'images':len(images),'masks':len(masks),'status_counts':dict(Counter(r['status'] for r in rows)),
      'pixel_counts_valid_pairs':dict(pixels),'class_image_counts_valid_pairs':dict(presence),
      'original_split_counts':{k:len(v) for k,v in splits.items()},'original_split_overlap':sorted(splits['train']&splits['val']),
      'split_ids_missing_image_or_mask':{k:sorted(s-(images.keys()&masks.keys())) for k,s in splits.items()},
      'exact_duplicate_groups':duplicates,'exact_duplicates_cross_original_splits':cross,
      'xml_mask_class_set_disagreements':label_mismatches,'issues':issues,
      'limits':['Casi duplicados y agrupacion por escena pendientes','ok indica integridad tecnica, no aprobacion semantica o espacial','No hay split final ni test congelado','EDA exploratorio del corpus; las decisiones futuras se ajustaran en train','La comparacion XML-mascara es por conjuntos de clases, no por instancia']}
    summary['visual_sample_ids']=make_panels(dataset,rows,output,cfg['seed'],cfg['sample_size'])
    (output/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=[{'path':str(p.relative_to(dataset)).replace('\\','/'),'sha256':sha(p)} for p in sorted([dataset/'labelmap.txt',*(dataset/'ImageSets').rglob('*.txt'),*(dataset/'bboxes/annotations').glob('*.xml')])]
    signature=hashlib.sha256(json.dumps([(r['id'],r['image_sha256'],r['mask_sha256']) for r in rows],separators=(',',':')).encode()).hexdigest()
    execution={'time_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),
      'packages':{'numpy':np.__version__,'Pillow':pillow_version},'seed':cfg['seed'],'config':cfg,'config_sha256':sha(config_path),
      'script_sha256':sha(Path(__file__)),'dataset_image_mask_signature':signature,'source_metadata':manifest,
      'command':'python scripts/inicio_voc.py --data-root <carpeta DEEP LEARNING>'}
    (output/'ejecucion.json').write_text(json.dumps(execution,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['images','masks','status_counts']},ensure_ascii=False),flush=True)
    print(f'Duplicados exactos: {len(duplicates)} grupos; entre splits: {len(cross)}; discrepancias de clases XML/mascara: {len(label_mismatches)}',flush=True)
    print(f'Resultados: {output}',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
