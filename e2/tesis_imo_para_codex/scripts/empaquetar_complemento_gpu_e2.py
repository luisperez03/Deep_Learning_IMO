"""Arma paquete_complemento_gpu_e2.zip a partir del ZIP de test cerrado, sin tocar el protocolo cerrado.

No reescribe config/protocolo_test_e2_cerrado.json ni ejecuta preparar_test_e2.py. Hace esto:
1. Verifica el SHA256 del ZIP cerrado (el que fija colab/Evaluacion_Final_E2_Colab.ipynb), su
   PACKAGE_MANIFEST.json y cada archivo cubierto por los hashes del protocolo cerrado.
2. Copia byte a byte todas las entradas del ZIP cerrado.
3. Agrega las imágenes y máscaras binarias de train/val listadas en los CSV del propio ZIP cerrado,
   verificadas contra image_sha256 y binary_mask_sha256. Fuentes, en este orden: el ZIP de entrenamiento
   paquete_colab_e2_v2.zip (--trainval-zip), una carpeta de datos (--images-root / --masks-dir) o, para las
   máscaras, derivarlas de SegmentationClass (--derive-masks; solo se aceptan si el PNG resultante coincide byte
   a byte con el hash del CSV).
4. Agrega el protocolo complementario y los scripts de diagnóstico (desde --source, por defecto este proyecto).
   Ninguno puede pisar una entrada del ZIP cerrado con contenido distinto.
5. Escribe PACKAGE_MANIFEST_COMPLEMENTO.json (hash de cada entrada y procedencia), relee el ZIP y deja .sha256.

Solo usa la biblioteca estándar, salvo --derive-masks (numpy y Pillow). Corre igual en Windows y en Colab.
"""
import argparse,csv,hashlib,io,json,os,sys,zipfile
from pathlib import Path

PREFIX='tesis_imo_para_codex/'
CLOSED_ZIP_SHA256='ce8b6ecc48df618e59ed4a4de86c693731e4434857830921f40eeddd8ef6076c'
TRAINVAL_ZIP_SHA256='954f4fe5663d9fedd676dcee4f1f7fb9086a2a7f5c1907ce902b5df4c109cb49'
COMPLEMENT=['config/complementos_e2_20261120.json','scripts/metricas_diagnostico_e2.py','scripts/diagnosticos_gpu_e2.py',
            'scripts/resumir_diagnosticos_gpu_e2.py','scripts/empaquetar_complemento_gpu_e2.py',
            'scripts/prueba_cpu_diagnosticos_e2.py','tests/test_metricas_diagnostico_e2.py']

def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read_closed(path,expected,local_test):
    actual=sha_file(path)
    if actual!=expected:
        if not local_test:raise RuntimeError(f'ZIP cerrado distinto: {actual} != {expected}')
        print('AVISO prueba local: ZIP cerrado sin el hash oficial',actual,flush=True)
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise RuntimeError('CRC inválido en el ZIP cerrado')
        entries={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
    manifest=json.loads(entries['PACKAGE_MANIFEST.json'])
    for name,h in manifest.items():
        if sha_bytes(entries[name])!=h:raise RuntimeError('Manifiesto cerrado no coincide: '+name)
    protocol=json.loads(entries[PREFIX+'config/protocolo_test_e2_cerrado.json'])
    for rel,h in protocol['files'].items():
        if PREFIX+rel not in entries or sha_bytes(entries[PREFIX+rel])!=h:raise RuntimeError('Archivo del protocolo cerrado alterado o ausente: '+rel)
    return entries,protocol,actual

def palette(text):
    colors=[]
    for line in text.splitlines():
        if line.strip() and not line.startswith('#'):colors.append(tuple(map(int,line.split(':')[1].split(','))))
    return colors

def derive_mask(images_root,row):
    """Misma conversión que preparar_base.binary_mask + Image.save; solo vale si el hash coincide."""
    import numpy as np
    from PIL import Image
    voc=Path(images_root)/'VOC2012';colors=palette((voc/'labelmap.txt').read_text(encoding='utf-8-sig'))
    with Image.open(Path(images_root)/row['original_mask_path']) as im:rgb=np.asarray(im.convert('RGB'))
    packed=rgb[:,:,0].astype(np.uint32)*65536+rgb[:,:,1].astype(np.uint32)*256+rgb[:,:,2]
    allowed=[r*65536+g*256+b for r,g,b in colors]
    if not np.isin(packed,allowed).all():raise RuntimeError('Color desconocido en '+row['original_mask_path'])
    buf=io.BytesIO();Image.fromarray((packed!=0).astype(np.uint8)).save(buf,format='PNG');return buf.getvalue()

def trainval_files(entries,args):
    tv=None
    if args.trainval_zip:
        actual=sha_file(args.trainval_zip)
        if actual!=args.trainval_zip_sha256:raise RuntimeError(f'ZIP de entrenamiento distinto: {actual}')
        tv=zipfile.ZipFile(args.trainval_zip);names=set(tv.namelist())
    added={};origin={'zip':0,'carpeta':0,'derivada':0}
    try:
        for split in ['train','val']:
            rows=list(csv.DictReader(io.StringIO(entries[PREFIX+f'resultados/preentrenamiento/{split}.csv'].decode('utf-8-sig'))))
            for row in rows:
                items=[(row['image_path'],row['image_sha256'],'image'),(PREFIX+row['binary_mask_path'],row['binary_mask_sha256'],'mask')]
                for name,expected,kind in items:
                    data=None;how=None
                    if tv is not None and name in names:data=tv.read(name);how='zip'
                    elif kind=='image' and args.images_root and (Path(args.images_root)/name).is_file():
                        data=(Path(args.images_root)/name).read_bytes();how='carpeta'
                    elif kind=='mask' and args.masks_dir and (Path(args.masks_dir)/Path(name).name).is_file():
                        data=(Path(args.masks_dir)/Path(name).name).read_bytes();how='carpeta'
                    elif kind=='mask' and args.derive_masks and args.images_root:
                        data=derive_mask(args.images_root,row);how='derivada'
                    if data is None:raise FileNotFoundError(f'No encuentro {name} en ninguna fuente')
                    if sha_bytes(data)!=expected:raise RuntimeError(f'Hash distinto al CSV ({how}): {name}')
                    added[name]=data;origin[how]+=1
    finally:
        if tv is not None:tv.close()
    return added,origin

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--closed-zip',required=True,type=Path);ap.add_argument('--closed-zip-sha256',default=CLOSED_ZIP_SHA256)
    ap.add_argument('--trainval-zip',type=Path);ap.add_argument('--trainval-zip-sha256',default=TRAINVAL_ZIP_SHA256)
    ap.add_argument('--images-root',type=Path,help='Carpeta que contiene VOC2012/ (image_path del CSV es relativo a ella)')
    ap.add_argument('--masks-dir',type=Path,help='Carpeta con las máscaras binarias image_XXXX.png')
    ap.add_argument('--derive-masks',action='store_true');ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1],
                    help='Proyecto del que se toman los archivos del complemento')
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--local-test',action='store_true',
                    help='Acepta un ZIP cerrado sin el hash oficial; queda marcado en el manifiesto. Nunca para la corrida real.')
    args=ap.parse_args(argv)
    entries,protocol,closed_sha=read_closed(args.closed_zip,args.closed_zip_sha256,args.local_test)
    added,origin=trainval_files(entries,args)
    for rel in COMPLEMENT:added[PREFIX+rel]=(args.source/rel).read_bytes()
    for name,data in added.items():
        if name in entries and entries[name]!=data:raise RuntimeError('El complemento pisaría una entrada cerrada: '+name)
    payload={**entries,**added}
    manifest={'schema':1,'closed_zip_sha256':closed_sha,'closed_zip_official':closed_sha==CLOSED_ZIP_SHA256,
              'trainval_origin':origin,'trainval_zip_sha256':sha_file(args.trainval_zip) if args.trainval_zip else None,
              'closed_entries':len(entries),'added_entries':len([n for n in added if n not in entries]),
              'local_test':bool(args.local_test),'files':{n:sha_bytes(d) for n,d in sorted(payload.items())}}
    args.output.parent.mkdir(parents=True,exist_ok=True);temp=Path(str(args.output)+'.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(payload.items()):z.writestr(name,data)
        z.writestr('PACKAGE_MANIFEST_COMPLEMENTO.json',json.dumps(manifest,indent=1,ensure_ascii=False))
    os.replace(temp,args.output)
    with zipfile.ZipFile(args.output) as z:
        if z.testzip() is not None:raise RuntimeError('CRC inválido al releer')
        for name,h in manifest['files'].items():
            if sha_bytes(z.read(name))!=h:raise RuntimeError('Relectura distinta: '+name)
        for rel,h in protocol['files'].items():
            if sha_bytes(z.read(PREFIX+rel))!=h:raise RuntimeError('Hash cerrado no preservado: '+rel)
    digest=sha_file(args.output)
    Path(str(args.output)+'.sha256').write_text(f'{digest}  {args.output.name}\n',encoding='utf-8')
    print(json.dumps({'zip':str(args.output),'MiB':round(args.output.stat().st_size/2**20,1),'sha256':digest,
                      'closed_entries':manifest['closed_entries'],'added_entries':manifest['added_entries'],
                      'trainval_origin':origin,'closed_zip_official':manifest['closed_zip_official'],
                      'protocol_files_preserved':len(protocol['files'])},indent=1,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
