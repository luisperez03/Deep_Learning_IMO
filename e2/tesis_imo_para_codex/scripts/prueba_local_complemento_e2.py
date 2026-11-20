"""Prueba local de punta a punta del complemento GPU, sin el ZIP cerrado real ni checkpoints.

1. Arma un ZIP cerrado SIMULADO con la misma estructura que paquete_test_e2_cerrado.zip: los archivos cubiertos por
   config/protocolo_test_e2_cerrado.json (tomados de este proyecto y verificados contra esos hashes), el protocolo,
   las imágenes de test (de --voc-root) y sus máscaras binarias (derivadas y verificadas contra el CSV).
2. Corre empaquetar_complemento_gpu_e2.py con --local-test, tomando train/val de --voc-root.
3. Extrae el paquete, verifica PACKAGE_MANIFEST_COMPLEMENTO.json y corre DENTRO del paquete extraído la prueba de humo
   en CPU (modelos sin entrenar), una segunda vez para comprobar la reanudación, el resumidor y la detección de un
   JSON alterado.
El SHA del ZIP simulado no es el oficial; eso se prueba solo en Colab con el ZIP real.
"""
import argparse,csv,io,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path
SCRIPTS=Path(__file__).resolve().parent;PROJECT=SCRIPTS.parent;PREFIX='tesis_imo_para_codex/'
sys.path.insert(0,str(SCRIPTS))
import empaquetar_complemento_gpu_e2 as E

def mock_closed_zip(voc_root,dest):
    protocol_bytes=(PROJECT/'config/protocolo_test_e2_cerrado.json').read_bytes();protocol=json.loads(protocol_bytes)
    payload={}
    for rel,h in protocol['files'].items():
        data=(PROJECT/rel).read_bytes()
        if E.sha_bytes(data)!=h:raise RuntimeError('El proyecto ya no coincide con el protocolo cerrado: '+rel)
        payload[PREFIX+rel]=data
    payload[PREFIX+'config/protocolo_test_e2_cerrado.json']=protocol_bytes
    rows=list(csv.DictReader(io.StringIO(payload[PREFIX+'resultados/preentrenamiento/test.csv'].decode('utf-8-sig'))))
    for row in rows:
        img=(Path(voc_root)/row['image_path']).read_bytes();mask=E.derive_mask(voc_root,row)
        if E.sha_bytes(img)!=row['image_sha256'] or E.sha_bytes(mask)!=row['binary_mask_sha256']:raise RuntimeError('Test local distinto: '+row['id'])
        payload[row['image_path']]=img;payload[PREFIX+row['binary_mask_path']]=mask
    manifest={n:E.sha_bytes(d) for n,d in payload.items()}
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for n,d in sorted(payload.items()):z.writestr(n,d)
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2))
    return len(rows)

def run(cmd,cwd):
    print('>',' '.join(map(str,cmd)),flush=True);t0=time.perf_counter()
    subprocess.run([str(c) for c in cmd],cwd=cwd,check=True);return time.perf_counter()-t0

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--voc-root',required=True,type=Path,help='Carpeta que contiene VOC2012/');ap.add_argument('--work',required=True,type=Path)
    ap.add_argument('--limit',type=int,default=3);ap.add_argument('--models',nargs='*')
    args=ap.parse_args();w=args.work
    if w.exists():shutil.rmtree(w)
    w.mkdir(parents=True);report={}
    report['test_images_mock']=mock_closed_zip(args.voc_root,w/'cerrado_simulado.zip')
    E.main(['--closed-zip',str(w/'cerrado_simulado.zip'),'--local-test','--images-root',str(args.voc_root),'--derive-masks',
            '--output',str(w/'paquete_complemento_gpu_e2.zip')])
    base=w/'extraido'
    with zipfile.ZipFile(w/'paquete_complemento_gpu_e2.zip') as z:
        for n in z.namelist():
            if not (base/n).resolve().is_relative_to(base.resolve()):raise RuntimeError('Ruta inválida '+n)
        z.extractall(base)
    man=json.loads((base/'PACKAGE_MANIFEST_COMPLEMENTO.json').read_text(encoding='utf-8'))
    bad=[n for n,h in man['files'].items() if E.sha_file(base/n)!=h];report['manifest_mismatches']=bad
    if bad:raise RuntimeError(f'{len(bad)} archivos no coinciden tras extraer')
    proj=base/'tesis_imo_para_codex';out=w/'diagnosticos_cpu';py=sys.executable
    cmd=[py,'-u','scripts/prueba_cpu_diagnosticos_e2.py','--output',out,'--limit',args.limit,'--ig-steps','2']
    if args.models:cmd+=['--models',*args.models]
    report['first_pass_seconds']=run(cmd,proj)
    report['resume_pass_seconds']=run(cmd,proj)
    report['summary_seconds']=run([py,'-u','scripts/resumir_diagnosticos_gpu_e2.py','--output',out],proj)
    summary=json.loads((out/'RESUMEN_DIAGNOSTICOS.json').read_text(encoding='utf-8'))
    report['cpu_times']=json.loads((out/'tiempos_prueba_cpu.json').read_text(encoding='utf-8'))
    report['summary']={'status':summary['status'],'clean_models':[c['model'] for c in summary['clean']],
        'robustness_curves':len(summary['robustness']),'ensembles':[e['model'] for e in summary['ensembles']],
        'interpretability':[i['model'] for i in summary['interpretability']],
        'conformal_label_coverage':{e['model']:e['conformal']['mondrian_by_class']['label']['coverage'] for e in summary['ensembles']},
        'missing':summary['missing']}
    victim=sorted(out.glob('fcn_seed11_noise_*.json'))[0];data=json.loads(victim.read_text(encoding='utf-8'))
    data['means']['Dice']=0.999;victim.write_text(json.dumps(data),encoding='utf-8')
    tampered=subprocess.run([py,'scripts/resumir_diagnosticos_gpu_e2.py','--output',str(out)],cwd=proj,capture_output=True,text=True)
    report['tamper_detected']=tampered.returncode!=0 and 'corrupto' in tampered.stderr
    (w/'INFORME_PRUEBA_LOCAL.json').write_text(json.dumps(report,indent=1,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=1,ensure_ascii=False))
    if not report['tamper_detected']:raise RuntimeError('El resumidor no detectó el JSON alterado')

if __name__=='__main__':main()
