"""Prueba sintética forward/backward/optimizer antes de comprometer corridas.
Cada modelo vive en un proceso separado para liberar memoria incluso ante fallo.
"""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil
import torch
from modelos_e2 import MODELS_E2, main_logits
from experimento_e2_colab import (PROJECT, atomic_json, compute_loss, digest,
    environment, make_optimizer, setup_seed, source_signature, verify_data, verify_sources)

def check_identity(name, model):
    names=[type(m).__name__ for m in model.modules()]
    if name=='segformer_b0' and not any(n=='Attention' for n in names): raise AssertionError('Falta atención MiT')
    if name=='segnext_t':
        assert len(model.encoder.block1)==3 and len(model.encoder.block3)==5
        assert 'LightHamHead' in names and 'NMF2D' in names
    if name=='pidnet_s': assert all(hasattr(model.model,k) for k in ['pag3','spp','dfm','seghead_d'])
    if name=='transunet_2d': assert len(model.model.transformer.encoder.layer)==12
    if name=='vmunet': assert 'SS2D' in names and 'GRU' not in names

def probe(name, config, device):
    setup_seed(11)
    started=time.perf_counter()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    model=MODELS_E2[name]().to(device)
    check_identity(name,model)
    model.train()
    batch=config['batch_size']+1  # Peor caso: último lote solitario unido al anterior.
    x=torch.randn(batch,3,256,256,device=device)
    y=torch.zeros(batch,1,256,256,device=device); y[:,:,70:160,90:190]=1
    valid=torch.ones_like(y,dtype=torch.bool); valid[:,:,:16]=False
    optimizer=make_optimizer(model,config['models'][name])
    output=model(x); logits=main_logits(output)
    assert logits.shape==y.shape and torch.isfinite(logits).all()
    loss=compute_loss(output,y,valid,config)
    assert torch.isfinite(loss)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g).item()>0 for g in grads)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(): assert main_logits(model(x[:1])).shape==(1,1,256,256)
    if device.type=='cuda': torch.cuda.synchronize()
    return {'status':'passed','parameters':sum(p.numel() for p in model.parameters()),
        'shape':list(logits.shape),'loss':loss.item(),'seconds':time.perf_counter()-started,
        'rss_after_bytes':psutil.Process().memory_info().rss,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
        'scope':'synthetic forward/backward/optimizer and eval; not dataset training'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--models',nargs='+',default=list(MODELS_E2),choices=list(MODELS_E2))
    p.add_argument('--output',required=True)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--worker',choices=list(MODELS_E2))
    p.add_argument('--timeout',type=int,default=1200)
    args=p.parse_args()
    config=json.loads((PROJECT/'config/experimento_e2.json').read_text())
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cpu' if args.cpu else 'cuda')
    if args.worker:
        try: result=probe(args.worker,config,device)
        except Exception as exc: result={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
        atomic_json(out/f'probe_{args.worker}.json',result)
        print(args.worker,result,flush=True)
        return
    verify_sources()
    report={'identity':{'source':source_signature(),'data':verify_data()},
        'environment':environment(),'config_hash':digest(config),'models':{}}
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('Selecciona runtime GPU')
    for name in args.models:
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',name,'--output',str(out)]
        if args.cpu: command.append('--cpu')
        result_file=out/f'probe_{name}.json'
        if result_file.exists(): result_file.unlink()
        try:
            process=subprocess.run(command,timeout=args.timeout)
            result=json.loads(result_file.read_text()) if process.returncode==0 and result_file.exists() else {'status':'failed','error':f'worker exit {process.returncode}'}
        except subprocess.TimeoutExpired: result={'status':'failed','error':'timeout; no aprobado'}
        report['models'][name]=result
        atomic_json(out/'preflight.json',report)
        print(name,result['status'],flush=True)
    failed=[n for n,r in report['models'].items() if r['status']!='passed']
    if failed: raise SystemExit('Preflight fallido: '+', '.join(failed))

if __name__=='__main__': main()
