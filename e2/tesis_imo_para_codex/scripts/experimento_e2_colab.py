"""Benchmark E2 v2. Train/val exclusivamente; identidad y checkpoints verificables.
No descarga pesos. No cambia modelos por fallos de dependencias.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import platform
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from dataset_ready import load_example, metrics, partition
from modelos_e2 import MODELS_E2, main_logits

PROJECT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def atomic_json(path, value):
    temp = Path(str(path) + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)

def atomic_torch(path, value):
    temp = Path(str(path) + '.tmp')
    with temp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)

def environment():
    packages = {p: importlib.metadata.version(p) for p in
        ['torch','torchvision','numpy','Pillow','segmentation-models-pytorch','timm','scipy','ml-collections','einops']}
    for p in ['mamba-ssm','causal-conv1d','triton','transformers']:
        try: packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: packages[p] = None
    return {'packages': packages, 'python':platform.python_version(), 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}

def setup_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def source_signature():
    paths = list((PROJECT/'scripts').glob('*.py')) + list((PROJECT/'third_party/e2').rglob('*.py'))
    paths += [PROJECT/'third_party/e2/sources.lock.json', PROJECT/'requirements-e2.txt']
    return digest({str(p.relative_to(PROJECT)): sha(p) for p in sorted(paths)})

def verify_sources():
    root = PROJECT/'third_party/e2'
    lock = json.loads((root/'sources.lock.json').read_text())
    for name, info in lock.items():
        for filename, expected in info['files'].items():
            if sha(root/name/filename) != expected:
                raise RuntimeError(f'Fuente modificada sin registrar: {name}/{filename}')

def verify_data():
    """Hashes reales de train/val; test sólo metadatos para detectar fuga de grupos."""
    base = PROJECT/'resultados/preentrenamiento'
    signature = {f: sha(base/f) for f in ['train.csv','val.csv','test.csv','normalization_train.json']}
    splits = {s: partition(s, allow_test=(s=='test')) for s in ['train','val','test']}
    groups = {s: {r['group_id'] for r in rows} for s,rows in splits.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if groups[a] & groups[b]: raise RuntimeError(f'Fuga de grupos entre {a}/{b}')
    for split in ['train','val']:
        if not splits[split]: raise RuntimeError(f'Partición vacía: {split}')
        for row in splits[split]:
            for path, expected in [(PROJECT.parent/row['image_path'],row['image_sha256']),
                                   (PROJECT/row['binary_mask_path'],row['binary_mask_sha256'])]:
                if sha(path) != expected: raise RuntimeError(f'Datos modificados: {path}')
    norm = json.loads((base/'normalization_train.json').read_text())
    if norm['size'] != 256: raise RuntimeError('Normalización no corresponde a 256')
    return digest(signature)

def batch_tensors(rows, device):
    loaded = [load_example(row) for row in rows]
    x = torch.from_numpy(np.stack([a[0] for a in loaded])).to(device)
    y = torch.from_numpy(np.stack([a[1] for a in loaded]).astype('float32'))[:,None].to(device)
    valid = torch.from_numpy(np.stack([a[2] for a in loaded]))[:,None].to(device)
    return x,y,valid

def binary_loss(logits, target, valid):
    if logits.shape != target.shape or valid.shape != target.shape: raise ValueError('Formas incompatibles')
    losses = []
    for z,t,v in zip(logits,target,valid):
        if not v.any(): raise ValueError('Imagen sin píxeles válidos')
        z,t = z[v],t[v]
        p = z.sigmoid()
        losses.append(.5*F.binary_cross_entropy_with_logits(z,t) + .5*(1-(2*(p*t).sum()+1e-6)/(p.sum()+t.sum()+1e-6)))
    return torch.stack(losses).mean()

def boundary_target(target, valid):
    hi = F.max_pool2d(target,3,1,1)
    lo = -F.max_pool2d(-target,3,1,1)
    # Erosión con fuera de imagen inválido: padding nunca produce bordes ficticios.
    padded = F.pad(valid.float(),(1,1,1,1),value=0)
    core = (-F.max_pool2d(-padded,3,1)) > .5
    return (hi-lo > 0).float(), core

def compute_loss(output, target, valid, config):
    loss = binary_loss(main_logits(output), target, valid)
    if isinstance(output, dict):
        edge, core = boundary_target(target,valid)
        loss = loss + config['pidnet_aux_weight']*binary_loss(output['aux'],target,valid)
        if core.any():
            loss = loss + config['pidnet_boundary_weight']*F.binary_cross_entropy_with_logits(output['boundary'][core],edge[core])
    return loss

def make_optimizer(model, cfg):
    args = {k:v for k,v in cfg.items() if k != 'optimizer'}
    return getattr(torch.optim,cfg['optimizer'])(model.parameters(),**args)

def training_batches(rows, seed, epoch, batch_size):
    order = list(rows); random.Random(seed*100000+epoch).shuffle(order)
    batches = [order[i:i+batch_size] for i in range(0,len(order),batch_size)]
    # No perder ni duplicar imágenes cuando el último lote contiene una sola.
    if len(batches)>1 and len(batches[-1])==1: batches[-2].extend(batches.pop())
    if len(batches[0])<2: raise ValueError('Se necesitan al menos dos ejemplos para BatchNorm')
    return batches

@torch.no_grad()
def evaluate_validation(model, rows, device):
    model.eval(); grouped=defaultdict(list); detail=[]
    # NMF aleatorio en SegNeXt: validación reproducible sin consumir RNG de train.
    devices=[torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(20261118)
        for row in rows:
            x,y,v=batch_tensors([row],device)
            probability=main_logits(model(x)).sigmoid()[0,0].cpu().numpy()
            scores=metrics(probability,y[0,0].cpu().numpy(),v[0,0].cpu().numpy())
            grouped[row['group_id']].append(scores['Dice'])
            detail.append({'id':row['id'],'group_id':row['group_id'],**scores})
    return float(np.mean([np.mean(values) for values in grouped.values()])),detail

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])

def train_model(name, seed, output, config, identity, device, stop_after=None):
    setup_seed(seed)
    dest=Path(output)/f'{name}_seed{seed}'; dest.mkdir(parents=True,exist_ok=True)
    spec={'name':name,'seed':seed,'config':config,'identity':identity,'environment':environment()}
    fingerprint=digest(spec)
    cfg_file=dest/'config.json'
    if cfg_file.exists():
        if json.loads(cfg_file.read_text())['fingerprint'] != fingerprint:
            raise RuntimeError(f'Corrida incompatible: {dest}. Usa una carpeta de experimento nueva.')
    else: atomic_json(cfg_file,{'fingerprint':fingerprint,**spec})
    last=dest/'last.pt'
    if (dest/'result.json').exists():
        result=json.loads((dest/'result.json').read_text())
        if result.get('fingerprint')!=fingerprint or not result.get('completed'): raise RuntimeError('Resultado incompatible')
        if not last.exists() or not (dest/result['best_file']).exists(): raise RuntimeError('Faltan checkpoints de corrida completada')
        if sha(last)!=result['last_sha256'] or sha(dest/result['best_file'])!=result['best_sha256']:
            raise RuntimeError('Checkpoint completado corrupto/modificado')
        return result
    model=MODELS_E2[name]().to(device)
    optimizer=make_optimizer(model,config['models'][name])
    start,best,stale,history,best_file=0,-1.,0,[],None
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['fingerprint']!=fingerprint: raise RuntimeError('Checkpoint incompatible')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start,best,stale,history,best_file=saved['epoch']+1,saved['best'],saved['stale'],saved['history'],saved['best_file']
        if not (dest/best_file).exists(): raise RuntimeError('Falta mejor checkpoint referenciado')
        restore_rng(saved['rng'])
    train_rows,val_rows=partition('train'),partition('val')
    for epoch in range(start,config['epochs']):
        if stale>=config['patience']: break
        model.train(); started=time.perf_counter(); loss_sum=0.; seen=0
        for rows in training_batches(train_rows,seed,epoch,config['batch_size']):
            x,y,v=batch_tensors(rows,device)
            optimizer.zero_grad(set_to_none=True)
            loss=compute_loss(model(x),y,v,config)
            if not torch.isfinite(loss): raise RuntimeError('Pérdida no finita')
            loss.backward(); optimizer.step()
            loss_sum+=loss.item()*len(rows); seen+=len(rows)
        score,details=evaluate_validation(model,val_rows,device)
        if not np.isfinite(score): raise RuntimeError('Validación no finita')
        if score>best:
            best,stale=score,0
            best_file=f'best_epoch_{epoch+1:03d}.pt'
            atomic_torch(dest/best_file,{'model':model.state_dict(),'epoch':epoch,'fingerprint':fingerprint,'validation':details})
        else: stale+=1
        history.append({'epoch':epoch+1,'train_loss':loss_sum/seen,'val_group_Dice':score,
            'seconds':time.perf_counter()-started})
        # last apunta a un best inmutable ya escrito: un corte no cambia su significado.
        atomic_torch(last,{'fingerprint':fingerprint,'epoch':epoch,'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'best':best,'stale':stale,'history':history,
            'best_file':best_file,'rng':rng_state()})
        atomic_json(dest/'history.json',history)
        for old in dest.glob('best_epoch_*.pt'):
            if old.name != best_file: old.unlink()
        print(f'{name} seed={seed} epoch={epoch+1} val={score:.5f}',flush=True)
        if stop_after is not None and epoch+1>=stop_after: return {'completed':False}
    result={'completed':True,'fingerprint':fingerprint,'model':name,'seed':seed,
        'best_val_group_Dice':best,'epochs_completed':len(history),'best_file':best_file,
        'parameters':sum(p.numel() for p in model.parameters()),
        'last_sha256':sha(last),'best_sha256':sha(dest/best_file)}
    atomic_json(dest/'result.json',result)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',required=True,choices=list(MODELS_E2))
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--output',required=True)
    parser.add_argument('--config',default=str(PROJECT/'config/experimento_e2.json'))
    parser.add_argument('--allow-local-output',action='store_true')
    args=parser.parse_args()
    output=Path(args.output).resolve()
    if not args.allow_local_output:
        import subprocess
        subprocess.run(['mountpoint','-q','/content/drive'],check=True)
        if not output.is_relative_to(Path('/content/drive/MyDrive')): raise RuntimeError('Salida debe estar en Drive')
    if not torch.cuda.is_available(): raise RuntimeError('Este comando de benchmark requiere GPU; ejecutar preflight CPU por separado')
    config=json.loads(Path(args.config).read_text())
    if (config['pretrained'] or config['size']!=256 or config['batch_size']<2
        or config['threshold']!=.5 or config['epochs']<1 or config['patience']<1
        or config['loss']!='0.5_BCEWithLogits_plus_0.5_softDice_per_image_valid'):
        raise ValueError('Configuración no soportada')
    verify_sources()
    identity={'source':source_signature(),'data':verify_data()}
    # No entrenar una red sin un forward/backward aprobado para esta firma/entorno.
    report=json.loads((output/'preflight.json').read_text())
    if report['identity']!=identity or report['environment']!=environment() or report['config_hash']!=digest(config):
        raise RuntimeError('Repetir preflight: código, datos, configuración o entorno cambiaron')
    for name in args.models:
        if report['models'].get(name,{}).get('status')!='passed': raise RuntimeError(f'Preflight pendiente/fallido: {name}')
    for name in args.models:
        for seed in args.seeds or config['seeds']:
            train_model(name,seed,output,config,identity,torch.device('cuda'))

if __name__=='__main__': main()
