"""Adquisición explícita de fuentes de autores; no descarga pesos ni datasets."""
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = Path(__file__).resolve().parents[1] / 'third_party/e2'
REPOS = {
    'pidnet': ('XuJiacong/PIDNet', ['models/pidnet.py', 'models/model_utils.py', 'LICENSE']),
    'transunet': ('Beckschen/TransUNet', ['networks/vit_seg_modeling.py', 'networks/vit_seg_configs.py', 'networks/vit_seg_modeling_resnet_skip.py', 'LICENSE']),
    'segnext': ('Visual-Attention-Network/SegNeXt', ['mmseg/models/backbones/mscan.py', 'mmseg/models/decode_heads/ham_head.py', 'LICENSE']),
    'vmunet': ('JCruan519/VM-UNet', ['models/vmunet/vmunet.py', 'models/vmunet/vmamba.py', 'LICENSE']),
}

def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'E2-research-audit'})
    return urllib.request.urlopen(request, timeout=60).read()

def main():
    lock_path = BASE / 'sources.lock.json'
    previous = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    lock = {}
    for key, (repo, files) in REPOS.items():
        commit = previous.get(key, {}).get('commit')
        if not commit:
            commit = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/main'))['sha']
        lock[key] = {'repo': repo, 'commit': commit, 'files': {}}
        for filename in files:
            data = fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{filename}')
            digest = hashlib.sha256(data).hexdigest()
            expected = previous.get(key, {}).get('files', {}).get(filename)
            if expected and expected != digest:
                raise RuntimeError(f'Hash incorrecto: {key}/{filename}')
            dest = BASE / key / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            lock[key]['files'][filename] = digest
        BASE.mkdir(parents=True, exist_ok=True)
        # Persistir cada repo para reanudar una descarga interrumpida.
        previous[key] = lock[key]
        lock_path.write_text(json.dumps(previous, indent=2), encoding='utf-8')
        print(key, commit, flush=True)

if __name__ == '__main__':
    main()
