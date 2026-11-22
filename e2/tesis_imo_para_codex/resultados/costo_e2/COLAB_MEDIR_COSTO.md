# Medir el costo en GPU de Colab (VM-UNet y comparación en GPU)

VM-UNet no corre en CPU (exige CUDA y `mamba-ssm`), así que sus FLOPs, memoria y latencia solo salen en GPU. El mismo
script mide las 10 redes con `--dispositivo cuda`, de modo que VM-UNet quede comparable con las demás en la misma GPU.
Úsese la **misma GPU que en el entrenamiento (NVIDIA L4)**; si Colab asigna otra, queda registrada en cada JSON y
no se mezclan.

Antes: el script y los resultados de esta carpeta deben estar en la rama `cierre-e2` ya publicada (el cuaderno clona el repo).

## Celdas

```python
# 1. Entorno de ejecución con GPU. Clonar el repo (público) y comprobar la GPU
import subprocess, sys, os
subprocess.run(['nvidia-smi'], check=True)
subprocess.run(['git', 'clone', '--depth', '1', '-b', 'cierre-e2',
                'https://github.com/luisperez03/Deep_Learning_IMO', '/content/repo'], check=True)
PROJECT = '/content/repo/e2/tesis_imo_para_codex'
```

```python
# 2. Entorno aislado igual al del entrenamiento (idéntico a la celda 3 de colab/Complemento_GPU_E2.ipynb)
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.6.9'], check=True)
ENV = Path('/content/imo-e2-env'); PY = str(ENV/'bin/python')
if not Path(PY).exists(): subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', str(ENV)], check=True)
def install(*a): subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PY, *a], check=True)
install('setuptools==75.8.0')
install('torch==2.5.1', 'torchvision==0.20.1', '--index-url', 'https://download.pytorch.org/whl/cu121')
install('-r', f'{PROJECT}/requirements-e2.txt')
install('transformers==4.46.3')   # VM-UNet lo importa (igual que colab/Complemento_GPU_E2.ipynb)
install('--no-deps', 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl')
install('--no-deps', 'https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl')
site = ENV/'lib/python3.11/site-packages'
dirs = [site/'torch/lib', *sorted((site/'nvidia').glob('*/lib'))]
os.environ['LD_LIBRARY_PATH'] = ':'.join([str(p) for p in dirs if p.is_dir()] + os.environ.get('LD_LIBRARY_PATH', '').split(':'))
```

```python
# 3. Medir las 10 redes en GPU (batch 1, 256x256, 5 de calentamiento, 30 repeticiones). Unos 5 minutos.
OUT = '/content/costo_e2_gpu'
subprocess.run([PY, '-u', f'{PROJECT}/scripts/medir_costo_e2.py', 'medir', '--dispositivo', 'cuda',
                '--hilos', '1', '--salida', OUT], check=True)
subprocess.run([PY, '-u', f'{PROJECT}/scripts/medir_costo_e2.py', 'tabla', '--salida', OUT], check=True)
```

```python
# 4. Descargar los JSON y la tabla (copiarlos a resultados/costo_e2/ del repo)
import shutil
shutil.make_archive('/content/costo_e2_gpu', 'zip', OUT)
from google.colab import files; files.download('/content/costo_e2_gpu.zip')
```

## Qué hacer con el resultado

1. Descomprimir en `resultados/costo_e2/` (los nombres `<modelo>_cuda_h1.json` no chocan con los de CPU).
2. `python scripts/medir_costo_e2.py tabla` (con el Python del entorno local) une CPU y GPU en `costo_e2.csv`/`.md`.
3. Si VM-UNet corre, sus FLOPs salen de `FlopCounterMode`, que **no cuenta** el escaneo selectivo de Mamba
   (`selective_scan_fn` es un kernel CUDA propio): el valor es una cota inferior y así debe declararse.
4. Actualizar `LECTURA.md` y, en `scripts/actualizar_checklist_costo.py`, el texto de la fila 44 (VM-UNet medido).
