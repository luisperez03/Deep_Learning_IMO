"""Genera ZIP determinista y notebook Colab sin modificar los datos originales."""
import ast
import csv
import hashlib
import json
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
OUT = PROJECT/'colab'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    files = {}
    def add(path, expected=None):
        path = path.resolve()
        actual = digest(path)
        if expected and actual != expected:
            raise ValueError(f'Hash cambió: {path}')
        files[path.relative_to(ROOT).as_posix()] = (path, actual)
    prep = PROJECT/'resultados/preentrenamiento'
    frozen = json.loads((prep/'freeze_manifest.json').read_text())
    for name in ['train.csv', 'val.csv', 'normalization_train.json']:
        add(prep/name, frozen['artifacts'][name])
    for split in ['train', 'val']:
        with (prep/f'{split}.csv').open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                add(ROOT/row['image_path'], row['image_sha256'])
                add(PROJECT/row['binary_mask_path'], row['binary_mask_sha256'])
    for name in ['inicio_voc.py', 'preparar_base.py', 'dataset_ready.py',
                 'modelos_base.py', 'colab_runner.py']:
        add(PROJECT/'scripts'/name)
    manifest = {name: sha for name, (_, sha) in sorted(files.items())}
    archive = OUT/'imo_colab_train_val.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        def write(name, data):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
        for name, (path, _) in sorted(files.items()):
            write(name, path.read_bytes())
        write('package_manifest.json', json.dumps(manifest, sort_keys=True, indent=2).encode())
    archive_sha = digest(archive)
    escribir_cuaderno(archive_sha)
    report = {'zip_sha256': archive_sha, 'zip_bytes': archive.stat().st_size,
              'files': len(files), 'test_images_included': False,
              'notebook_code_cells_syntax_valid': True, 'executed_in_colab': False}
    (OUT/'verificacion_paquete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def escribir_cuaderno(archive_sha):
    cells = []
    def md(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})
    def code(text):
        ast.parse(text)
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(True),
                      'execution_count': None, 'outputs': []})
    md('''# IMO: segmentación binaria — prueba de recursos y piloto

Cada píxel es etiqueta o fondo. No se predice todavía el tipo IMO.
Tres adaptaciones pequeñas desde cero: FCN, SegNet y U-Net; no son réplicas
de sus artículos. Solo train (1015 imágenes) y val (218); test queda fuera.

**Ejecuta las celdas en orden.** Selecciona GPU en Entorno de ejecución →
Cambiar tipo de entorno de ejecución. No hace falta Colab de pago.
Si no hay GPU disponible, la comprobación se detendrá para revisar el presupuesto.
No publiques los datos ni el cuaderno con imágenes sin revisar las licencias.

Las celdas están preparadas, pero no han sido ejecutadas en Colab por el asistente.
El código registra el entorno real: la primera sesión establece sus versiones;
una reanudación rechaza cambios de entorno. No se promete igualdad bit a bit
entre equipos. Las advertencias de determinismo deben conservarse.''')
    code('''import os, sys, json, hashlib, zipfile, platform, subprocess
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch, numpy, PIL
print('Python:', platform.python_version(), 'PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No disponible')
assert torch.cuda.is_available(), 'Selecciona GPU; si no está disponible, no inicies el piloto todavía.'
print('VRAM total GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
''')
    md('''## 1. Leer el ZIP de la sesión de Colab
En el explorador de VS Code, haz clic derecho sobre `imo_colab_train_val.zip`
y selecciona **Upload to Colab**. Espera la confirmación antes de ejecutar esta
celda. Si falla la transferencia del ZIP, usa **Upload to Colab** sobre la
carpeta local `imo_zip_partes`. La celda reconstruye y verifica el paquete.
El SHA-256 verifica que es exactamente el paquete preparado. Los archivos se
extraen al disco temporal de Colab y se verifican uno por uno antes de usarlos.''')
    code(f'''import shutil
PACKAGE = Path('/content/imo_colab_train_val.zip')
firma_paquete = {archive_sha!r}
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
if not PACKAGE.is_file() or sha(PACKAGE) != firma_paquete:
    carpeta = PACKAGE.parent / 'imo_zip_partes'
    manifiesto = carpeta / 'partes.json'
    if not manifiesto.is_file():
        raise FileNotFoundError('En VS Code, usa Upload to Colab sobre la carpeta imo_zip_partes y espera a que termine.')
    partes = json.loads(manifiesto.read_text())
    temporal = PACKAGE.with_suffix('.tmp')
    with temporal.open('wb') as salida:
        for indice, firma in enumerate(partes):
            parte = carpeta / f'parte_{{indice:03d}}.bin'
            if not parte.is_file() or sha(parte) != firma:
                raise ValueError(f'Parte incompleta: {{parte.name}}. Vuelve a transferir la carpeta imo_zip_partes.')
            with parte.open('rb') as entrada:
                shutil.copyfileobj(entrada, salida)
    if sha(temporal) != firma_paquete:
        raise ValueError('El paquete reconstruido no coincide con este cuaderno.')
    temporal.replace(PACKAGE)
print('Leyendo paquete:', PACKAGE)
BASE = Path('/content/imo_workspace')
BASE.mkdir(exist_ok=True)
with zipfile.ZipFile(PACKAGE) as z:
    for item in z.infolist():
        target = (BASE/item.filename).resolve()
        assert target.is_relative_to(BASE.resolve()), 'Ruta insegura en el ZIP'
    z.extractall(BASE)
manifest = json.loads((BASE/'package_manifest.json').read_text())
for name, expected in manifest.items():
    assert sha(BASE/name) == expected, f'Archivo alterado: {{name}}'
PROJECT = BASE/'tesis_imo_para_codex'
sys.path.insert(0, str(PROJECT/'scripts'))
print('Integridad verificada:', len(manifest), 'archivos. Test no incluido.')
''')
    md('''## 2. Guardar resultados en la sesión de Colab
Los resultados quedan en `/content/IMO_colab_resultados_v1`. Descarga
`smoke.json` desde el explorador del servidor Colab en VS Code al terminar la
prueba. Si ejecutas el piloto, conserva también su carpeta en tu computador
antes de que se elimine la máquina temporal.''')
    code('''OUTPUT = Path('/content/IMO_colab_resultados_v1')
OUTPUT.mkdir(exist_ok=True)
from colab_runner import environment, smoke, run
env = {**environment(), 'python': platform.python_version()}
envpath = OUTPUT/'environment.json'
if envpath.exists():
    assert json.loads(envpath.read_text()) == env, 'Entorno distinto: revisar antes de reanudar.'
else:
    envpath.write_text(json.dumps(env, indent=2))
    (OUTPUT/'requirements_observed.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
(OUTPUT/'package_sha256.txt').write_text(sha(PACKAGE))
print('Resultados en la sesión de Colab:', OUTPUT)
''')
    md('''## 3. Prueba corta: cinco pasos por modelo
Comprueba dimensiones, pérdida finita, retropropagación, memoria GPU y tiempo.
No mide calidad y descarta esos pesos. Si falla, conserva el error y no ejecutes
el piloto. El primer paso incluye inicializaciones: cinco pasos no bastan para
predecir con precisión el tiempo total ni el comportamiento de una sesión larga.''')
    code('''report = smoke(OUTPUT)
print(json.dumps(report, indent=2))
SMOKE_OK = all(row['steps'] == 5 for row in report['results'])
assert SMOKE_OK
''')
    md('''## 4. Piloto, activación explícita
Primero comparte `smoke.json` o la salida anterior para revisar el presupuesto.
Cambiar `EJECUTAR_PILOTO` a `True` inicia hasta 15 corridas: tres modelos y cinco
semillas comunes. Se usa una tasa fija 0.001 para estimar viabilidad y variación;
**no sustituye** la búsqueda propuesta de dos tasas ni el benchmark definitivo.
Máximo 30 épocas, paciencia 5, lote 1, resolución 256. Se selecciona checkpoint
por Dice medio entre grupos de validación; el padding se excluye de la pérdida
y las métricas. El test no está disponible en este cuaderno.
No reduzcas presupuesto solo para un modelo. Si cambia el diseño, crea una
versión nueva y documenta el cambio antes de comparar.''')
    code('''EJECUTAR_PILOTO = False
if EJECUTAR_PILOTO:
    assert globals().get('SMOKE_OK', False), 'Ejecuta primero la prueba corta.'
    results = []
    for seed in [11, 23, 37, 53, 71]:
        for model in ['fcn', 'segnet', 'unet']:
            results.append(run(model, seed, OUTPUT/'piloto', epochs=30, lr=0.001, patience=5))
    (OUTPUT/'pilot_summary.json').write_text(json.dumps(results, indent=2))
else:
    print('Piloto detenido intencionalmente: revisa primero tiempos y memoria.')
''')
    md('''## 5. Curvas y lectura de resultados
La pérdida de train mide ajuste; Dice de validación mide solapamiento en imágenes
no usadas para actualizar pesos. Se promedian primero imágenes por grupo y luego
grupos, para no dar peso excesivo a imágenes similares. Una curva de validación
que empeora mientras baja la pérdida puede indicar sobreajuste. No interpretar
variación entre cinco semillas como evidencia estadística definitiva.''')
    code('''import matplotlib.pyplot as plt
histories = sorted((OUTPUT/'piloto').glob('*/history.json'))
if not histories:
    print('Todavía no hay épocas del piloto guardadas.')
else:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for path in histories:
        rows = json.loads(path.read_text())
        axes[0].plot([r['epoch'] for r in rows], [r['train_loss'] for r in rows], label=path.parent.name)
        axes[1].plot([r['epoch'] for r in rows], [r['val_group_Dice'] for r in rows], label=path.parent.name)
    axes[0].set(title='Pérdida de entrenamiento', xlabel='Época', ylabel='BCE + Dice / 2')
    axes[1].set(title='Dice por grupo en validación', xlabel='Época', ylabel='Dice', ylim=(0, 1))
    axes[1].legend(fontsize=6, bbox_to_anchor=(1.02, 1))
    fig.tight_layout(); fig.savefig(OUTPUT/'curvas_piloto.png', dpi=160); plt.show()
''')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'colab': {'name': '02_IMO_Colab.ipynb'},
        'accelerator': 'GPU'}, 'cells': cells}
    for i, cell in enumerate(cells):
        cell['id'] = f'imo-cell-{i:02d}'
    (OUT/'02_IMO_Colab.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
