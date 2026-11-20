"""Empaqueta los datos y checkpoints para ejecutar el test exclusivamente en Colab."""
import ast
import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
OUT = PROJECT / 'colab'
PARTS = OUT / 'imo_test_partes'
PREP = PROJECT / 'resultados/preentrenamiento'


def main():
    files = [PROJECT / 'scripts' / f'{name}.py' for name in
             ('evaluar_test_final', 'modelos_base', 'inicio_voc', 'dataset_ready', 'preparar_base')]
    files += [PREP / name for name in ('train.csv', 'val.csv', 'test.csv',
                                      'normalization_train.json', 'freeze_manifest.json')]
    files.append(PROJECT / 'piloto_T4_final.tar.gz')
    for model, seed in [('fcn', 11), ('segnet', 11), ('unet', 11), ('fcn', 23)]:
        files.append(PROJECT / f'resultados/piloto_local_v1/{model}_seed{seed}_lr0.001/best.pt')
    with (PREP / 'test.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            files.extend([ROOT / row['image_path'], PROJECT / row['binary_mask_path']])
    PARTS.mkdir(exist_ok=True)
    hashes = {}
    with tempfile.TemporaryFile() as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
            for file in files:
                archive.write(file, file.relative_to(ROOT).as_posix())
        stream.seek(0)
        while chunk := stream.read(6 * 1024**2):
            name = f'parte_{len(hashes):03d}.bin'
            (PARTS / name).write_bytes(chunk)
            hashes[name] = hashlib.sha256(chunk).hexdigest()

    setup = f"""from pathlib import Path
import hashlib, json, shutil, subprocess, sys, zipfile
import torch

if not torch.cuda.is_available():
    raise RuntimeError('Selecciona la GPU de Colab antes de continuar.')
print('GPU:', torch.cuda.get_device_name(0))
parts = Path('/content/imo_test_partes')
base = Path('/content/imo_test_final')
base.mkdir(exist_ok=True)
expected = {hashes!r}
package = base / 'test.zip'
with package.open('wb') as output:
    for name, digest in expected.items():
        file = parts / name
        if not file.exists():
            raise FileNotFoundError(f'Falta {{file}}. Usa Upload to Colab sobre la carpeta imo_test_partes.')
        data = file.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'Transferencia incompleta: {{name}}')
        output.write(data)
with zipfile.ZipFile(package) as archive:
    for member in archive.infolist():
        if not (base / member.filename).resolve().is_relative_to(base.resolve()):
            raise ValueError(member.filename)
    archive.extractall(base)
project = base / 'tesis_imo_para_codex'
results = project / 'resultados/test_final_lr0001'
print('Paquete verificado. Test listo para ejecutar.')
"""
    evaluate = """if not results.exists():
    subprocess.run([sys.executable, '-u', str(project / 'scripts/evaluar_test_final.py')], check=True)
print((results / 'resumen.csv').read_text())
output = shutil.make_archive('/content/IMO_test_final', 'zip', results)
print('Resultados listos para descargar:', output)
"""
    cells = []
    for kind, source in [
        ('markdown', '# Test final de segmentación IMO\n\nEn VS Code selecciona Colab GPU T4. '
         'En el explorador local usa **Upload to Colab** sobre la carpeta completa `imo_test_partes`. '
         'Cuando termine la transferencia, ejecuta las dos celdas en orden.'),
        ('code', setup), ('code', evaluate),
        ('markdown', 'En el explorador del servidor Colab descarga `/content/IMO_test_final.zip` al PC antes de cerrar la sesión.')]:
        cell = dict(cell_type=kind, source=source.splitlines(True), metadata={}, id=f'test-{len(cells)}')
        if kind == 'code':
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
        'kernelspec': {'name': 'python3', 'language': 'python', 'display_name': 'Python 3'},
        'accelerator': 'GPU'})
    (OUT / '03_IMO_Test_Colab.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(files)} archivos empaquetados en {len(hashes)} partes.')


if __name__ == '__main__':
    main()
