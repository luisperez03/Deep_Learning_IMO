"""Cuadernos ordenados de entrenamiento y recepción, con destinos nuevos y verificaciones."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

from crear_colab_original_e3 import celda
import crear_colab_semillas_e3 as rivals

E3 = Path(__file__).resolve().parents[1]
ROOT_DRIVE = '/content/drive/MyDrive/tesis_imo_e3_secuencia_v4'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e2', type=Path, required=True)
    a = ap.parse_args()
    base = json.loads((a.e2 / 'colab/Benchmark_E2_Colab_v2.ipynb').read_text(encoding='utf-8'))
    dest = E3 / 'training/colab/secuencia_v4'
    dest.mkdir(parents=True, exist_ok=True)
    for family in ['fcn', 'unet', 'segnet']:
        source = a.e2 / f'resultados/busqueda_arquitectura_e2/fase2/{family}_a0498_s11/config.json'
        (E3 / f'config/plantilla_{family}_a0498.json').write_bytes(source.read_bytes())
    files = ['src/__init__.py', 'src/geometria_at.py', 'src/original_model.py',
             'training/entrenar_original.py', 'training/entrenar_compactas_e3.py',
             'tests/test_original_model.py', 'tests/conftest.py', 'experiments/auditar_campana.py',
             'training/compactas_e2.py', 'experiments/evaluar_campana.py']
    files += [str(p.relative_to(E3)).replace('\\', '/') for p in sorted((E3 / 'config').glob('hpo_original_t*.json'))]
    files += [f'config/plantilla_{f}_a0498.json' for f in ['fcn', 'unet', 'segnet']]
    content = {f: (E3 / f).read_text(encoding='utf-8') for f in files}
    hashes = {f: hashlib.sha256(s.encode()).hexdigest() for f, s in content.items()}
    historical = (E3.parent / 'resultados/geometria_sintetica.json').read_text()
    manifest = dict(version='secuencia_v4', opencv='4.11.0.86', archivos=hashes,
                    nota='Fijación operativa; no acredita tag Git ni cumplimiento científico de E3')
    (dest / 'MANIFIESTO.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def save(name, cells):
        nb = copy.deepcopy(base)
        nb['cells'] = cells
        for c in cells:
            if c['cell_type'] == 'code': ast.parse(''.join(c['source']))
        (dest / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')

    def setup(title, folder, intro, full_env=False):
        cells = [celda('markdown', f'# {title}\n\n{intro}\n\nEjecutar todas las celdas con GPU. '
                       'Requiere paquete_colab_e2_v2.zip. Salida nueva: `' + ROOT_DRIVE + '/' + folder +
                       '`. Repetir reanuda; no mezclar con el piloto v3. No se evalúan imágenes de test.\n')]
        for c in base['cells'][1:5]:
            c = copy.deepcopy(c)
            s = ''.join(c['source']).replace('/content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2', ROOT_DRIVE + '/' + folder)
            if "pip_install('--no-deps'" in s and not full_env:
                s = '\n'.join(line for line in s.splitlines() if not line.startswith("pip_install('--no-deps'")
                              and not line.startswith("pip_install('transformers")) + '\n'
                s = s.replace('import torch, torchvision, mamba_ssm;', 'import torch, torchvision;')
            c['source'] = s.splitlines(True)
            if c['cell_type'] == 'code': c['outputs'], c['execution_count'] = [], None
            cells.append(c)
        cells.append(celda('code', "pip_install('opencv-python-headless==4.11.0.86', 'pytest==8.3.5', "
                           "'pyclipper==1.3.0.post6', 'onnx==1.17.0', 'onnxruntime==1.20.1')\n"
                           "subprocess.run([PY, '-c', \"import cv2,numpy; assert cv2.__version__=='4.11.0'; "
                           "assert numpy.__version__=='1.26.4'; print(cv2.__version__,numpy.__version__)\"],check=True)\n"))
        cells.append(celda('code', 'CONTENIDO = ' + repr(content) + '\nFIRMAS = ' + repr(hashes) +
                           '\nHISTORICO = ' + repr(historical) + f"\nCAMPANA = Path({ROOT_DRIVE!r})\n" + '''
E3 = Path('/content/secuencia_e3_v4/e3')
sello = OUTPUT / 'firmas_secuencia.json'
if sello.exists() and json.loads(sello.read_text()) != FIRMAS:
    raise RuntimeError('La carpeta contiene otra versión; no mezclar')
for rel, texto in CONTENIDO.items():
    p = E3 / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(texto, encoding='utf-8')
    assert sha_file(p) == FIRMAS[rel]
sello.write_text(json.dumps(FIRMAS, indent=2))
ref = E3.parent / 'resultados/geometria_sintetica.json'
ref.parent.mkdir(exist_ok=True)
ref.write_text(HISTORICO)
(OUTPUT / 'environment.freeze.txt').write_text(subprocess.check_output(
    [sys.executable, '-m', 'uv', 'pip', 'freeze', '--python', PY], text=True))
def verificar_original(semillas):
    for arm in ['dbnet', 'dbnet_at']:
        for seed in semillas:
            folder = CAMPANA / 'original' / f'unetat_{arm}_seed{seed}'
            r = json.loads((folder / 'result.json').read_text())
            assert r['completed'] and r['seed']==seed and r['variante']==arm
            assert sha_file(folder / r['best_file']) == r['best_sha256']
def entrenar(semillas):
    subprocess.run([PY, str(E3 / 'training/entrenar_original.py'), '--output', str(OUTPUT),
        '--variante', 'dbnet', 'dbnet_at', '--seeds', *map(str, semillas),
        '--voc', str(BASE), '--e2-scripts', str(PROJECT / 'scripts')], check=True)
'''))
        return cells

    cells = setup('01 · Sanidad y semilla 11 del original', 'original',
                  'Reinicia ambos brazos con OpenCV 4.11. Primero exige que pase la batería de pruebas. '
                  'Una prueba de equivalencia con el E2 histórico se omite porque falta su fuente; queda declarada.')
    cells.append(celda('code', '''env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
subprocess.run([PY, '-m', 'pytest', str(E3 / 'tests/test_original_model.py'), '-q', '-rs'], env=env, check=True)
entrenar([11])
verificar_original([11])
print('Etapa 01 completa: semilla 11 verificada para ambos brazos')
'''))
    save('01_Original_semilla11.ipynb', cells)
    for number, seeds, before in [('02', [23,37,53,71], [11]), ('03', [89,97,101,103,107], [11,23,37,53,71])]:
        cells = setup(number + ' · Ampliar original', 'original', 'Continúa con semillas ' + str(seeds) + '. Mantiene la receta fijada, sin ajustarla por resultados.')
        cells.append(celda('code', f'verificar_original({before!r})\nentrenar({seeds!r})\nverificar_original({before+seeds!r})\n'))
        save(number + '_Original_semillas.ipynb', cells)
    cells = setup('04 · Búsqueda exploratoria original', 'hpo', 'Cuatro recetas, diez épocas como máximo y semilla 11. Exploratoria: no sustituye los resultados de la receta principal.')
    cells.append(celda('code', '''verificar_original([11,23,37,53,71,89,97,101,103,107])
for cfg in sorted((E3 / 'config').glob('hpo_original_t*.json')):
    subprocess.run([PY, str(E3 / 'training/entrenar_original.py'), '--output', str(OUTPUT / cfg.stem),
        '--variante', 'dbnet_at', '--seeds', '11', '--config', str(cfg),
        '--voc', str(BASE), '--e2-scripts', str(PROJECT / 'scripts')], check=True)
'''))
    save('04_HPO_original.ipynb', cells)
    cells = setup('05 · Cinco semillas nuevas de compactas a0498', 'compactas',
                  'FCN, U-Net y SegNet: 89,97,101,103,107. Usa las recetas de los archivos originales recuperados; no repite las primeras cinco semillas.')
    supplemental = ['scripts/busqueda_arquitectura_e2.py', 'scripts/complemento_entreno_e2.py',
                    'config/busqueda_arquitectura_e2.json', 'config/arquitecturas_compactas_e2.csv']
    extra = {p: (a.e2 / p).read_text(encoding='utf-8') for p in supplemental}
    cells.append(celda('code', 'FUENTES_E2 = ' + repr(extra) + '''
for rel, texto in FUENTES_E2.items():
    p = PROJECT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(texto, encoding='utf-8')
subprocess.run([PY, str(E3 / 'training/entrenar_compactas_e3.py'), '--e2', str(PROJECT),
                '--output', str(OUTPUT)], check=True)
'''))
    save('05_Compactas_a0498.ipynb', cells)
    rivals.DRIVE_SALIDA = ROOT_DRIVE + '/rivales'
    for num, session in [('06','A'),('07','B'),('08','C')]:
        nb = rivals.generar(base, session, rivals.SESIONES[session])
        save(f'{num}_Rivales_{session}.ipynb', nb['cells'])
    source = (E3 / 'experiments/auditar_campana.py').read_text(encoding='utf-8')
    compact_hashes = json.loads((E3 / 'config/hashes_compactas.json').read_text())
    cells = [celda('markdown', '# 09 · Verificar y devolver resultados\n\nUsa CPU. No entrena ni evalúa test. '
                  'Verifica las 120 corridas principales esperadas, incluyendo las históricas. '
                  'Los resultados precalculados o archivos ausentes se reportan; no se inventan.')]
    cells.append(celda('code', f'''from google.colab import drive, files
drive.mount('/content/drive')
from pathlib import Path
import json, hashlib, zipfile, subprocess, sys
CAMPANA = Path({ROOT_DRIVE!r})
BENCHMARK = Path('/content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2')
COMPACTAS = Path('/content/compactas_e3_recibidas')
ZIP = Path('/content/drive/MyDrive/compactas_a0498_e3.zip')
assert BENCHMARK.is_dir(), 'Falta la carpeta histórica que confirmaste en Drive'
assert ZIP.is_file(), 'Sube compactas_a0498_e3.zip a MyDrive'
COMPACTAS.mkdir(exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    for name in z.namelist():
        assert (COMPACTAS/name).resolve().is_relative_to(COMPACTAS)
    z.extractall(COMPACTAS)
ESPERADOS = {compact_hashes!r}
for run, expected in ESPERADOS.items():
    assert hashlib.sha256((COMPACTAS/'fase2'/run/'best.pt').read_bytes()).hexdigest()==expected
'''))
    cells.append(celda('code', 'SCRIPT = ' + repr(source) + '''
p = Path('/content/auditar_campana.py'); p.write_text(SCRIPT, encoding='utf-8')
r = subprocess.run([sys.executable,str(p),'--campana',str(CAMPANA),
    '--benchmark',str(BENCHMARK),'--compactas',str(COMPACTAS)])
assert r.returncode in [0,2], 'Error del auditor'
print((CAMPANA/'estado_campana.json').read_text())
archive = Path('/content/resultados_e3_secuencia_v4.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in CAMPANA.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.csv','.txt']:
            z.write(p,str(p.relative_to(CAMPANA)))
files.download(str(archive))
print('Los pesos permanecen en Drive. Devuelve este ZIP para integración y revisión antes de test.')
'''))
    save('09_Verificar_y_devolver.ipynb', cells)
    receipt = copy.deepcopy(cells[1])
    cells = setup('10 · Evaluación reproducible de validación', 'evaluacion_validacion',
                  'Solo ejecutar cuando 09 informe 120/120 verificadas. Recalcula Dice e IoU por imagen '
                  'con los pesos elegidos. No usa test. Puede tardar: admite reanudación por corrida.', full_env=True)
    cells.append(receipt)
    cells.append(celda('code', '''subprocess.run([PY,str(E3/'experiments/auditar_campana.py'),
    '--campana',str(CAMPANA),'--benchmark',str(BENCHMARK),'--compactas',str(COMPACTAS)],check=True)
subprocess.run([PY,str(E3/'experiments/evaluar_campana.py'),'--e2',str(PROJECT),'--e3',str(E3),
    '--voc',str(BASE),'--campana',str(CAMPANA)],check=True)
archive=Path('/content/evaluacion_validacion_e3_v4.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in (CAMPANA/'evaluacion_validacion').glob('*'):
        if p.is_file():z.write(p,p.name)
files.download(str(archive))
print('Devuelve este ZIP para contraste con E2 y revisión estadística antes de test.')
'''))
    save('10_Evaluacion_validacion.ipynb', cells)
    print(dest)


if __name__ == '__main__':
    main()
