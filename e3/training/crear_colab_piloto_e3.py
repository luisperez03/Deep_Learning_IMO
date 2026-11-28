"""Genera piloto reanudable separado; no cambia arquitectura, pérdida ni receta."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

from crear_colab_original_e3 import ARCHIVOS, E3, celda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e2', required=True)
    args = ap.parse_args()
    base = json.loads((Path(args.e2) / 'colab/Benchmark_E2_Colab_v2.ipynb').read_text(encoding='utf-8'))
    cells = [celda('markdown', '''# E3: piloto de UNet-AT, dos épocas por brazo

Selecciona un entorno GPU y ejecuta todas las celdas. Este cuaderno termina después de dos épocas
de `dbnet` y dos de `dbnet_at`, semilla 11, usando train y validación completos. No lee imágenes de test.
No lanza las otras semillas ni la búsqueda de hiperparámetros.

Necesita **paquete_colab_e2_v2.zip**, ya existente, en MyDrive o en el panel de archivos.
No necesita los pesos compactos: el original se entrena desde cero.
La carpeta de salida es NUEVA: `/content/drive/MyDrive/tesis_imo_e3_original_piloto_v3`.
Es un destino que este cuaderno crea, no una carpeta recuperada de Daniel.

La pausa no cambia el máximo de 30 épocas ni la paciencia 5 de la receta. Guarda `last.pt` y
el mejor checkpoint en Drive. La reanudación requiere el mismo código, configuración y entorno.
El tiempo y las unidades de procesamiento solo se podrán estimar después de esta medición GPU.

**Estado de revisión:** hay una discrepancia sin resolver entre el oráculo geométrico histórico y
el código recuperado (rombo de 10 px, brazo DBNet: 0,9730 frente a 0,9823). Este cuaderno es un piloto
técnico de tiempo y reanudación; no demuestra superioridad ni autoriza considerar cerrado el protocolo.
La campaña confirmatoria debe esperar a que esa discrepancia se explique y se cierre la auditoría.
''')]
    output = '/content/drive/MyDrive/tesis_imo_e3_original_piloto_v3'
    for c in base['cells'][1:5]:
        c = copy.deepcopy(c)
        s = ''.join(c['source']).replace('/content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2', output)
        # El original no usa los kernels opcionales de VM-UNet.
        if "pip_install('--no-deps'" in s:
            s = '\n'.join(line for line in s.splitlines()
                          if not line.startswith("pip_install('--no-deps'") and
                          not line.startswith("pip_install('transformers")) + '\n'
            s = s.replace('import torch, torchvision, mamba_ssm;', 'import torch, torchvision;')
        c['source'] = s.splitlines(True)
        if c['cell_type'] == 'code':
            ast.parse(s)
            c['outputs'], c['execution_count'] = [], None
        cells.append(c)
    cells.append(celda('code', "pip_install('opencv-python-headless==4.10.0.84')\n"))
    sources = {r: (E3 / r).read_text(encoding='utf-8') for r in ARCHIVOS}
    hashes = {r: hashlib.sha256(s.encode()).hexdigest() for r, s in sources.items()}
    cells.append(celda('code', "E3 = Path('/content/e3_piloto')\nARCHIVOS = " + repr(sources) +
                       "\nFIRMAS = " + repr(hashes) + '''
manifest = OUTPUT / 'firmas_codigo_e3.json'
if manifest.exists() and json.loads(manifest.read_text()) != FIRMAS:
    raise RuntimeError('Hay otro código en esta carpeta de salida; no mezclar corridas.')
for rel, texto in ARCHIVOS.items():
    p = E3 / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(texto, encoding='utf-8')
    assert sha_file(p) == FIRMAS[rel]
manifest.write_text(json.dumps(FIRMAS, indent=2))
freeze = subprocess.check_output([sys.executable, '-m', 'uv', 'pip', 'freeze', '--python', PY], text=True)
(OUTPUT / 'environment.freeze.txt').write_text(freeze)
print('Código verificado. Destino nuevo:', OUTPUT)
'''))
    cells.append(celda('code', '''import time
inicio = time.perf_counter()
comando = [PY, str(E3 / 'training/entrenar_original.py'), '--output', str(OUTPUT),
           '--variante', 'dbnet', 'dbnet_at', '--seeds', '11',
           '--voc', str(BASE), '--e2-scripts', str(PROJECT / 'scripts')]
subprocess.run(comando + ['--pause-after-epoch', '2'], check=True)
duracion = time.perf_counter() - inicio
'''))
    cells.append(celda('code', '''import statistics
resumen = {'estado': 'piloto_pausado_no_resultado_final', 'segundos_celda': duracion,
           'test_evaluado': False, 'brazos': {}}
for brazo in ['dbnet', 'dbnet_at']:
    carpeta = OUTPUT / f'unetat_{brazo}_seed11'
    hist = json.loads((carpeta / 'history.json').read_text())
    assert len(hist) >= 2, 'Piloto incompleto'
    assert (carpeta / 'last.pt').is_file()
    promedio = statistics.mean(h['seconds'] for h in hist)
    resumen['brazos'][brazo] = {'epocas': len(hist), 'segundos_medios_epoca': promedio,
        'segundos_medios_entrenamiento': statistics.mean(h['seconds_train'] for h in hist),
        'dice_validacion_ultima_epoca': hist[-1]['val_group_Dice'],
        'checkpoint_sha256': sha_file(carpeta / 'last.pt')}
resumen['horas_proyectadas_20_corridas_30_epocas'] = sum(
    b['segundos_medios_epoca'] * 30 * 10 for b in resumen['brazos'].values()) / 3600
resumen['nota_proyeccion'] = ('Extrapolación de dos épocas, no promesa: excluye preparación, HPO, '
    'rivales y cambios de costo del posproceso; early stopping puede reducir el tiempo.')
(OUTPUT / 'resumen_piloto.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
print(json.dumps(resumen, indent=2, ensure_ascii=False))
from google.colab import files
files.download(str(OUTPUT / 'resumen_piloto.json'))
'''))
    cells.append(celda('markdown', '''## Al terminar

Guarda `resumen_piloto.json` y anota las unidades de Colab antes y después. Desconecta el entorno GPU.
Los checkpoints quedan en Drive. Comparte el resumen para decidir el presupuesto de la campaña completa.

Para reanudar únicamente estos dos brazos de semilla 11, en la MISMA sesión y tras revisar el presupuesto,
agrega una celda con `subprocess.run(comando, check=True)`. Esto quita la pausa y conserva la receta.
Si cambió el entorno, el entrenador verifica su identidad y rechaza una reanudación incompatible.
No uses el cuaderno histórico `Original_E3.ipynb` para reanudar esta carpeta: tiene otro código y destino.
'''))
    nb = copy.deepcopy(base)
    nb['cells'] = cells
    dest = E3 / 'training/colab/Piloto_Original_E3_v3.ipynb'
    dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(dest)


if __name__ == '__main__':
    main()
