"""Genera `colab/Original_E3.ipynb`: entrenamiento de los dos brazos del modelo original (dbnet y dbnet_at), 10 semillas.

Reutiliza del cuaderno del benchmark del E2 las celdas de Drive, paquete y entorno (mismo ZIP, mismo SHA-256, misma
receta), e incrusta el código del E3 (`src/geometria_at.py`, `src/original_model.py`, `training/entrenar_original.py`)
con el SHA-256 de cada archivo, para que lo que corre en Colab sea exactamente lo versionado en la etiqueta del
protocolo. Semillas del protocolo (§3): 11, 23, 37, 53, 71, 89, 97, 101, 103, 107. No usa test.

Uso: python crear_colab_original_e3.py --e2 <carpeta tesis_imo_para_codex>
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
E3 = AQUI.parent
ARCHIVOS = ['src/__init__.py', 'src/geometria_at.py', 'src/original_model.py', 'training/entrenar_original.py']
SEMILLAS = [11, 23, 37, 53, 71, 89, 97, 101, 103, 107]
SALIDA_DRIVE = '/content/drive/MyDrive/tesis_imo_e3_original_v1'


def celda(tipo, texto):
    if tipo == 'code':
        ast.parse(texto)
        return {'cell_type': 'code', 'metadata': {}, 'source': texto.splitlines(True), 'execution_count': None,
                'outputs': []}
    return {'cell_type': 'markdown', 'metadata': {}, 'source': texto.splitlines(True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e2', required=True)
    a = ap.parse_args()
    base = json.loads((Path(a.e2) / 'colab/Benchmark_E2_Colab_v2.ipynb').read_text(encoding='utf-8'))
    celdas = [celda('markdown', f"""# E3, bloque C: modelo original UNet-AT, dos brazos × 10 semillas

1. Entorno de ejecución → Cambiar tipo → **GPU L4**.
2. Ejecutar las celdas en orden. Escribe en `{SALIDA_DRIVE}` una carpeta por brazo y semilla. Si se cae la sesión,
   repetir las celdas: cada corrida se reanuda desde su última época completa.
3. Brazos: `dbnet` (la representación de DBNet sobre la U-Net compacta, base del mecanismo) y `dbnet_at` (el modelo
   original, con C1 y C2). Semillas: {', '.join(map(str, SEMILLAS))}.

No se usa test. Estimado (no medido en GPU): de 4 a 8 h en total, porque la validación de cada época aplica el
posproceso C2 en CPU.
""")]
    for c in base['cells'][1:5]:
        c = copy.deepcopy(c)
        texto = ''.join(c['source'])
        if c['cell_type'] == 'code' and "OUTPUT = Path('/content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2')" in texto:
            c['source'] = texto.replace('/content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2', SALIDA_DRIVE).splitlines(True)
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
        celdas.append(c)
    # geometria_at usa OpenCV (contornos y relleno de polígonos), que no está en requirements-e2.txt.
    celdas.append(celda('code', "pip_install('opencv-python-headless==4.10.0.84')\n"
                                "subprocess.run([PY, '-c', 'import cv2; print(cv2.__version__)'], check=True)\n"))
    contenido = {r: (E3 / r).read_text(encoding='utf-8') for r in ARCHIVOS}
    firmas = {r: hashlib.sha256(t.encode('utf-8')).hexdigest() for r, t in contenido.items()}
    celdas.append(celda('code', "E3 = Path('/content/e3')\nARCHIVOS = " + repr(contenido) + "\nFIRMAS = " + repr(firmas) + """
for rel, texto in ARCHIVOS.items():
    destino = E3 / rel
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding='utf-8')
    assert hashlib.sha256(destino.read_bytes()).hexdigest() == FIRMAS[rel], f'Archivo alterado: {rel}'
(OUTPUT / 'firmas_codigo_e3.json').write_text(json.dumps(FIRMAS, indent=2))
print('Código del E3 escrito y verificado:', len(ARCHIVOS), 'archivos')
"""))
    celdas.append(celda('markdown', '## Entrenar\nPrimero los dos brazos con la semilla 11 (permite revisar el tiempo por '
                                    'época); después las demás.\n'))
    celdas.append(celda('code', f"""SEMILLAS = {SEMILLAS!r}
def entrenar(semillas):
    subprocess.run([PY, str(E3 / 'training/entrenar_original.py'), '--output', str(OUTPUT),
                    '--variante', 'dbnet', 'dbnet_at', '--seeds', *map(str, semillas),
                    '--voc', str(BASE), '--e2-scripts', str(PROJECT / 'scripts')], check=True)
entrenar(SEMILLAS[:1])
"""))
    celdas.append(celda('code', 'entrenar(SEMILLAS[1:])\n'
                                "for p in sorted(OUTPUT.glob('*/result.json')):\n"
                                "    r = json.loads(p.read_text())\n"
                                "    print(p.parent.name, r.get('best_val_group_Dice'), r.get('epochs_completed'))\n"))
    # Búsqueda exploratoria del original con el presupuesto de la del E2 (protocolo v2, §2): brazo dbnet_at, 4
    # intentos, semilla 11, 10 épocas, paciencia 3. No cambia la configuración confirmatoria.
    hpo = {f.name: f.read_text(encoding='utf-8') for f in sorted((E3 / 'config').glob('hpo_original_t*.json'))}
    assert len(hpo) == 4, 'Faltan las configuraciones: correr configs_e3.py'
    celdas.append(celda('markdown', '## Búsqueda exploratoria (mismo presupuesto que la del E2)\nCuatro intentos del brazo '
                                    'dbnet_at, semilla 11, 10 épocas. Unos 30 a 60 minutos. Se corre al final.\n'))
    celdas.append(celda('code', "HPO = " + repr(hpo) + """
for nombre, texto in HPO.items():
    cfg = E3 / 'config' / nombre
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(texto, encoding='utf-8')
    subprocess.run([PY, str(E3 / 'training/entrenar_original.py'), '--output', str(OUTPUT / 'hpo' / cfg.stem),
                    '--variante', 'dbnet_at', '--seeds', '11', '--config', str(cfg),
                    '--voc', str(BASE), '--e2-scripts', str(PROJECT / 'scripts')], check=True)
for p in sorted((OUTPUT / 'hpo').glob('*/*/result.json')):
    print(p.parent.parent.name, json.loads(p.read_text()).get('best_val_group_Dice'))
"""))
    nb = copy.deepcopy(base)
    nb['cells'] = celdas
    destino = AQUI / 'colab/Original_E3.ipynb'
    destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(destino)
    print(json.dumps(firmas, indent=2))


if __name__ == '__main__':
    main()
