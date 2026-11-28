"""Genera `colab/Exportar_ONNX_E3.ipynb`: exportación de las 10 redes del E2 a ONNX con prueba de ida y vuelta.

Reutiliza del cuaderno del benchmark del E2 las celdas de Drive, paquete y entorno (mismo ZIP y mismo SHA-256), y
agrega: instalación de onnx, onnxruntime y fvcore en ese entorno; el script `exportar_onnx.py` incrustado (se
escribe en el disco de Colab desde la celda, con su SHA-256 verificado); y la corrida.

No usa test. Corre en GPU (VM-UNet necesita CUDA para su forward) y mide la latencia en CPU con ONNX Runtime.

Uso: python crear_colab_exportar_e3.py --e2 <carpeta tesis_imo_para_codex>
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SALIDA_DRIVE = '/content/drive/MyDrive/tesis_imo_e3_artefactos_a0498_v2'
CORRIDAS_DRIVE = '/content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2'


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
    script = (AQUI / 'exportar_onnx.py').read_text(encoding='utf-8')
    firma = hashlib.sha256(script.encode('utf-8')).hexdigest()
    celdas = [celda('markdown', f"""# E3, bloque B: exportación a ONNX de las 10 redes del E2

1. Entorno de ejecución → Cambiar tipo → **GPU** (sirve una T4; no hace falta la L4).
2. Ejecutar las celdas en orden. Lee los checkpoints de `{CORRIDAS_DRIVE}` (solo lectura) y escribe en
   `{SALIDA_DRIVE}` una carpeta por red con el `.onnx` y su `metadata.json`, más `resumen_exportacion.json`.
3. Coloca también `compactas_a0498_e3.zip` en la raíz de MyDrive.
4. Al terminar, descargar la carpeta de salida completa desde Drive.

FCN/U-Net/SegNet usan a0498, cinco semillas verificadas y selección de semilla mediana.
Las otras redes son artefactos históricos E2, no las futuras diez semillas E3.

No se usa test. Una red que no exporta no detiene a las demás: queda con su error en `metadata.json`.
""")]
    for c in base['cells'][1:5]:  # Drive, paquete, título del entorno, entorno
        c = copy.deepcopy(c)
        texto = ''.join(c['source'])
        if c['cell_type'] == 'code' and 'OUTPUT = Path(' in texto:
            texto = texto.replace(f"OUTPUT = Path('{CORRIDAS_DRIVE}')", f"CORRIDAS = Path('{CORRIDAS_DRIVE}')")
            texto = texto.replace('OUTPUT.mkdir(parents=True, exist_ok=True)',
                                  f"assert CORRIDAS.is_dir(), 'No está la carpeta de corridas del E2 en Drive'\n"
                                  f"OUTPUT = Path('{SALIDA_DRIVE}')\nOUTPUT.mkdir(parents=True, exist_ok=True)")
            c['source'] = texto.splitlines(True)
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
        celdas.append(c)
    celdas.append(celda('code', "pip_install('onnx==1.17.0', 'onnxruntime==1.20.1', 'fvcore==0.1.5.post20221221')\n"
                                "subprocess.run([PY, '-c', 'import onnx, onnxruntime; print(onnx.__version__, onnxruntime.__version__)'], check=True)\n"))
    rutas = ['training/exportar_onnx.py', 'training/compactas_e2.py',
             'training/seleccion_compactas.py', 'training/unpool_exportable.py',
             'training/respaldo_torchscript.py',
             'config/hashes_compactas.json', 'config/fuentes_compactas.json']
    contenidos = {r: (AQUI.parent / r).read_text(encoding='utf-8') for r in rutas}
    firmas = {r: hashlib.sha256(t.encode('utf-8')).hexdigest() for r, t in contenidos.items()}
    celdas.append(celda('code', 'CONTENIDOS = ' + repr(contenidos) + '\nFIRMAS = ' + repr(firmas) + '''
E3_EXPORT = Path('/content/e3_export_a0498')
for rel, texto in CONTENIDOS.items():
    destino = E3_EXPORT / rel
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding='utf-8')
    assert hashlib.sha256(destino.read_bytes()).hexdigest() == FIRMAS[rel]
SCRIPT = E3_EXPORT / 'training/exportar_onnx.py'
ZIP_COMPACTAS = Path('/content/drive/MyDrive/compactas_a0498_e3.zip')
assert ZIP_COMPACTAS.is_file(), 'Sube compactas_a0498_e3.zip a la raiz de MyDrive'
COMPACTAS = Path('/content/compactas_a0498')
esperados = json.loads(CONTENIDOS['config/hashes_compactas.json'])
permitidos = {'MANIFIESTO.json'} | {f'fase2/{r}/{n}' for r in esperados for n in ('best.pt','config.json','result.json')}
with zipfile.ZipFile(ZIP_COMPACTAS) as z:
    assert len(z.namelist()) == len(set(z.namelist())) and set(z.namelist()) == permitidos
    for corrida, firma in esperados.items():
        assert hashlib.sha256(z.read(f'fase2/{corrida}/best.pt')).hexdigest() == firma
    z.extractall(COMPACTAS)  # Todos los nombres se validaron contra una lista cerrada.
print('15 pesos a0498 verificados')
'''))
    celdas.append(celda('markdown', '## Exportar\nTarda de 20 a 60 minutos. Si se cae la sesión, repetir todas las celdas: '
                                    'las redes ya exportadas se vuelven a exportar (no hay reanudación parcial).\n'))
    celdas.append(celda('code', "subprocess.run([PY, str(SCRIPT), '--e2', str(PROJECT), '--corridas', str(CORRIDAS),\n"
                                "                '--salida', str(OUTPUT), '--compactas', str(COMPACTAS), '--dispositivo', 'cuda'], check=True)\n"
                                "print((OUTPUT / 'resumen_exportacion.json').read_text())\n"))
    nb = copy.deepcopy(base)
    nb['cells'] = celdas
    destino = AQUI / 'colab/Exportar_ONNX_E3_a0498_v2.ipynb'
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(destino, firma)


if __name__ == '__main__':
    main()
