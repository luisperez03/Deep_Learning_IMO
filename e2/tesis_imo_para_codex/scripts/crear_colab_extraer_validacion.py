"""Genera colab/Extraer_Validacion_E2.ipynb (base del ítem 33), determinista.

Incrusta scripts/extraer_validacion_por_imagen.py con su SHA-256. El cuaderno monta Drive, lee las 50 corridas de
MyDrive/tesis_imo_e2_arquitecturas_reales_v2 (carpeta propia o acceso directo a la compartida), vuelca el Dice por
imagen de validación de cada mejor checkpoint (verificando su SHA-256 y que la media por grupos reproduzca
result.json) y descarga un ZIP con los 50 JSON. Solo lee los checkpoints: no entrena ni evalúa, basta la CPU.
La limitación por área se calcula después en local con scripts/limitacion_area_segformer.py.
  python scripts/crear_colab_extraer_validacion.py
"""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / 'scripts/extraer_validacion_por_imagen.py'
SALIDA = PROJECT / 'colab/Extraer_Validacion_E2.ipynb'


def md(texto):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': texto.strip('\n').splitlines(True)}


def code(texto):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': texto.strip('\n').splitlines(True)}


def celdas():
    script = SCRIPT.read_text(encoding='utf-8')
    firma = hashlib.sha256(script.encode('utf-8')).hexdigest()
    return [
        md("""
# Dice por imagen de validación de las 50 corridas (base del ítem 33)

Lee el mejor checkpoint de cada corrida, verifica su SHA-256 contra `result.json`, comprueba que la media por grupos
reproduzca `best_val_group_Dice` y escribe un JSON por corrida. No necesita GPU. Al final descarga
`validacion_por_imagen_e2.zip`, que va a `resultados/validacion_por_imagen_e2/` del repositorio.
"""),
        code("""
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
RUNS = Path('/content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2')
corridas = sorted(RUNS.glob('*_seed*/result.json'))
print(len(corridas), 'corridas con result.json')
assert len(corridas) == 50, 'Se esperaban 50 corridas'
"""),
        code(f"""
import hashlib
SCRIPT = {script!r}
assert hashlib.sha256(SCRIPT.encode('utf-8')).hexdigest() == {firma!r}
Path('/content/extraer_validacion_por_imagen.py').write_text(SCRIPT, encoding='utf-8')
"""),
        code("""
!python /content/extraer_validacion_por_imagen.py --corridas "$RUNS" --salida /content/validacion_por_imagen_e2
"""),
        code("""
import shutil
from google.colab import files
assert len(list(Path('/content/validacion_por_imagen_e2').glob('*.json'))) == 50
shutil.make_archive('/content/validacion_por_imagen_e2', 'zip', '/content/validacion_por_imagen_e2')
files.download('/content/validacion_por_imagen_e2.zip')
"""),
    ]


def main():
    nb = {'cells': celdas(), 'metadata': {'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                          'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    SALIDA.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(SALIDA)


if __name__ == '__main__':
    main()
