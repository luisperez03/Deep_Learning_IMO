"""Genera los cuadernos de Colab de las semillas 6 a 10 de los rivales del E3 (opción 2, plan.md).

Parte del cuaderno con el que se corrió el benchmark del E2 (`colab/Benchmark_E2_Colab_v2.ipynb`) y cambia solo
tres cosas:
- la carpeta de salida en Drive (nueva, para no mezclar corridas con las del E2; una subcarpeta por sesión, porque
  cada una escribe su propio preflight.json);
- las semillas: 89, 97, 101, 103 y 107, las cinco siguientes de la lista fijada en el E2
  (`entregable_2/codigo/fases.py`, SEMILLAS), elegidas antes de ver resultado alguno;
- las redes de cada cuaderno, repartidas en sesiones de L4 que pueden correr en paralelo.

Sesión C (protocolo v2): SegFormer-B0 con la configuración ganadora de la búsqueda equitativa del cierre del E2
(`e3/config/experimento_e3_rivales.json`, generada por `configs_e3.py`), con las 10 semillas, porque la receta lo dejaba
mal ajustado. La celda de configuración reemplaza `config/experimento_e2.json` dentro del paquete descomprimido en
Colab (no en el repo), porque el preflight del E2 lee esa ruta fija; verifica que solo cambie SegFormer-B0.

El paquete de datos y código es el mismo ZIP del E2 (mismo SHA-256), así que el código y los datos son idénticos y
la receta de cada red es la del E2 (config/experimento_e2.json): esa es la "misma política" que exige el enunciado.

Las tres compactas (FCN, SegNet, U-Net) no van aquí: esperan el resultado de la búsqueda de arquitectura del E2
(checklist 38, B14). Si la búsqueda elige otra arquitectura, sus 10 semillas se corren con la elegida.

Uso:  python crear_colab_semillas_e3.py --e2 <carpeta tesis_imo_para_codex> --salida <carpeta>
"""
import argparse
import ast
import copy
import json
from pathlib import Path

SEMILLAS_E3 = [89, 97, 101, 103, 107]
DRIVE_SALIDA = '/content/drive/MyDrive/tesis_imo_e3_semillas_v1'

# Horas medidas por 5 semillas en L4 (libro/resultados_e2.md). Reparto para que las dos sesiones duren parecido.
SESIONES = {
    'A': ['transunet_2d', 'pidnet_s'],                                  # 3,46 + 2,29 = 5,75 h
    'B': ['vmunet', 'segnext_t', 'deeplabv3plus', 'unetplusplus'],      # 2,91 + 2,04 + 1,03 + 1,00 = 6,98 h
    'C': ['segformer_b0'],                                              # 1,60 × 2 = 3,2 h (10 semillas)
}
SEMILLAS_C = [11, 23, 37, 53, 71, 89, 97, 101, 103, 107]
CONFIG_C = Path(__file__).resolve().parents[1] / 'config' / 'experimento_e3_rivales.json'


def fuente(celda):
    return ''.join(celda['source'])


def poner(celda, texto):
    if celda['cell_type'] == 'code':
        ast.parse(texto)  # el cuaderno no se genera con una celda que no compila
    celda['source'] = texto.splitlines(True)


def generar(base, sesion, modelos):
    semillas = SEMILLAS_C if sesion == 'C' else SEMILLAS_E3
    nb = copy.deepcopy(base)
    celdas = nb['cells']
    cambios = 0
    for c in celdas:
        s = fuente(c)
        if c['cell_type'] == 'markdown' and s.startswith('# Benchmark E2'):
            poner(c, f"""# E3: semillas 6 a 10 de los rivales, sesión {sesion}

Mismo cuaderno, mismo paquete y misma receta que el benchmark del E2. Cambian la carpeta de Drive
(`tesis_imo_e3_semillas_v1`) y las semillas ({', '.join(map(str, semillas))}).
{'**Sesión C:** SegFormer-B0 con la configuración ganadora de la búsqueda del cierre del E2 (lr 5,57e-4, wd 1e-4).' if sesion == 'C' else ''}

Redes de esta sesión: {', '.join(modelos)}. Las otras sesiones corren las demás; pueden ir en paralelo y escriben
en subcarpetas distintas (`sesion_A`, `sesion_B`, `sesion_C`), porque cada una guarda su propio `preflight.json`.

1. Entorno de ejecución → Cambiar tipo → **GPU L4**.
2. Ejecutar las celdas en orden. Si se cae la sesión, volver a ejecutarlas: cada corrida se reanuda desde su
   última época completa.
3. No se evalúa test ni se descargan pesos.
""")
            cambios += 1
        elif "OUTPUT = Path('/content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2')" in s:
            poner(c, s.replace('/content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2', f'{DRIVE_SALIDA}/sesion_{sesion}'))
            cambios += 1
        elif s.startswith('MODELS=['):
            poner(c, s.replace(s.split('\n')[0], 'MODELS=' + repr(modelos)))
            cambios += 1
        elif s.startswith('MODELS_RUN=MODELS'):
            poner(c, s.replace('SEEDS=[11,23,37,53,71]', 'SEEDS=' + repr(semillas)))
            cambios += 1
        elif c['cell_type'] == 'markdown' and s.startswith('## Ejecutar o reanudar'):
            poner(c, '## Ejecutar o reanudar\nCorre las redes de esta sesión con las cinco semillas nuevas. '
                     'Se recupera la última época completa al reconectar.\n')
            cambios += 1
    if cambios != 5:
        raise RuntimeError(f'El cuaderno base cambió: se esperaban 5 reemplazos y hubo {cambios}')
    for c in celdas:
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
    if sesion == 'C':
        config = CONFIG_C.read_text(encoding='utf-8')
        texto = ("CONFIG_E3 = " + repr(config) + """
ruta = PROJECT / 'config/experimento_e2.json'
original, nueva = json.loads(ruta.read_text()), json.loads(CONFIG_E3)
cambios = [m for m in original['models'] if original['models'][m] != nueva['models'][m]]
assert cambios == ['segformer_b0'], cambios
ruta.write_text(CONFIG_E3)
print('Configuración del E3 escrita en el paquete descomprimido; cambia solo:', cambios, nueva['models']['segformer_b0'])
""")
        ast.parse(texto)
        i = next(k for k, c in enumerate(celdas) if fuente(c).startswith('MODELS=['))
        celdas.insert(i, {'cell_type': 'code', 'metadata': {}, 'source': texto.splitlines(True), 'execution_count': None,
                          'outputs': []})
    return nb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--e2', required=True, help='carpeta tesis_imo_para_codex del repo del benchmark')
    p.add_argument('--salida', required=True)
    a = p.parse_args()
    base = json.loads((Path(a.e2) / 'colab/Benchmark_E2_Colab_v2.ipynb').read_text(encoding='utf-8'))
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    for sesion, modelos in SESIONES.items():
        destino = salida / f'Semillas_E3_sesion_{sesion}.ipynb'
        destino.write_text(json.dumps(generar(base, sesion, modelos), ensure_ascii=False, indent=1), encoding='utf-8')
        print(destino)


if __name__ == '__main__':
    main()
