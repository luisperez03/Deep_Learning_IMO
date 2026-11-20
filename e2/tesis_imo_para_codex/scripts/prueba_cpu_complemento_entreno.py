"""Prueba funcional en CPU del complemento de entrenamiento. NO mide calidad.

Corre, sobre 6 imágenes de train y 4 de validación, una época de cada variante de ablación,
dos réplicas de la permutación de FCN y una de TransUNet, dos intentos de HPO en tres redes
baratas, el sobreajuste con 3 pasos, el piloto (que deja la época 1 a medias y la reanuda) y
el resumidor. VM-UNet queda fuera: exige CUDA y mamba-ssm.

Las máscaras binarias no viajan en el repositorio: se regeneran fuera de él con la misma
función de preparar_base.py y se comprueba su SHA-256 contra el CSV congelado antes de usarlas.

Uso:
  python scripts/prueba_cpu_complemento_entreno.py --raiz-imagenes <carpeta con VOC2012> --trabajo <carpeta temporal>
"""
import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def shim_ml_collections():
    """Solo si falta el paquete: ConfigDict mínimo para construir TransUNet en CPU.

    No cambia el modelo: vit_seg_configs solo usa asignación y lectura de atributos y claves.
    En Colab se usa ml-collections==1.0.0 (requirements-e2.txt).
    """
    if importlib.util.find_spec('ml_collections') is not None:
        return False

    class ConfigDict(dict):
        def __init__(self, data=None):
            super().__init__()
            for k, v in (data or {}).items():
                self[k] = v

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

        def __setattr__(self, key, value):
            self[key] = value

    sys.modules['ml_collections'] = types.SimpleNamespace(ConfigDict=ConfigDict)
    return True


def regenerar_mascaras(rows, raiz_imagenes, destino):
    from PIL import Image
    from inicio_voc import palette, sha
    from preparar_base import binary_mask
    destino.mkdir(parents=True, exist_ok=True)
    colores = palette(Path(raiz_imagenes)/'VOC2012/labelmap.txt')
    distintas = []
    for row in rows:
        out = destino/Path(row['binary_mask_path']).name
        if not out.exists():
            with Image.open(Path(raiz_imagenes)/row['original_mask_path']) as m:
                Image.fromarray(binary_mask(m, colores)).save(out)
        if sha(out) != row['binary_mask_sha256']:
            distintas.append(row['id'])
    return distintas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raiz-imagenes', required=True)
    p.add_argument('--trabajo', required=True)
    p.add_argument('--informe')
    args = p.parse_args()
    trabajo = Path(args.trabajo).resolve(); trabajo.mkdir(parents=True, exist_ok=True)
    uso_shim = shim_ml_collections()

    from dataset_ready import partition
    import complemento_entreno_e2 as C
    import resumir_complemento_entreno_e2 as R

    n_train, n_val = 6, 4
    mascaras = trabajo/'masks_binary'
    distintas = regenerar_mascaras(partition('train')[:n_train] + partition('val')[:n_val],
                                   args.raiz_imagenes, mascaras)
    if distintas:
        raise RuntimeError(f'Máscaras regeneradas con otro SHA-256: {distintas}')
    salida = trabajo/'salida'
    comun = ['--output', str(salida), '--local', '--raiz-imagenes', args.raiz_imagenes,
             '--raiz-mascaras', str(mascaras), '--limite-train', str(n_train), '--limite-val', str(n_val)]
    pasos, tiempos = [], {}

    def correr(nombre, extra):
        t = time.perf_counter()
        C.main([*extra, *comun])
        tiempos[nombre] = round(time.perf_counter()-t, 1)
        pasos.append(nombre)
        print(f'== {nombre}: {tiempos[nombre]} s', flush=True)

    # 1. Caché idéntica a load_example y permutación real.
    import torch
    from dataset_ready import load_example
    train = C.reubicar(partition('train')[:n_train], args.raiz_imagenes, mascaras)
    for row in train:
        a, b = C.ejemplo(row), load_example(row)
        assert all(np.array_equal(u, w) for u, w in zip(a, b)), 'La caché cambia los datos'
    fuente, info = C.mapa_permutacion(train, {'semilla': 20261121, 'b': 1})
    x, y, v = C.lote(train[:2], torch.device('cpu'), fuente)
    y_esperada = np.stack([C.ejemplo(fuente[r['id']])[1] for r in train[:2]])
    assert np.array_equal(y[:, 0].numpy(), y_esperada)
    assert sorted(r['id'] for r in fuente.values()) == sorted(r['id'] for r in train)
    fuente2, _ = C.mapa_permutacion(train, {'semilla': 20261121, 'b': 1})
    assert fuente2 == fuente, 'La permutación no es reproducible'
    pasos.append('cache_y_permutacion')

    # 2. Variantes: quitan el componente y conservan el resto.
    parametros = {}
    for var in ['completo', 'sin_transformer', 'sin_saltos']:
        import experimento_e2_colab as E
        E.setup_seed(11)
        m = C.construir({'modelo': 'transunet_2d', 'variante': var})
        C.comprobar_variante({'variante': var}, m)
        parametros[var] = sum(q.numel() for q in m.parameters())
        if var == 'completo':
            from modelos_e2 import MODELS_E2
            E.setup_seed(11)
            ref = MODELS_E2['transunet_2d']()
            assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), ref.state_dict().values()))
            del ref
        del m
    pasos.append('variantes_construidas')

    # 3. Bloques con una época (el piloto deja época 1 y la ablación la reanuda hasta 2).
    correr('piloto', ['piloto', '--epocas-max', '2'])
    correr('estimar', ['estimar'])
    correr('ablacion', ['ablacion', '--semillas', '11', '--epocas-max', '2'])
    correr('permutacion_fcn', ['permutacion', '--permutacion', 'pipeline_fcn', '--B', '2', '--epocas-max', '2'])
    correr('permutacion_transunet', ['permutacion', '--permutacion', 'transunet', '--B', '1', '--epocas-max', '1'])
    correr('hpo', ['hpo', '--T', '2', '--modelos', 'fcn', 'segnet', 'segformer_b0', '--epocas-max', '1'])
    correr('sobreajuste', ['sobreajuste', '--modelos', 'fcn', 'unet', 'transunet_2d', '--pasos', '3'])
    # Reanudar sobre corridas cerradas no reentrena (lee result.json y comprueba el checkpoint).
    correr('reanudar_cerradas', ['ablacion', '--semillas', '11', '--epocas-max', '2'])
    R.main(['--output', str(salida)])
    pasos.append('resumen')

    informe = {
        'alcance': 'Prueba funcional en CPU con 6 imágenes de train y 4 de validación. No mide calidad '
                   'ni tiempo de GPU; no ejecuta VM-UNet.',
        'shim_ml_collections': uso_shim,
        'torch': torch.__version__,
        'pasos_ok': pasos,
        'segundos_cpu': tiempos,
        'parametros_transunet': parametros,
        'ablacion_epocas': {p.parent.name: len(json.loads((p).read_text(encoding='utf-8')))
                            for p in sorted(salida.glob('abl_*/history.json'))},
        'resultados_cerrados': sorted(p.parent.name for p in salida.glob('*/result.json')),
        'sin_last_pt_tras_cerrar': not any(salida.glob('*/last.pt')),
        'sobreajuste': sorted(p.stem for p in (salida/'sobreajuste').glob('*.json')),
    }
    texto = json.dumps(informe, indent=2, ensure_ascii=False)
    print(texto)
    if args.informe:
        Path(args.informe).parent.mkdir(parents=True, exist_ok=True)
        Path(args.informe).write_text(texto + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
