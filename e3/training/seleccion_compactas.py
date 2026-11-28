"""Selección por validación y carga estricta de las compactas a0498 del E2."""
import hashlib
import json
import statistics
from pathlib import Path

FAMILIAS = ('fcn', 'unet', 'segnet')
SEMILLAS = (11, 23, 37, 53, 71)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seleccionar(raiz, familia, esperado):
    if familia not in FAMILIAS:
        raise ValueError(f'Familia desconocida: {familia}')
    raiz = Path(raiz)
    registros = []
    for seed in SEMILLAS:
        nombre = f'{familia}_a0498_s{seed}'
        carpeta = raiz / 'fase2' / nombre
        resultado = json.loads((carpeta / 'result.json').read_text(encoding='utf-8'))
        config = json.loads((carpeta / 'config.json').read_text(encoding='utf-8'))
        tarea = config['tarea']
        if (not resultado['completed'] or resultado['id'] != nombre
                or tarea['familia'] != familia or tarea['arq'] != 'a0498'
                or tarea['semilla'] != seed or tarea['px'] != 256
                or tarea['w'] != [64, 64, 64] or tarea['b'] != [0, 1, 0]
                or tarea['dropout_p'] != 0.2
                or config['fingerprint'] != resultado['fingerprint']):
            raise ValueError(f'Configuración inesperada: {nombre}')
        peso = carpeta / 'best.pt'
        if sha(peso) != esperado[nombre] or resultado['best_sha256'] != esperado[nombre]:
            raise ValueError(f'Peso alterado: {nombre}')
        registros.append({'seed': seed, 'val': resultado['best_val_group_Dice'],
                          'checkpoint': peso, 'config': config, 'result': resultado})
    orden = sorted(registros, key=lambda r: (r['val'], r['seed']))
    valor_mediano = orden[len(orden) // 2]['val']
    elegida = min((r for r in registros if r['val'] == valor_mediano), key=lambda r: r['seed'])
    return {'semilla': elegida['seed'], 'checkpoint': elegida['checkpoint'],
            'val_dice': elegida['val'], 'semillas_disponibles': list(SEMILLAS),
            'val_por_semilla': {r['seed']: r['val'] for r in registros},
            'configuracion': elegida['config'], 'fingerprint': elegida['result']['fingerprint'],
            'media_val_configuracion': statistics.mean(r['val'] for r in registros),
            'de_val_configuracion': statistics.stdev(r['val'] for r in registros),
            'arquitectura': 'a0498', 'criterio': 'arquitectura ganadora E2; semilla mediana por Dice val'}


def cargar(eleccion, familia):
    import torch
    from compactas_e2 import construir_red
    tarea = eleccion['configuracion']['tarea']
    modelo = construir_red(familia, tarea['w'], tarea['b'], tarea['dropout_p'])
    estado = torch.load(eleccion['checkpoint'], map_location='cpu', weights_only=True)
    if estado['fingerprint'] != eleccion['fingerprint']:
        raise ValueError('Fingerprint del checkpoint diferente')
    modelo.load_state_dict(estado['model'], strict=True)
    return modelo.eval()
