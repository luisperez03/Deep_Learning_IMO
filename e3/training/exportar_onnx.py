"""Bloque B del E3: elige el checkpoint de cada red, lo exporta a ONNX y hace la prueba de ida y vuelta.

Corre en Colab, en el entorno del benchmark del E2 (el `PY` del cuaderno, con torch 2.5.1 y mamba_ssm), con el
paquete del E2 descomprimido (`--e2` = carpeta tesis_imo_para_codex) y los checkpoints en Drive.

Reglas, fijadas antes de mirar resultados del E3:
- **Configuración:** FCN, U-Net y SegNet usan a0498, ganadora de la búsqueda E2,
  con hashes de los 15 checkpoints verificados. Las demás redes siguen siendo
  artefactos históricos E2; no representan las futuras diez semillas E3 ni el SegFormer corregido.
- **Reentrenamiento final (p. 6):** la política, idéntica para todas las redes y para el original, es que el modelo
  final es el checkpoint de la época elegida en validación, entrenado sobre el conjunto de entrenamiento completo
  (las 1.015 imágenes de train). No hay un segundo entrenamiento: la validación solo eligió la época.
- **Artefacto por red:** la semilla cuyo Dice medio por grupo en validación es la mediana de las semillas
  disponibles (con 5 semillas, la tercera; empate: la semilla menor). Representa a la red sin elegir su mejor corrida.
- **Prueba de ida y vuelta:** en las 218 imágenes de validación, error absoluto máximo de la probabilidad < 1e-4,
  coincidencia de la máscara (umbral 0,5) y diferencia del Dice medio por grupo entre el ONNX y el modelo en memoria.
- **Tamaño:** si el ONNX pasa de 95 MB (límite de GitHub, 100 MB), se cuantiza a INT8 dinámico y se mide la pérdida de
  Dice en validación; se acepta si es menor que δ = 0,02. Si no se acepta o la exportación falla, la red queda con
  resultados precalculados en el dashboard y el motivo en `metadata.json`.
- **Costo:** latencia por imagen en CPU con ONNX Runtime (mediana y p95 de 50 pasadas tras 5 de calentamiento, un hilo
  y todos los hilos), parámetros y FLOPs (fvcore, si está), memoria residente pico durante la inferencia.

Uso:
  python exportar_onnx.py --e2 /content/imo_e2_v2/tesis_imo_para_codex \
      --corridas /content/drive/MyDrive/tesis_imo_e2_arquitecturas_reales_v2 --salida /content/drive/MyDrive/tesis_imo_e3_artefactos_v1 \
      [--modelos fcn unet ...] [--dispositivo cuda]
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

TOL_PROB = 1e-4
LIMITE_MB = 95
DELTA = 0.02
OPSET = 17


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dice_por_grupo(probs, filas, metrics):
    grupos = defaultdict(list)
    for p, (fila, y, v) in zip(probs, filas):
        grupos[fila['group_id']].append(metrics(p, y, v)['Dice'])
    return float(np.mean([np.mean(x) for x in grupos.values()]))


def elegir_semilla(corridas, modelo):
    """Semilla de Dice de validación mediano entre las corridas completas del modelo."""
    res = []
    for r in sorted(Path(corridas).glob(f'{modelo}_seed*/result.json')):
        d = json.loads(r.read_text())
        if d.get('completed'):
            res.append((d['best_val_group_Dice'], d['seed'], r.parent / d['best_file']))
    if not res:
        raise FileNotFoundError(f'Sin corridas completas de {modelo} en {corridas}')
    res.sort(key=lambda t: (t[0], t[1]))
    mediana = res[(len(res) - 1) // 2]
    return {'semillas_disponibles': [s for _, s, _ in sorted(res, key=lambda t: t[1])],
            'val_por_semilla': {s: v for v, s, _ in res}, 'semilla': mediana[1], 'val_dice': mediana[0],
            'checkpoint': mediana[2]}


def latencia(sesion, x, hilos):
    import onnxruntime as ort
    nombre = sesion.get_inputs()[0].name
    for _ in range(5):
        sesion.run(None, {nombre: x})
    t = []
    for _ in range(50):
        t0 = time.perf_counter()
        sesion.run(None, {nombre: x})
        t.append((time.perf_counter() - t0) * 1000)
    t.sort()
    return {'hilos': hilos, 'mediana_ms': statistics.median(t), 'p95_ms': t[int(0.95 * len(t)) - 1]}


def sesion_cpu(ruta, hilos=None):
    import onnxruntime as ort
    op = ort.SessionOptions()
    if hilos:
        op.intra_op_num_threads = hilos
    return ort.InferenceSession(str(ruta), op, providers=['CPUExecutionProvider'])


def exportar_uno(nombre, args, MODELS, main_logits, filas_val, metrics, meta):
    import psutil
    import torch
    from torch import nn

    destino = Path(args.salida) / nombre
    destino.mkdir(parents=True, exist_ok=True)
    meta.update({'modelo': nombre, 'fecha': time.strftime('%Y-%m-%d %H:%M:%S'), 'estado': 'iniciado'})
    compacta = nombre in ('fcn', 'unet', 'segnet')
    if compacta:
        from seleccion_compactas import seleccionar, cargar
        if not args.compactas:
            raise ValueError('Las compactas requieren --compactas con los pesos a0498; no se exporta la heredada')
        esperados = json.loads(Path(args.hashes_compactas).read_text(encoding='utf-8'))
        eleccion = seleccionar(args.compactas, nombre, esperados)
    else:
        eleccion = elegir_semilla(args.corridas, nombre)
    meta.update({k: (str(v) if k == 'checkpoint' else v) for k, v in eleccion.items()})
    meta['checkpoint_sha256'] = sha(eleccion['checkpoint'])

    dispositivo = torch.device(args.dispositivo)
    if compacta:
        modelo = cargar(eleccion, nombre)
    else:
        modelo = MODELS[nombre]()
        estado = torch.load(eleccion['checkpoint'], map_location='cpu', weights_only=True)
        modelo.load_state_dict(estado['model'])
    modelo.eval().to(dispositivo)

    class Probabilidad(nn.Module):
        """Salida única del artefacto: probabilidad por píxel, sigmoid del logit principal."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.sigmoid(main_logits(self.m(x)))

    envuelto = Probabilidad(modelo).eval()
    meta['parametros'] = int(sum(p.numel() for p in modelo.parameters()))
    x0 = torch.from_numpy(filas_val[0][0][None]).to(dispositivo)

    try:
        from fvcore.nn import FlopCountAnalysis
        meta['flops'] = int(FlopCountAnalysis(envuelto, x0).unsupported_ops_warnings(False).total())
        meta['flops_fuente'] = 'fvcore (multiplicación-suma contada como 1)'
    except Exception as e:  # noqa: BLE001
        meta['flops'], meta['flops_fuente'] = None, f'no medido: {type(e).__name__}'

    # Salidas del modelo en memoria, imagen por imagen, con el mismo orden que el ONNX.
    with torch.no_grad():
        p_mem = [envuelto(torch.from_numpy(x[None]).to(dispositivo))[0, 0].float().cpu().numpy()
                 for x, _, _ in filas_val_xyv(filas_val)]
    meta['val_dice_memoria'] = dice_por_grupo(p_mem, filas_meta(filas_val), metrics)

    ruta = destino / f'{nombre}.onnx'
    envuelto_cpu = envuelto.to('cpu')
    if nombre == 'segnet':
        from unpool_exportable import adaptar_segnet
        # Misma red y pesos; unpool sin solapamiento expresado como ScatterElements.
        envuelto_cpu.m = adaptar_segnet(modelo)
    torch.onnx.export(envuelto_cpu, x0.cpu(), str(ruta), opset_version=OPSET, input_names=['imagen'],
                      output_names=['probabilidad'], do_constant_folding=True, dynamo=False)
    import onnx
    onnx.checker.check_model(str(ruta))
    meta['onnx_mb'] = ruta.stat().st_size / 2 ** 20

    def evaluar(ruta_onnx):
        s = sesion_cpu(ruta_onnx)
        nombre_in = s.get_inputs()[0].name
        proc = psutil.Process()
        rss0, pico = proc.memory_info().rss, 0
        p = []
        for x, _, _ in filas_val_xyv(filas_val):
            p.append(s.run(None, {nombre_in: x[None]})[0][0, 0])
            pico = max(pico, proc.memory_info().rss)
        return p, (pico - rss0) / 2 ** 20, s

    p_onnx, mem_mb, s = evaluar(ruta)
    dif = max(float(np.abs(a - b).max()) for a, b in zip(p_mem, p_onnx))
    coinc = float(np.mean([((a >= .5) == (b >= .5)).mean() for a, b in zip(p_mem, p_onnx)]))
    meta['ida_y_vuelta'] = {'imagenes': len(p_onnx), 'error_abs_max': dif, 'tolerancia': TOL_PROB,
                            'coincidencia_mascara': coinc, 'val_dice_onnx': dice_por_grupo(p_onnx, filas_meta(filas_val), metrics),
                            'aprueba': dif < TOL_PROB and coinc == 1.0}
    meta['ida_y_vuelta']['diferencia_dice'] = meta['ida_y_vuelta']['val_dice_onnx'] - meta['val_dice_memoria']
    meta['memoria_inferencia_mb'] = mem_mb
    x_np = filas_val[0][0][None]
    meta['latencia_cpu'] = [latencia(sesion_cpu(ruta, 1), x_np, 1), latencia(s, x_np, os.cpu_count())]
    meta['artefacto'] = ruta.name

    if meta['onnx_mb'] > LIMITE_MB:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        ruta_q = destino / f'{nombre}_int8.onnx'
        quantize_dynamic(str(ruta), str(ruta_q), weight_type=QuantType.QInt8)
        p_q, _, sq = evaluar(ruta_q)
        dq = dice_por_grupo(p_q, filas_meta(filas_val), metrics) - meta['val_dice_memoria']
        meta['int8'] = {'mb': ruta_q.stat().st_size / 2 ** 20, 'diferencia_dice_val': dq,
                        'acepta': abs(dq) < DELTA and ruta_q.stat().st_size / 2 ** 20 <= LIMITE_MB,
                        'latencia_cpu': latencia(sq, x_np, os.cpu_count())}
        if meta['int8']['acepta']:
            meta['artefacto'] = ruta_q.name
        else:
            meta['artefacto'] = None
            meta['motivo_sin_artefacto'] = 'No cabe en GitHub ni cuantizado sin perder δ: resultados precalculados'
    meta['artefacto_sha256'] = sha(destino / meta['artefacto']) if meta['artefacto'] else None
    meta['preprocesamiento'] = {'tamano': 256, 'letterbox': True, 'normalizacion': args.norm,
                                'entrada': 'float32 [1, 3, 256, 256], RGB normalizado; padding = 0',
                                'salida': 'float32 [1, 1, 256, 256], probabilidad; máscara = probabilidad >= 0,5'}
    meta['entorno'] = {'python': platform.python_version(), 'torch': torch.__version__,
                       'onnx': __import__('onnx').__version__, 'onnxruntime': __import__('onnxruntime').__version__,
                       'opset': OPSET, 'cpu': platform.processor() or platform.machine(), 'hilos': os.cpu_count()}
    meta['estado'] = 'ok' if meta['ida_y_vuelta']['aprueba'] else 'ida_y_vuelta_fallida'
    if meta['estado'] != 'ok':
        meta['artefacto'] = None
        meta['artefacto_sha256'] = None
        meta['motivo_sin_artefacto'] = 'No superó la prueba de ida y vuelta; no habilitar inferencia'
        if nombre == 'segnet':
            from respaldo_torchscript import exportar_respaldo
            from seleccion_compactas import cargar
            meta['respaldo_torchscript'] = exportar_respaldo(cargar(eleccion, nombre), filas_val, destino, metrics)



def filas_val_xyv(filas):
    return [(x, y, v) for x, y, v, _ in filas]


def filas_meta(filas):
    return [(f, y, v) for _, y, v, f in filas]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e2', required=True)
    ap.add_argument('--corridas', required=True)
    ap.add_argument('--salida', required=True)
    ap.add_argument('--modelos', nargs='+')
    ap.add_argument('--dispositivo', default='cpu')
    ap.add_argument('--compactas', help='Carpeta que contiene fase2/*_a0498_s*/best.pt')
    ap.add_argument('--hashes-compactas', default=str(Path(__file__).resolve().parents[1] / 'config/hashes_compactas.json'))
    ap.add_argument('--data-root', help='Raíz que contiene VOC2012, sin modificar datos')
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    proyecto = Path(args.e2)
    sys.path.insert(0, str(proyecto / 'scripts'))
    from dataset_ready import load_example, metrics, partition
    import torch
    torch.set_num_threads(args.threads)
    if args.modelos and set(args.modelos) <= {'fcn', 'unet', 'segnet'}:
        MODELS_E2 = {}
        def main_logits(salida):
            return salida
    else:
        from modelos_e2 import MODELS_E2, main_logits
    args.norm = json.loads((proyecto / 'resultados/preentrenamiento/normalization_train.json').read_text())
    filas = []
    for fila in sorted(partition('val'), key=lambda r: r['id']):
        x, y, v = load_example(fila, data_root=args.data_root)
        filas.append((x, y, v, fila))
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    resumen = {}
    for nombre in args.modelos or list(MODELS_E2):
        print('==', nombre, flush=True)
        meta = {}
        try:
            exportar_uno(nombre, args, MODELS_E2, main_logits, filas, metrics, meta)
        except Exception as e:  # noqa: BLE001  una red que no exporta no detiene a las demás
            # Se conserva lo ya medido (semilla, parámetros, Dice en memoria) y se agrega el error.
            meta.update({'estado': 'fallo', 'error': f'{type(e).__name__}: {e}', 'traza': traceback.format_exc()[-2000:],
                         'artefacto': None, 'artefacto_sha256': None,
                         'motivo_sin_artefacto': 'Exportación fallida: resultados precalculados'})
        (salida / nombre).mkdir(exist_ok=True)
        (salida / nombre / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        resumen[nombre] = {k: meta.get(k) for k in ['estado', 'semilla', 'onnx_mb', 'artefacto', 'error']}
        resumen[nombre]['ida_y_vuelta'] = (meta.get('ida_y_vuelta') or {}).get('error_abs_max')
        print(json.dumps(resumen[nombre], ensure_ascii=False, default=str), flush=True)
    (salida / 'resumen_exportacion.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
