"""Costo de la búsqueda de arquitectura: medición en CPU local y estimación de horas de L4.

1. Mide en CPU el segundo por paso de entrenamiento (lote 2, 256 px, Adam, pérdida del benchmark) y por
   imagen de validación de 15 configuraciones (5 anchos x 3 familias), con datos sintéticos: aísla el
   cómputo de la red, sin lectura de imágenes.
2. Calibra la L4 con los segundos por época del benchmark (config/complemento_entreno_e2.json,
   referencia_original) y los MACs de las cinco redes convolucionales del benchmark:
   t_epoca = a + c * MACs_por_epoca, con MACs_por_epoca = MACs * (3 * 1015 + 218).
3. Suma sobre las 1000 arquitecturas por familia y escribe escenarios de fidelidad.

Todo lo que dice "L4" es ESTIMADO: el piloto del cuaderno lo reemplaza por una medición.
Salida: resultados/busqueda_arquitectura_e2/estimacion_costo_local.json
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import busqueda_arquitectura_e2 as B
import experimento_e2_colab as E
from modelos_e2 import MODELS_E2

PROJECT = B.PROJECT
OUT = PROJECT/'resultados/busqueda_arquitectura_e2/estimacion_costo_local.json'
CONFIGS = [((16, 16, 16), (0, 0, 0)), ((16, 32, 64), (0, 0, 0)), ((64, 64, 64), (0, 1, 0)),
           ((128, 128, 128), (0, 0, 0)), ((256, 256, 256), (1, 1, 1))]
N_TRAIN, N_VAL = 1015, 218


def medir_cpu(familia, w, b, proto, pasos=3, val=3):
    torch.manual_seed(97)
    m = B.construir_red(familia, w, b, 0.2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 256, 256); y = (torch.rand(2, 1, 256, 256) > .7).float()
    v = torch.ones(2, 1, 256, 256, dtype=torch.bool)
    m.train()
    for i in range(pasos + 1):  # el primero calienta
        if i == 1:
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        E.compute_loss(m(x), y, v, proto).backward(); opt.step()
    paso = (time.perf_counter() - t0)/pasos
    m.eval()
    with torch.no_grad():
        m(x[:1]); t0 = time.perf_counter()
        for _ in range(val):
            m(x[:1])
    return paso, (time.perf_counter() - t0)/val, B.parametros(m), B.macs(m)


def macs_benchmark():
    out = {}
    for nombre in ['fcn', 'segnet', 'unet', 'deeplabv3plus', 'unetplusplus']:
        out[nombre] = B.macs(MODELS_E2[nombre](), size=256, size_calculo=256)
    return out


def ajuste(x, y):
    c, a = np.polyfit(x, y, 1)
    pred = a + c*np.asarray(x)
    r2 = 1 - np.sum((y - pred)**2)/np.sum((y - np.mean(y))**2)
    return float(a), float(c), float(r2)


def main():
    torch.set_num_threads(6)
    cfg = json.loads(B.CONFIG.read_text(encoding='utf-8'))
    proto = json.loads((PROJECT/'config/experimento_e2.json').read_text(encoding='utf-8'))
    ref = json.loads((PROJECT/'config/complemento_entreno_e2.json').read_text(encoding='utf-8'))['referencia_original']
    arqs = B.espacio(cfg); mt = B.tabla_macs(cfg, arqs, B.FAMILIAS)

    cpu = []
    for f in B.FAMILIAS:
        for w, b in CONFIGS:
            paso, val, npar, nmac = medir_cpu(f, w, b, proto)
            epoca = paso*np.ceil(N_TRAIN/2) + val*N_VAL
            cpu.append({'familia': f, 'w': w, 'b': b, 'parametros': npar, 'macs': nmac,
                        'segundos_por_paso_lote2': round(paso, 3), 'segundos_por_imagen_val': round(val, 4),
                        'segundos_por_epoca_cpu_sin_lectura': round(epoca, 1)})
            print(cpu[-1], flush=True)
    a_cpu, c_cpu, r2_cpu = ajuste([r['macs']*(3*N_TRAIN + N_VAL) for r in cpu],
                                  np.array([r['segundos_por_epoca_cpu_sin_lectura'] for r in cpu]))

    mb = macs_benchmark(); s_l4 = ref['segundos_por_epoca_l4']
    xs = [mb[k]*(3*N_TRAIN + N_VAL) for k in mb]; ys = np.array([s_l4[k] for k in mb])
    a_l4, c_l4, r2_l4 = ajuste(xs, ys)
    tflops_ef = 2/c_l4/1e12

    def epoca_l4(macs, a=a_l4, c=c_l4):
        return a + c*macs*(3*N_TRAIN + N_VAL)

    f1, f2 = cfg['fase1'], cfg['fase2']
    sum_epoca = {f: float(sum(epoca_l4(mt[(f, x['arq'])]) for x in arqs)) for f in B.FAMILIAS}
    med_epoca = {f: float(np.median([epoca_l4(mt[(f, x['arq'])]) for x in arqs])) for f in B.FAMILIAS}
    top_epoca = {f: sorted(epoca_l4(mt[(f, x['arq'])]) for x in arqs) for f in B.FAMILIAS}
    solo_computo = {f: float(sum(epoca_l4(mt[(f, x['arq'])], a=0.) for x in arqs)) for f in B.FAMILIAS}

    def h(s):
        return round(s/3600, 1)

    n_r = cfg['ruido']['n_arquitecturas']
    fase1 = {f: h(sum_epoca[f]*f1['epocas']) for f in B.FAMILIAS}
    ruido = h(sum(med_epoca[f]*n_r*f1['epocas'] for f in B.FAMILIAS))
    a_ref = B.buscar(arqs, cfg['referencia']['w'], cfg['referencia']['b'])['arq']
    ajuste_h = h(sum(epoca_l4(mt[(f, a_ref)]) for f in B.FAMILIAS)*f1['epocas']*len(cfg['ajuste_previo']['lr']))
    fase2 = {}
    for k in (10, 20):
        n = k + 2
        fase2[f'k{k}'] = {'mediana_30_epocas': h(sum(med_epoca[f]*n for f in B.FAMILIAS)*5*f2['epocas']),
                          'mas_caros_30_epocas': h(sum(sum(top_epoca[f][-n:]) for f in B.FAMILIAS)*5*f2['epocas']),
                          'mas_caros_22_epocas': h(sum(sum(top_epoca[f][-n:]) for f in B.FAMILIAS)*5*22)}
    # Rangos de L4: la calibración con las redes SMP no es fiable para convoluciones 3x3 anchas (R2 bajo),
    # así que se dan tres rendimientos efectivos y dos sobrecostos fijos por época con la caché llena.
    n_r = cfg['ruido']['n_arquitecturas']; n_aj = len(cfg['ajuste_previo']['lr'])
    corridas = 3*(1000 + n_r) + 3*n_aj
    macs_total = sum(mt.values()) + sum(mt[(f, x['arq'])] for f in B.FAMILIAS for x in B.orden_fase1(cfg, arqs)[:n_r])         + n_aj*sum(mt[(f, a_ref)] for f in B.FAMILIAS)
    escenarios = {}
    for o in cfg['fidelidad_fase1']['opciones']:
        px, fr = o['px'], o['fraccion_train']
        filas = {}
        for tf in (8, 15, 25):
            for a in (5, 12):
                comp = macs_total*(px/256)**2*(3*N_TRAIN*fr + N_VAL)*2/(tf*1e12)*f1['epocas']
                filas[f'{tf}_TFLOPS_a{a}s'] = h(comp + corridas*f1['epocas']*a)
        escenarios[o['nombre']] = {'px': px, 'fraccion_train': fr, 'horas_ajuste_fase1_ruido': filas}
    fase2_rangos = {}
    for k in (10, 20):
        n = k + 2
        for tf in (8, 15, 25):
            med = sum(float(np.median([mt[(f, x['arq'])] for x in arqs])) for f in B.FAMILIAS)*n
            top = sum(sum(sorted(mt[(f, x['arq'])] for x in arqs)[-n:]) for f in B.FAMILIAS)
            for nombre, M in (('mediana', med), ('mas_caros', top)):
                fase2_rangos[f'k{k}_{tf}_TFLOPS_{nombre}'] = h((M*(3*N_TRAIN + N_VAL)*2/(tf*1e12) + 3*n*5)*5*24)
    out = {'estimado': True, 'fecha': time.strftime('%Y-%m-%d'),
           'equipo_cpu': {'procesador': platform.processor(), 'hilos_torch': torch.get_num_threads(),
                          'torch': torch.__version__},
           'cpu_medido': cpu,
           'cpu_ajuste_epoca_vs_macs': {'a_segundos': a_cpu, 'c_segundos_por_mac': c_cpu, 'r2': r2_cpu,
                                        'gmacs_por_segundo': round(1/c_cpu/1e9, 1)},
           'l4_calibracion': {'fuente': 'segundos_por_epoca_l4 del benchmark (lectura desde disco, sin caché)',
                              'macs_benchmark': mb, 'segundos_epoca_l4': {k: s_l4[k] for k in mb},
                              'a_segundos': a_l4, 'c_segundos_por_mac': c_l4, 'r2': r2_l4,
                              'tflops_efectivos': round(tflops_ef, 2),
                              'nota': 'ResNet-18 de las redes SMP es menos eficiente por MAC que convoluciones 3x3 anchas a resolución completa: la estimación del cómputo tiende a ser alta. El término fijo a incluye lectura desde disco, que la caché en RAM reduce.'},
           'l4_segundos_epoca_mediana_por_familia': {f: round(v, 1) for f, v in med_epoca.items()},
           'horas_l4_fase1_por_familia_6_epocas': fase1,
           'horas_l4_repeticion_ruido': ruido, 'horas_l4_ajuste_previo': ajuste_h,
           'horas_l4_fase2_calibracion_smp': fase2, 'escenarios_fidelidad_fase1': escenarios,
           'fase2_rangos_24_epocas_a5s': fase2_rangos,
           'supuestos_rangos': 'Rendimiento efectivo de 8, 15 o 25 TFLOPS (TF32 de cuDNN en L4) y sobrecosto fijo de 5 o 12 s por época con la caché llena, un proceso. Fase 2: 5 semillas, 24 épocas medias, k + 2 arquitecturas por familia.'}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('cpu_ajuste_epoca_vs_macs', 'l4_calibracion',
                                           'l4_segundos_epoca_mediana_por_familia',
                                           'horas_l4_fase1_por_familia_6_epocas', 'horas_l4_repeticion_ruido',
                                           'horas_l4_ajuste_previo', 'escenarios_fidelidad_fase1',
                                           'fase2_rangos_24_epocas_a5s')},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
