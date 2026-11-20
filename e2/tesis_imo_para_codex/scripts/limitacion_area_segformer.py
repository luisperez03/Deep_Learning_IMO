"""Limitación del paper guía (ítem 33): caída de Dice en rótulos pequeños.

Sigue config/limitacion_segformer_area.json, fijado antes de ver resultados por
área en validación. Entrada: un JSON por corrida con el Dice por imagen de
validación (scripts/extraer_validacion_por_imagen.py) y el área de las máscaras
(resultados/complementos_locales_e2/eda_por_imagen.json).

Uso:
  python limitacion_area_segformer.py --por-imagen resultados/validacion_por_imagen_e2 \
      --salida resultados/limitacion_segformer_area
  python limitacion_area_segformer.py --simular especifica --salida <tmp>   # prueba sin datos reales
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOLO = PROJECT / 'config/limitacion_segformer_area.json'
EDA = PROJECT / 'resultados/complementos_locales_e2/eda_por_imagen.json'
TODAS = ['fcn', 'segnet', 'unet', 'deeplabv3plus', 'unetplusplus', 'pidnet_s',
         'segnext_t', 'segformer_b0', 'transunet_2d', 'vmunet']
SEMILLAS = [11, 23, 37, 53, 71]


def leer_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ---------------------------------------------------------------- datos
def grupos_validacion(eda, variable='foreground_fraction'):
    """Área (u otra variable) media por grupo en validación, en orden estable."""
    valores = defaultdict(list)
    imagen_a_grupo = {}
    for fila in eda:
        if fila['split'] == 'val':
            valores[fila['group_id']].append(fila[variable])
            imagen_a_grupo[fila['id']] = fila['group_id']
    orden = sorted(valores)
    return orden, np.array([np.mean(valores[g]) for g in orden]), imagen_a_grupo


def terciles(valores):
    cortes = np.quantile(valores, [1 / 3, 2 / 3])
    return np.digitize(valores, cortes, right=True), cortes


def cargar_por_imagen(carpeta):
    """{(modelo, semilla): [filas]} desde los JSON del extractor."""
    corridas = {}
    for archivo in sorted(Path(carpeta).glob('*.json')):
        datos = leer_json(archivo)
        if datos.get('split') != 'val':
            raise ValueError(f'{archivo.name} no es de validación')
        clave = (datos['model'], int(datos['seed']))
        if clave in corridas:
            raise ValueError(f'Corrida duplicada {clave}')
        corridas[clave] = datos['detail']
    return corridas


def matriz_dice(corridas, modelos, semillas, orden_grupos, imagen_a_grupo):
    """D[m, s, g] = media del Dice de las imágenes del grupo g."""
    indice = {g: i for i, g in enumerate(orden_grupos)}
    D = np.full((len(modelos), len(semillas), len(orden_grupos)), np.nan)
    for i, m in enumerate(modelos):
        for j, s in enumerate(semillas):
            if (m, s) not in corridas:
                raise ValueError(f'Falta la corrida {m} semilla {s}')
            por_grupo = defaultdict(list)
            for fila in corridas[(m, s)]:
                if imagen_a_grupo.get(fila['id']) != fila['group_id']:
                    raise ValueError(f"Imagen {fila['id']} con grupo distinto al de la EDA")
                por_grupo[fila['group_id']].append(fila['Dice'])
            ids = [fila['id'] for fila in corridas[(m, s)]]
            if len(ids) != len(set(ids)) or set(ids) != set(imagen_a_grupo):
                raise ValueError(f'{m} semilla {s}: imágenes de validación incompletas o repetidas')
            for g, v in por_grupo.items():
                D[i, j, indice[g]] = np.mean(v)
    return D


# ---------------------------------------------------------------- estadística
def medias_estrato(D, tercil, pesos=None):
    """Media por tercil, promediada sobre semillas. Con pesos: (M, 3, B)."""
    salida = []
    for t in range(3):
        idx = np.flatnonzero(tercil == t)
        Dt = D[:, :, idx]  # M, S, n_t
        if pesos is None:
            salida.append(Dt.mean(axis=2).mean(axis=1))
        else:
            W = pesos[t]  # B, n_t
            salida.append((np.einsum('msn,bn->msb', Dt, W) / len(idx)).mean(axis=1))
    return np.stack(salida, axis=1)


def pesos_bootstrap(tercil, replicas, semilla):
    rng = np.random.default_rng(semilla)
    pesos = []
    for t in range(3):
        n = int((tercil == t).sum())
        pesos.append(np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(replicas)]).astype(float))
    return pesos


def ic95(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def p_bootstrap(x):
    return float(min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0))))


def holm(pvalores):
    orden = np.argsort(pvalores)
    m = len(pvalores)
    ajustados = np.empty(m)
    acumulado = 0.0
    for rango, i in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (m - rango) * pvalores[i]))
        ajustados[i] = acumulado
    return ajustados.tolist()


def analizar(D, tercil, modelos, protocolo):
    inc = protocolo['incertidumbre']
    crit = protocolo['criterio_a_priori']
    guia = protocolo['redes']['paper_guia']
    piloto = protocolo['redes']['piloto']
    punto = medias_estrato(D, tercil)                      # M, 3
    pesos = pesos_bootstrap(tercil, inc['replicas'], inc['semilla'])
    boot = medias_estrato(D, tercil, pesos)                # M, 3, B
    brecha, brecha_boot = punto[:, 2] - punto[:, 0], boot[:, 2] - boot[:, 0]
    # DE entre semillas de la brecha
    por_semilla = np.stack([D[:, :, tercil == t].mean(axis=2) for t in range(3)], axis=2)  # M, S, 3
    de_semillas = (por_semilla[:, :, 2] - por_semilla[:, :, 0]).std(axis=1, ddof=1)
    redes = {}
    for i, m in enumerate(modelos):
        redes[m] = {
            'Dice_T1': float(punto[i, 0]), 'IC95_T1': ic95(boot[i, 0]),
            'Dice_T2': float(punto[i, 1]), 'IC95_T2': ic95(boot[i, 1]),
            'Dice_T3': float(punto[i, 2]), 'IC95_T3': ic95(boot[i, 2]),
            'Delta_T3_menos_T1': float(brecha[i]), 'IC95_Delta': ic95(brecha_boot[i]),
            'DE_semillas_Delta': float(de_semillas[i]),
            'razon_T1_T3': float(punto[i, 0] / punto[i, 2]),
        }
    g = modelos.index(guia)
    contrastes = {}
    for c in piloto:
        k = modelos.index(c)
        C = brecha_boot[g] - brecha_boot[k]
        N = boot[g, 0] - boot[k, 0]
        contrastes[c] = {'C': float(brecha[g] - brecha[k]), 'IC95_C': ic95(C), 'p_bootstrap': p_bootstrap(C),
                         'nivel_T1_guia_menos_c': float(punto[g, 0] - punto[k, 0]), 'IC95_nivel_T1': ic95(N)}
    ajustados = holm([contrastes[c]['p_bootstrap'] for c in piloto])
    for c, p in zip(piloto, ajustados):
        contrastes[c]['p_holm'] = p
        contrastes[c]['significativo_holm'] = p < 0.05
    delta = crit['delta_minimo']
    existe = redes[guia]['IC95_Delta'][0] > delta
    especifica = existe and all(contrastes[c]['IC95_C'][0] > 0 and contrastes[c]['significativo_holm'] for c in piloto)
    compartida = (existe and sum(redes[c]['IC95_Delta'][0] > delta for c in piloto) >= 2
                  and not any(contrastes[c]['significativo_holm'] for c in piloto))
    if not existe:
        veredicto = 'no_confirmada'
    elif especifica:
        veredicto = 'especifica'
    elif compartida:
        veredicto = 'compartida'
    else:
        veredicto = 'parcial'
    return {'redes': redes, 'contrastes_paper_guia': contrastes,
            'L_existe': bool(existe), 'veredicto': veredicto}


def resumen_markdown(res):
    lineas = [f"# Limitación de SegFormer-B0 por área del rótulo ({res['origen']})", '',
              f"Veredicto a priori: **{res['primario']['veredicto']}**. Protocolo SHA-256 `{res['sha256_protocolo'][:12]}`.",
              f"Cortes de terciles (fracción de área del grupo): {', '.join(f'{c:.4f}' for c in res['cortes_terciles'])}; "
              f"grupos por tercil {res['grupos_por_tercil']}.", '',
              '| Red | Dice T1 | Dice T3 | Δ T3-T1 [IC95] | DE semillas Δ |', '|---|---|---|---|---|']
    for m, r in res['primario']['redes'].items():
        lineas.append(f"| {m} | {r['Dice_T1']:.3f} | {r['Dice_T3']:.3f} | {r['Delta_T3_menos_T1']:.3f} "
                      f"[{r['IC95_Delta'][0]:.3f}; {r['IC95_Delta'][1]:.3f}] | {r['DE_semillas_Delta']:.3f} |")
    lineas += ['', '| Comparador | C = Δ guía - Δ c [IC95] | p Holm | Nivel T1 guía - c [IC95] |', '|---|---|---|---|']
    for c, r in res['primario']['contrastes_paper_guia'].items():
        lineas.append(f"| {c} | {r['C']:.3f} [{r['IC95_C'][0]:.3f}; {r['IC95_C'][1]:.3f}] | {r['p_holm']:.4f} | "
                      f"{r['nivel_T1_guia_menos_c']:.3f} [{r['IC95_nivel_T1'][0]:.3f}; {r['IC95_nivel_T1'][1]:.3f}] |")
    if res['origen'] != 'datos reales de validación':
        lineas += ['', '**Datos simulados: estos números no son resultados del proyecto.**']
    return '\n'.join(lineas) + '\n'


# ---------------------------------------------------------------- simulación
def simular(eda, escenario, semilla=5):
    """Dice por imagen simulado sobre los grupos reales de validación.

    'especifica': SegFormer cae 0,15 de T3 a T1 y los demás 0,04.
    'compartida': todas caen 0,10. 'nula': ninguna cae.
    """
    rng = np.random.default_rng(semilla)
    orden, area, imagen_a_grupo = grupos_validacion(eda)
    tercil, _ = terciles(area)
    t_de = dict(zip(orden, tercil))
    efecto_guia = {'especifica': 0.15, 'compartida': 0.10, 'nula': 0.0}[escenario]
    efecto_otros = {'especifica': 0.04, 'compartida': 0.10, 'nula': 0.0}[escenario]
    base = {m: 0.70 + 0.015 * i for i, m in enumerate(TODAS)}
    efecto_grupo = {g: rng.normal(0, 0.06) for g in orden}
    corridas = {}
    for m in TODAS:
        caida = efecto_guia if m == 'segformer_b0' else efecto_otros
        for s in SEMILLAS:
            filas = []
            for img, g in imagen_a_grupo.items():
                mu = base[m] + efecto_grupo[g] - caida * (2 - t_de[g]) / 2 + rng.normal(0, 0.01)
                filas.append({'id': img, 'group_id': g, 'Dice': float(np.clip(mu + rng.normal(0, 0.04), 0, 1))})
            corridas[(m, s)] = filas
    return corridas


def ejecutar(corridas, eda, protocolo, origen, sha_protocolo):
    modelos_disp = sorted({m for m, _ in corridas})
    requeridas = [protocolo['redes']['paper_guia']] + protocolo['redes']['piloto']
    faltan = [m for m in requeridas if m not in modelos_disp]
    if faltan:
        raise ValueError(f'Faltan redes piloto: {faltan}')
    modelos = [m for m in TODAS if m in modelos_disp]
    orden, area, imagen_a_grupo = grupos_validacion(eda)
    tercil, cortes = terciles(area)
    D = matriz_dice(corridas, modelos, SEMILLAS, orden, imagen_a_grupo)
    primario = analizar(D, tercil, modelos, protocolo)
    _, region, _ = grupos_validacion(eda, 'smallest_region_pixels')
    tercil_reg, cortes_reg = terciles(region)
    secundario = analizar(D, tercil_reg, modelos, protocolo)
    return {'origen': origen, 'sha256_protocolo': sha_protocolo, 'particion': 'val',
            'grupos': len(orden), 'imagenes': len(imagen_a_grupo), 'semillas': SEMILLAS, 'redes': modelos,
            'cortes_terciles': cortes.tolist(), 'grupos_por_tercil': np.bincount(tercil, minlength=3).tolist(),
            'primario': primario,
            'secundario_region_minima': {'cortes_terciles_pixeles': cortes_reg.tolist(),
                                         'nota': 'Descriptivo; no decide el veredicto.',
                                         'redes': {m: {k: v for k, v in r.items() if 'Delta' in k}
                                                   for m, r in secundario['redes'].items()}}}


def main():
    parser = argparse.ArgumentParser()
    fuente = parser.add_mutually_exclusive_group(required=True)
    fuente.add_argument('--por-imagen', help='carpeta con *_val_por_imagen.json')
    fuente.add_argument('--simular', choices=['especifica', 'compartida', 'nula'])
    parser.add_argument('--salida', required=True)
    parser.add_argument('--protocolo', default=str(PROTOCOLO))
    parser.add_argument('--eda', default=str(EDA))
    args = parser.parse_args()
    protocolo = leer_json(args.protocolo)
    sha = hashlib.sha256(Path(args.protocolo).read_bytes()).hexdigest()
    eda = leer_json(args.eda)
    if args.simular:
        corridas, origen = simular(eda, args.simular), f'SIMULADO ({args.simular})'
    else:
        corridas, origen = cargar_por_imagen(args.por_imagen), 'datos reales de validación'
    res = ejecutar(corridas, eda, protocolo, origen, sha)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    (salida / 'RESUMEN.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    (salida / 'RESUMEN.md').write_text(resumen_markdown(res), encoding='utf-8')
    print(resumen_markdown(res))


if __name__ == '__main__':
    main()
