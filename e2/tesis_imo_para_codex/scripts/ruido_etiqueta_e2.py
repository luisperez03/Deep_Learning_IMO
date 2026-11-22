"""Ruido de etiqueta (checklist 12): revisión de 100 imágenes y rótulos cortados por el borde. Exploratorio.

  python scripts/ruido_etiqueta_e2.py --hoja RUTA/hoja_revision_propuesta.csv [--confirmar RUTA/confirmar.csv]

1. Revisión de 100 imágenes de train y validación (muestra al azar, fuera de test). La hoja trae, por imagen,
   veredicto (ok / error / dudoso), tipo de error (falta / sobra / borde / clase) y nota. Es una PRIMERA PASADA
   asistida, no una segunda anotación: los 12 casos marcados error o dudoso están en `confirmar.csv` con dos columnas
   vacías (`confirma`, `veredicto_final`). Si esas columnas están llenas, se usan; si no, se reporta la primera
   pasada con sus tres lecturas (dudoso como ok, dudoso como error, sin dudosos). IC de Wilson 95 %.
2. Rótulos que tocan el borde de la foto. Todos los errores de la primera pasada son rótulos cortados por el borde
   que quedaron sin máscara. Si el conjunto NUNCA anotara rótulos cortados, sería una regla de anotación (aprendible,
   no ruido); si a veces los anota, es inconsistencia. Se mide, en train y validación, la fracción de imágenes con al
   menos una región anotada que toca el borde de la foto (resolución original, máscaras binarias de los
   manifiestos). No se abre el test.

Escribe resultados/ruido_etiqueta_e2/{RESUMEN.json, hoja_revision_propuesta.csv, confirmar.csv}. No copia imágenes.
"""
import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preparar_base import PROJECT, read_csv  # noqa: E402

OUT = PROJECT / 'resultados/ruido_etiqueta_e2'


def wilson(k, n, z=1.959964):
    if n == 0:
        return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - r) / d, 4), round((c + r) / d, 4)]


def tasa(k, n):
    return {'k': k, 'n': n, 'tasa': round(k / n, 4) if n else None, 'ic95_wilson': wilson(k, n)}


def leer_csv(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def borde(split):
    filas = read_csv(PROJECT / 'resultados/preentrenamiento' / f'{split}.csv')
    img_con, reg_borde, reg_total, ids = 0, 0, 0, []
    for r in filas:
        with Image.open(PROJECT / r['binary_mask_path']) as im:
            y = np.asarray(im) > 0
        lab, n = ndi.label(y)
        if n == 0:
            continue
        bordes = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        reg_total += n; reg_borde += len(bordes)
        if bordes:
            img_con += 1; ids.append(r['id'])
    return {'imagenes': len(filas), 'imagenes_con_region_anotada_en_borde': img_con,
            'fraccion_imagenes': round(img_con / len(filas), 4), 'regiones': reg_total,
            'regiones_que_tocan_borde': reg_borde, 'fraccion_regiones': round(reg_borde / reg_total, 4)}, set(ids)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hoja', type=Path, required=True)
    ap.add_argument('--confirmar', type=Path)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    hoja = leer_csv(a.hoja)
    if len(hoja) != 100 or {h['particion'] for h in hoja} - {'train', 'val'}:
        raise SystemExit('La hoja debe tener 100 imágenes de train/val')
    shutil.copyfile(a.hoja, OUT / 'hoja_revision_propuesta.csv')
    conf = leer_csv(a.confirmar) if a.confirmar and a.confirmar.exists() else []
    if a.confirmar and a.confirmar.exists():
        shutil.copyfile(a.confirmar, OUT / 'confirmar.csv')
    final = {c['id']: c['veredicto_final'].strip().lower() for c in conf if c.get('veredicto_final', '').strip()}
    confirmados = len(final)
    ver = {h['id']: final.get(h['id'], h['veredicto']).split(':')[0].strip() for h in hoja}
    n = len(ver); err = sum(v == 'error' for v in ver.values()); dud = sum(v == 'dudoso' for v in ver.values())
    por_split = {}
    for s in ('train', 'val'):
        vs = [ver[h['id']] for h in hoja if h['particion'] == s]
        e, d = vs.count('error'), vs.count('dudoso')
        por_split[s] = {'n': len(vs), 'ok': vs.count('ok'), 'error': e, 'dudoso': d,
                        'dudoso_como_ok': tasa(e, len(vs)), 'dudoso_como_error': tasa(e + d, len(vs))}
    # veredicto_final admite 'error:<tipo>' (p. ej. 'error:falta'); el tipo confirmado a mano manda sobre el de la hoja.
    tipo = {h['id']: (final[h['id']].split(':', 1)[1].strip() if ':' in final.get(h['id'], '') else h['tipo_error'])
            for h in hoja}
    tipos = {}
    for h in hoja:
        if ver[h['id']] == 'error':
            tipos[tipo[h['id']] or 'sin_tipo'] = tipos.get(tipo[h['id']] or 'sin_tipo', 0) + 1
    casos = [{'id': h['id'], 'particion': h['particion'], 'veredicto': ver[h['id']], 'tipo': tipo[h['id']], 'nota': h['nota']}
             for h in hoja if ver[h['id']] != 'ok']

    est_borde, ids_borde = {}, set()
    for s in ('train', 'val'):
        est_borde[s], ids = borde(s); ids_borde |= ids
    ok_con_borde = [h['id'] for h in hoja if ver[h['id']] == 'ok' and h['id'] in ids_borde]

    estado = ('Exploratorio. Primera pasada asistida con los casos de error y dudosos confirmados a mano; no es acuerdo '
              'entre anotadores.' if conf and confirmados == len(conf) else
              'Exploratorio. Primera pasada asistida; no es segunda anotación independiente ni acuerdo entre anotadores.')
    R = {'estado': estado,
         'muestra': {'n': n, 'particiones': {s: sum(h['particion'] == s for h in hoja) for s in ('train', 'val')},
                     'seleccion': 'al azar en train y validación; test excluido', 'revisor': sorted({h.get('revisor', '') for h in hoja})},
         'casos_confirmados_a_mano': confirmados,
         'casos_por_confirmar': sum(1 for c in conf if not c.get('veredicto_final', '').strip()) if conf else err + dud,
         'veredictos': {'ok': n - err - dud, 'error': err, 'dudoso': dud},
         'tasa_imagenes_con_error': {'dudoso_como_ok': tasa(err, n), 'dudoso_como_error': tasa(err + dud, n),
                                     'sin_dudosos': tasa(err, n - dud)},
         'por_tipo': tipos, 'por_particion': por_split, 'casos': casos,
         'rotulos_en_borde_anotados': {**est_borde,
             'imagenes_de_la_muestra_ok_con_region_anotada_en_borde': len(ok_con_borde),
             'ids': ok_con_borde},
         'lectura': [
             'Todos los errores de la primera pasada son del mismo tipo: un rótulo cortado por el borde de la foto que '
             'quedó sin máscara (tipo falta). No hubo contornos mal trazados, sobrantes ni clases cambiadas.',
             'El conjunto sí anota muchos rótulos que tocan el borde (ver fracciones), así que la omisión no es una regla '
             'consistente sino una inconsistencia: el modelo recibe señales contradictorias justo en los rótulos cortados.',
             'La tasa es por imagen, no por píxel: un rótulo cortado suele ser pequeño, así que el efecto sobre el Dice '
             'por imagen está acotado pero no cuantificado. No se corrigieron máscaras de test.']}
    (OUT / 'RESUMEN.json').write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: R[k] for k in ('veredictos', 'tasa_imagenes_con_error', 'casos_por_confirmar')}, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in R['rotulos_en_borde_anotados'].items() if k != 'ids'}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
