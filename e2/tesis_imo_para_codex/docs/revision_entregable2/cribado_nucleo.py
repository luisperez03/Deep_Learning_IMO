"""Cribado de elegibilidad del núcleo de dominio y validación por muestra del precribado automático.

Alcance acordado: (1) texto completo de todo registro de dominio (RETENER_DOMINIO y cualquier registro cuyo título o
resumen mencione rótulos o mercancía peligrosa); (2) muestra aleatoria estratificada del precribado automático,
revisada a título y resumen, para estimar cuánto se equivoca el filtro. El resto del corpus queda con la decisión
automática y su tasa de error estimada; no se presenta como cribado humano completo.

Pasos:
  python docs/revision_entregable2/cribado_nucleo.py preparar
      Requiere registros_conciliados.json (procesar_exportaciones.py con los alias). Usa las decisiones por registro
      de precribado_registros.json si existe ({"E2-0001": "RETENER_DOMINIO", ...}); si no existe, lo genera con las
      REGLAS de abajo y lo declara (sus conteos pueden diferir de resumen_cribado.json, que no trae registros).
      Escribe cribado_manual.csv: una fila por registro del núcleo y de la muestra, con columnas vacías para la
      decisión. Sin resúmenes (repositorio público): se leen en registros_conciliados.json por ID.
  python docs/revision_entregable2/cribado_nucleo.py cerrar
      Lee cribado_manual.csv completo y escribe PRISMA_FINAL.md y prisma_final.json: flujo con exclusiones por
      motivo, error del filtro por estrato con IC95 de Wilson y denominador de estudios de dominio para el gap.
Revisión asistida por IA validada por el investigador (protocolo, paso 6); sin kappa: no hay dos revisores.
"""
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEMILLA = 20261122
MUESTRA = {'EXCLUIR': 60, 'DUDA': 60, 'RETENER_MECANISMO': 40, 'RETENER_ARQUITECTURA': 40}  # 200 registros
DOMINIO = re.compile(r'hazmat|hazardous material|dangerous good|placard|\bimdg\b|\bghs\b|hazard (?:label|sign|symbol)|'
                     r'warning (?:label|sign)|\bun number\b', re.I)  # igual que en precribado_reconstruido.py
# Reglas de respaldo, solo si no hay decisiones por registro del precribado original.
REGLAS = [
    ('RETENER_DOMINIO', DOMINIO),
    ('EXCLUIR_TIPO', re.compile(r'^(?:erratum|correction|editorial|retraction|proceedings)', re.I)),
    ('RETENER_ARQUITECTURA', re.compile(r'segformer|deeplab|segnet|transunet|unet\+\+|u-net\+\+|pidnet|segnext|'
                                        r'vm-unet|swin-unet|attention u-net|\bfcn\b', re.I)),
    ('RETENER_MECANISMO', re.compile(r'(?:small object|boundary|edge-aware|multi-scale|attention|state space|mamba)'
                                     r'.{0,80}segment', re.I)),
    ('EXCLUIR_TAREA', re.compile(r'object detection|bounding box|image classification', re.I)),
]
MOTIVOS = ['tarea sin transferencia', 'sin método original', 'información insuficiente', 'duplicado o versión',
           'fuera de ventana', 'idioma no evaluable', 'texto completo no recuperado']
CAMPOS = ['id', 'estrato', 'decision_automatica', 'anio', 'titulo', 'doi', 'nivel', 'decision_manual',
          'clasificacion', 'motivo', 'segmenta_rotulo_por_pixel', 'compara_arquitecturas_con_semillas', 'nota']


def estrato(dec):
    return 'EXCLUIR' if dec.startswith('EXCLUIR') else dec


def texto(g):
    r = g['registros'][0]
    return f"{g['titulo']} {r.get('resumen', '')}"


def preparar():
    grupos = json.loads((ROOT / 'registros_conciliados.json').read_text(encoding='utf-8'))
    ruta = ROOT / 'precribado_registros.json'
    if ruta.exists():
        dec, origen = json.loads(ruta.read_text(encoding='utf-8')), 'precribado_registros.json'
    else:
        dec = {g['id']: next((c for c, rx in REGLAS if rx.search(texto(g))), 'DUDA') for g in grupos}
        ruta.write_text(json.dumps(dec, ensure_ascii=False, indent=0), encoding='utf-8')
        origen = 'REGLAS de respaldo de cribado_nucleo.py'
    faltan = [g['id'] for g in grupos if g['id'] not in dec]
    if faltan:
        sys.exit(f'{len(faltan)} grupos sin decisión automática (p. ej. {faltan[:3]})')
    # Núcleo: decisión de dominio, mención de rótulos o mercancía peligrosa, o procedencia de las consultas de
    # dominio (Q0 y Q1), aunque el filtro lo haya mandado a otra categoría.
    nucleo = {g['id'] for g in grupos if dec[g['id']] == 'RETENER_DOMINIO' or DOMINIO.search(texto(g))
              or any(re.search(r'_Q[01]_', r['origen']) for r in g['registros'])}
    rng = random.Random(SEMILLA)
    muestra = {}
    for est, n in MUESTRA.items():
        ids = sorted(g['id'] for g in grupos if estrato(dec[g['id']]) == est and g['id'] not in nucleo)
        for i in rng.sample(ids, min(n, len(ids))):
            muestra[i] = est
    por_id = {g['id']: g for g in grupos}
    with open(ROOT / 'cribado_manual.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, CAMPOS)
        w.writeheader()
        for i in sorted(nucleo) + sorted(muestra):
            g = por_id[i]
            w.writerow({'id': i, 'estrato': 'NUCLEO_DOMINIO' if i in nucleo else muestra[i],
                        'decision_automatica': dec[i], 'anio': g['registros'][0]['anio'], 'titulo': g['titulo'],
                        'doi': g['doi'], 'nivel': 'texto completo' if i in nucleo else 'título y resumen'})
    print(json.dumps({'grupos': len(grupos), 'origen_decisiones': origen,
                      'conteo_automatico': Counter(dec.values()), 'nucleo_dominio': len(nucleo),
                      'muestra': Counter(muestra.values())}, ensure_ascii=False, indent=1))


def ampliar():
    """Agrega a cribado_manual.csv, sin tocar las filas existentes ni la muestra ya sorteada, los grupos nuevos tras
    una exportación posterior: todo grupo con algún registro de las consultas transversales (Q5, Q6) va al estrato
    TRANSVERSAL, que se revisa completo (censo, no muestra); un grupo nuevo que cumpla la regla del núcleo va a
    NUCLEO_DOMINIO."""
    grupos = json.loads((ROOT / 'registros_conciliados.json').read_text(encoding='utf-8'))
    dec = json.loads((ROOT / 'precribado_registros.json').read_text(encoding='utf-8'))
    ruta = ROOT / 'cribado_manual.csv'
    filas = list(csv.DictReader(open(ruta, encoding='utf-8')))
    ya = {f['id'] for f in filas}
    nuevas = Counter()
    for g in grupos:
        if g['id'] in ya:
            continue
        if dec[g['id']] == 'RETENER_DOMINIO' or DOMINIO.search(texto(g)) or \
                any(re.search(r'_Q[01]_', r['origen']) for r in g['registros']):
            est = 'NUCLEO_DOMINIO'
        elif any(re.search(r'_Q[56]_', r['origen']) for r in g['registros']):
            est = 'TRANSVERSAL'
        else:
            continue
        filas.append({'id': g['id'], 'estrato': est, 'decision_automatica': dec[g['id']],
                      'anio': g['registros'][0]['anio'], 'titulo': g['titulo'], 'doi': g['doi'],
                      'nivel': 'texto completo' if est == 'NUCLEO_DOMINIO' else 'título y resumen'})
        nuevas[est] += 1
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, CAMPOS)
        w.writeheader()
        w.writerows({k: f.get(k, '') for k in CAMPOS} for f in filas)
    print(f'Filas nuevas: {dict(nuevas)}; total {len(filas)}')


def registrar(*rutas):
    """Vuelca en cribado_manual.csv decisiones de archivos JSON [{id, decision_manual, motivo, clasificacion, nota,
    (segmenta_rotulo_por_pixel), (compara_arquitecturas_con_semillas)}]. Falla si un ID no está en la hoja."""
    ruta = ROOT / 'cribado_manual.csv'
    filas = list(csv.DictReader(open(ruta, encoding='utf-8')))
    por_id = {f['id']: f for f in filas}
    n = 0
    for r in rutas:
        for d in json.loads(Path(r).read_text(encoding='utf-8')):
            if d['id'] not in por_id:
                sys.exit(f"{d['id']} no está en cribado_manual.csv")
            por_id[d['id']].update({k: v for k, v in d.items() if k in CAMPOS and k != 'id'})
            n += 1
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, CAMPOS)
        w.writeheader()
        w.writerows(filas)
    print(f'{n} decisiones registradas; faltan {sum(not f["decision_manual"] for f in filas)}')


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return (round((p + z * z / (2 * n) - c) / d, 4), round((p + z * z / (2 * n) + c) / d, 4))


def cerrar():
    filas = list(csv.DictReader(open(ROOT / 'cribado_manual.csv', encoding='utf-8')))
    vacias = [f['id'] for f in filas if f['decision_manual'] not in ('incluir', 'excluir')]
    if vacias:
        sys.exit(f'{len(vacias)} filas sin decisión manual (incluir/excluir), p. ej. {vacias[:3]}')
    malos = [f['id'] for f in filas if f['decision_manual'] == 'excluir' and f['motivo'] not in MOTIVOS]
    if malos:
        sys.exit(f'Exclusiones sin motivo del protocolo: {malos[:5]}')
    dec = json.loads((ROOT / 'precribado_registros.json').read_text(encoding='utf-8'))
    conc = json.loads((ROOT / 'conciliacion.json').read_text(encoding='utf-8'))
    total_estrato = Counter(estrato(d) for d in dec.values())
    nucleo = [f for f in filas if f['estrato'] == 'NUCLEO_DOMINIO']
    transversal = [f for f in filas if f['estrato'] == 'TRANSVERSAL']
    muestra = [f for f in filas if f['estrato'] in MUESTRA]
    censo = nucleo + transversal  # revisados completos: no entran en la extrapolación de los estratos
    error = {}
    for est in MUESTRA:
        fs = [f for f in muestra if f['estrato'] == est]
        # EXCLUIR: incluibles perdidos por el filtro; RETENER_*: retenidos que se excluirían; DUDA: proporción
        # incluible (en DUDA no hay decisión automática que pueda errar).
        if est in ('EXCLUIR', 'DUDA'):
            k = sum(f['decision_manual'] == 'incluir' for f in fs)
        else:
            k = sum(f['decision_manual'] == 'excluir' for f in fs)
        lo, hi = wilson(k, len(fs))
        n_est = total_estrato[est] - sum(1 for f in censo if estrato(f['decision_automatica']) == est)
        ki = sum(f['decision_manual'] == 'incluir' for f in fs)
        li, hi_ = wilson(ki, len(fs))
        error[est] = {'revisados': len(fs), 'discrepancias': k, 'tasa': round(k / len(fs), 4) if fs else None,
                      'ic95_wilson': [lo, hi], 'registros_en_estrato': n_est,
                      'discrepancias_esperadas': [round(lo * n_est), round(hi * n_est)] if fs else None,
                      'incluibles_estimados': round(ki / len(fs) * n_est) if fs else None,
                      'incluibles_ic95': [round(li * n_est), round(hi_ * n_est)] if fs else None}
    incl = [f for f in nucleo if f['decision_manual'] == 'incluir']
    gap = {'denominador_estudios_dominio': len(incl),
           'segmentan_por_pixel': sum(f['segmenta_rotulo_por_pixel'] == 'si' for f in incl),
           'comparan_con_semillas': sum(f['compara_arquitecturas_con_semillas'] == 'si' for f in incl),
           'refutadores': [f['id'] for f in incl if 'si' in (f['segmenta_rotulo_por_pixel'],
                                                           f['compara_arquitecturas_con_semillas'])]}
    salida = {'identificados': conc['registros_importados'], 'duplicados': conc['duplicados_conciliados'],
              'grupos': conc['registros_unicos'], 'precribado_automatico': Counter(dec.values()),
              'nucleo_texto_completo': {'revisados': len(nucleo), 'incluidos': len(incl),
                                        'excluidos_por_motivo': Counter(f['motivo'] for f in nucleo
                                                                        if f['decision_manual'] == 'excluir'),
                                        'clasificacion_incluidos': Counter(f['clasificacion'] for f in incl)},
              'error_filtro': error, 'gap': gap}
    tin = [f for f in transversal if f['decision_manual'] == 'incluir']
    salida['transversal_q5_q6'] = {'revisados': len(transversal), 'incluidos': len(tin),
                                   'excluidos_por_motivo': Counter(f['motivo'] for f in transversal
                                                                   if f['decision_manual'] == 'excluir'),
                                   'clasificacion_incluidos': Counter(f['clasificacion'] for f in tin)}
    (ROOT / 'prisma_final.json').write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding='utf-8')
    md = ['# PRISMA del núcleo de dominio y error del precribado', '',
          'Generado por `cribado_nucleo.py cerrar` desde `cribado_manual.csv` (decisiones en '
          '`decisiones_cribado_muestra.json` y `decisiones_cribado_nucleo.json`; criterios en '
          '`CRITERIOS_ELEGIBILIDAD.md`). Decisiones propuestas por un asistente de IA con criterios fijos; auditoría de '
          'consistencia en `auditoria_cribado.json`. Sin kappa: no hay dos revisores humanos. Precribado: '
          '`precribado_reconstruido.py`.', '', '```text',
          f"Identificados: {salida['identificados']}  ->  duplicados: {salida['duplicados']}  ->  grupos: {salida['grupos']}",
          f"Precribado automático: {dict(salida['precribado_automatico'])}",
          f"Núcleo de dominio revisado: {len(nucleo)} (texto completo leído en "
          f"{sum(f['nota'].startswith('Texto completo') for f in nucleo)}; el resto con el resumen, ver nota)  ->  "
          f"incluidos: {len(incl)}",
          f"Excluidos con motivo: {dict(salida['nucleo_texto_completo']['excluidos_por_motivo'])}",
          f"Transversales Q5/Q6 no revisados antes (censo, título y resumen): {len(transversal)}  ->  incluidos: "
          f"{len(tin)}; excluidos con motivo: {dict(salida['transversal_q5_q6']['excluidos_por_motivo'])}", '```', '',
          'Discrepancia: en EXCLUIR, registro que se incluiría; en RETENER, registro que se excluiría; en DUDA, '
          'registro incluible.', '',
          '| Estrato del precribado | Revisados | Discrepancias | Tasa [IC95 Wilson] | Registros del estrato | Discrepancias esperadas | Incluibles estimados [IC95] |',
          '|---|---:|---:|---|---:|---|---|']
    for est, e in error.items():
        md.append(f"| {est} | {e['revisados']} | {e['discrepancias']} | {e['tasa']} {e['ic95_wilson']} | "
                  f"{e['registros_en_estrato']} | {e['discrepancias_esperadas']} | {e['incluibles_estimados']} "
                  f"{e['incluibles_ic95']} |")
    tot = sum(e['incluibles_estimados'] for e in error.values())
    md += ['', f'Incluibles a título y resumen fuera del núcleo, estimados por estratos: {tot} '
           f'(suma de los IC95 por estrato: {sum(e["incluibles_ic95"][0] for e in error.values())}–'
           f'{sum(e["incluibles_ic95"][1] for e in error.values())}; intervalo conservador, no simultáneo). No se '
           'evaluaron a texto completo: quedan fuera del alcance de la síntesis, que se limita al núcleo de dominio y a '
           'las arquitecturas puntuadas.']
    md += ['', f"**Gap.** Denominador: {gap['denominador_estudios_dominio']} estudios de dominio incluidos; "
           f"segmentan el rótulo por píxel con red profunda: {gap['segmentan_por_pixel']}; comparan arquitecturas con "
           f"varias semillas: {gap['comparan_con_semillas']}. Refutadores: {gap['refutadores'] or 'ninguno'}."]
    (ROOT / 'PRISMA_FINAL.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    accion = sys.argv[1] if len(sys.argv) > 1 else ''
    if accion == 'registrar':
        registrar(*sys.argv[2:])
    else:
        {'preparar': preparar, 'ampliar': ampliar, 'cerrar': cerrar}.get(accion, lambda: sys.exit(__doc__))()
