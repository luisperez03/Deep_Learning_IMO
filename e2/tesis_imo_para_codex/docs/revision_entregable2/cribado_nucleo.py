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
DOMINIO = re.compile(r'hazmat|hazardous material|dangerous good|placard|\bimdg\b|\bghs\b|hazard (?:label|sign)|'
                     r'warning (?:label|sign)|\bun number\b', re.I)
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
        dec, origen = json.loads(ruta.read_text(encoding='utf-8')), 'precribado original por registro'
    else:
        dec = {g['id']: next((c for c, rx in REGLAS if rx.search(texto(g))), 'DUDA') for g in grupos}
        ruta.write_text(json.dumps(dec, ensure_ascii=False, indent=0), encoding='utf-8')
        origen = 'REGLAS de respaldo de cribado_nucleo.py'
    faltan = [g['id'] for g in grupos if g['id'] not in dec]
    if faltan:
        sys.exit(f'{len(faltan)} grupos sin decisión automática (p. ej. {faltan[:3]})')
    nucleo = {g['id'] for g in grupos if dec[g['id']] == 'RETENER_DOMINIO' or DOMINIO.search(texto(g))}
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
    muestra = [f for f in filas if f['estrato'] != 'NUCLEO_DOMINIO']
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
        n_est = total_estrato[est] - sum(1 for f in nucleo if estrato(f['decision_automatica']) == est)
        error[est] = {'revisados': len(fs), 'discrepancias': k, 'tasa': round(k / len(fs), 4) if fs else None,
                      'ic95_wilson': [lo, hi], 'registros_en_estrato': n_est,
                      'discrepancias_esperadas': [round(lo * n_est), round(hi * n_est)] if fs else None}
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
    (ROOT / 'prisma_final.json').write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding='utf-8')
    md = ['# PRISMA del núcleo de dominio y error del precribado', '',
          'Generado por `cribado_nucleo.py cerrar` desde `cribado_manual.csv`. Revisión asistida por IA validada por el '
          'investigador; sin kappa (un solo revisor).', '', '```text',
          f"Identificados: {salida['identificados']}  ->  duplicados: {salida['duplicados']}  ->  grupos: {salida['grupos']}",
          f"Precribado automático: {dict(salida['precribado_automatico'])}",
          f"Núcleo de dominio a texto completo: {len(nucleo)}  ->  incluidos: {len(incl)}",
          f"Excluidos con motivo: {dict(salida['nucleo_texto_completo']['excluidos_por_motivo'])}", '```', '',
          'Discrepancia: en EXCLUIR, registro que se incluiría; en RETENER, registro que se excluiría; en DUDA, '
          'registro incluible.', '',
          '| Estrato del precribado | Revisados | Discrepancias | Tasa [IC95 Wilson] | Registros del estrato | Discrepancias esperadas |',
          '|---|---:|---:|---|---:|---|']
    for est, e in error.items():
        md.append(f"| {est} | {e['revisados']} | {e['discrepancias']} | {e['tasa']} {e['ic95_wilson']} | "
                  f"{e['registros_en_estrato']} | {e['discrepancias_esperadas']} |")
    md += ['', f"**Gap.** Denominador: {gap['denominador_estudios_dominio']} estudios de dominio incluidos; "
           f"segmentan el rótulo por píxel con red profunda: {gap['segmentan_por_pixel']}; comparan arquitecturas con "
           f"varias semillas: {gap['comparan_con_semillas']}. Refutadores: {gap['refutadores'] or 'ninguno'}."]
    (ROOT / 'PRISMA_FINAL.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    {'preparar': preparar, 'cerrar': cerrar}.get(sys.argv[1] if len(sys.argv) > 1 else '', lambda: sys.exit(__doc__))()
