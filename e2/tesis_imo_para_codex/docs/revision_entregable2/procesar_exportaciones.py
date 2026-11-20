"""Concilia exportaciones bibliográficas locales sin modificar archivos originales.

Uso: python docs/revision_entregable2/procesar_exportaciones.py
No descarga datos, no consulta bases, no decide elegibilidad científica.
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Alias de DOI adjudicados con el registro oficial (Crossref y doi.org) el 20/11/2026.
# Evidencia: RESOLUCION_CONFLICTOS_DOI.md y evidencia_doi_registro.json (verificar_doi_registro.py).
# Clave: DOI tal como llega en la exportacion (minusculas). Valor: DOI canonico.
# El DOI original de cada registro se conserva en 'doi_original'; el ID del grupo absorbido,
# en 'ids_absorbidos' del grupo que lo recibe. Ningun ID se renumera.
DOI_ALIAS = {
    '10.7717/peerjcs.2146': '10.7717/peerj-cs.2146',            # no registrado (404 Crossref y doi.org)
    '10.1504/ijes.2021.10039166': '10.1504/ijes.2021.116134',   # alias registrado: doi.org 301 al canonico
    '10.3788/cjl202118.1507001': '10.3788/cjl202148.1507004',   # no registrado (404 Crossref y doi.org)
}


def canon(doi):
    return DOI_ALIAS.get(doi, doi) if doi else doi


def parse_tagged(path, ris):
    records, current, key = [], {}, None
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        match = re.match(r'^([A-Z0-9]{2})  - ?(.*)$' if ris else r'^([A-Z0-9]{2}) ?(.*)$', line)
        if match:
            key, value = match.groups()
            if key == 'ER':
                if current:
                    records.append(current)
                current, key = {}, None
            elif key not in ('FN', 'VR', 'EF'):
                current.setdefault(key, []).append(value)
        elif key and line.strip():
            current[key][-1] += ' ' + line.strip()
    if current:
        raise ValueError(f'Registro sin terminador ER en {path.name}')
    return records


def norm_title(title):
    title = title.replace('[Formula presented]', '')
    return re.sub(r'\W+', '', unicodedata.normalize('NFKD', title).casefold())


def main():
    previous_path = ROOT / 'registros_conciliados.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else []
    previous_doi = {}
    for g in previous:
        key = canon(g['doi'])
        if key and (key not in previous_doi or g['doi'] == key):
            previous_doi[key] = g
    previous_record_id = {(r['origen'], r['registro_origen']): g['id'] for g in previous for r in g['registros']}
    previous_title = {(norm_title(r['titulo']), r['anio']): g for g in previous for r in g['registros']}
    next_id = max((int(g['id'].split('-')[1]) for g in previous), default=0) + 1
    used_ids = set()
    records, files = [], []
    for path in sorted((ROOT / 'exportaciones').glob('*')):
        if path.suffix not in ('.ris', '.txt'):
            continue
        ris = path.suffix == '.ris'
        raw = parse_tagged(path, ris)
        files.append({'archivo': path.name, 'registros': len(raw), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        for index, item in enumerate(raw, 1):
            get = lambda tag: ' '.join(item.get(tag, []))
            doi_original = get('DO' if ris else 'DI').strip().lower().removeprefix('https://doi.org/')
            doi = canon(doi_original)
            records.append({'origen': path.name, 'registro_origen': index,
                'titulo': get('TI'), 'doi': doi, 'doi_original': doi_original, 'anio': get('PY'),
                'autores': item.get('AU', []), 'resumen': get('AB'),
                'revista': get('T2') or get('JO') or get('SO'),
                'tipo': get('TY') or get('DT'), 'idioma': get('LA'),
                'id_base': get('UT') or get('ID'), 'metadatos_originales': item})
    groups, doi_map, title_map, ambiguities = [], {}, {}, []
    for r in records:
        title_key = (norm_title(r['titulo']), r['anio'])
        idx = doi_map.get(r['doi']) if r['doi'] else None
        if idx is None and title_key[0] and title_key in title_map:
            candidate = title_map[title_key]
            existing = {v['doi'] for v in groups[candidate]['registros'] if v['doi']}
            if not r['doi'] or not existing or r['doi'] in existing:
                idx = candidate
            else:
                ambiguities.append({'titulo': r['titulo'], 'dois': sorted(existing | {r['doi']})})
        if idx is None:
            idx = len(groups)
            old = previous_doi.get(r['doi']) or previous_title.get(title_key)
            if old and old['id'] not in used_ids and (not old['doi'] or not r['doi'] or canon(old['doi']) == r['doi']):
                stable_id = old['id']
            else:
                stable_id = f'E2-{next_id:04d}'
                next_id += 1
            used_ids.add(stable_id)
            groups.append({'id': stable_id, 'titulo': r['titulo'], 'doi': r['doi'],
                'estado_cribado': 'pendiente', 'decision': None, 'motivo': None,
                'texto_completo_leido': False, 'registros': []})
            if old and old['id'] == stable_id:
                for field in ('estado_cribado', 'decision', 'motivo', 'texto_completo_leido'):
                    groups[-1][field] = old[field]
        groups[idx]['registros'].append(r)
        if r['doi']:
            doi_map[r['doi']] = idx
        title_map[title_key] = idx
    alias_aplicados = {}
    for group in groups:
        absorbed = {previous_record_id.get((v['origen'], v['registro_origen'])) for v in group['registros']}
        group['ids_absorbidos'] = sorted(i for i in absorbed if i and i != group['id'])
        for v in group['registros']:
            if v['doi_original'] != v['doi']:
                a = alias_aplicados.setdefault(v['doi_original'], {'alias': v['doi_original'], 'canonico': v['doi'],
                                                                   'grupo': group['id'], 'registros': 0})
                a['registros'] += 1
    report = {'archivos': files, 'registros_importados': len(records), 'registros_unicos': len(groups),
              'duplicados_conciliados': len(records)-len(groups), 'coincidencias_ambiguas': ambiguities,
              'alias_doi_aplicados': list(alias_aplicados.values()),
              'ids_absorbidos': {g['id']: g['ids_absorbidos'] for g in groups if g['ids_absorbidos']},
              'nota': 'Conciliación de exportaciones disponibles; no es PRISMA final ni inclusión científica.'}
    (ROOT / 'registros_conciliados.json').write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'conciliacion.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Matriz inicial de cribado', '', 'Generada de exportaciones reales. Todos los registros siguen pendientes de cribado científico.', '',
        '| ID | Año | Título | DOI | Fuentes | Estado |', '|---|---|---|---|---|---|']
    for group in groups:
        r = group['registros'][0]
        sources = ', '.join(sorted({'Scopus' if v['origen'].startswith('scopus') else 'WoS' for v in group['registros']}))
        lines.append(f"| {group['id']} | {r['anio']} | {r['titulo'].replace('|','/')} | {group['doi']} | {sources} | Pendiente |")
    (ROOT / 'MATRIZ_CRIBADO.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
