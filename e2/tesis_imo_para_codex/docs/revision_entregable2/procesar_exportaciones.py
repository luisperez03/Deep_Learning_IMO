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
    previous_doi = {g['doi']: g for g in previous if g['doi']}
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
            doi = get('DO' if ris else 'DI').strip().lower().removeprefix('https://doi.org/')
            records.append({'origen': path.name, 'registro_origen': index,
                'titulo': get('TI'), 'doi': doi, 'anio': get('PY'),
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
            if old and old['id'] not in used_ids and (not old['doi'] or not r['doi'] or old['doi'] == r['doi']):
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
    report = {'archivos': files, 'registros_importados': len(records), 'registros_unicos': len(groups),
              'duplicados_conciliados': len(records)-len(groups), 'coincidencias_ambiguas': ambiguities,
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
