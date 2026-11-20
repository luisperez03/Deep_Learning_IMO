"""Verifica DOI contra el registro oficial (Crossref o DataCite) y guarda la evidencia.

Uso: python docs/revision_entregable2/verificar_doi_registro.py
Salida: docs/revision_entregable2/evidencia_doi_registro.json

Para cada DOI guarda: fecha de consulta, codigo HTTP de Crossref, de DataCite y de doi.org
(sin seguir redirecciones), destino de la redireccion, y los metadatos del registro (titulo,
autores, contenedor, volumen, numero, paginas, anio). No modifica ningun otro archivo ni
decide elegibilidad. Un DOI con 404 en Crossref, DataCite y doi.org no esta registrado.
"""
import datetime as dt
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UA = {'User-Agent': 'verificacion-bibliografica/1.0'}

# (grupo, doi, nota). Los pares en conflicto conservan ambos DOI.
DOIS = [
    ('conflicto_1', '10.7717/peerj-cs.2146', 'E2-3569/E2-5674'),
    ('conflicto_1', '10.7717/peerjcs.2146', 'E2-3569/E2-5674'),
    ('conflicto_2', '10.1504/ijes.2021.116134', 'E2-5639/E2-5682'),
    ('conflicto_2', '10.1504/ijes.2021.10039166', 'E2-5639/E2-5682'),
    ('conflicto_3', '10.3788/cjl202148.1507004', 'E2-5459/E2-5878'),
    ('conflicto_3', '10.3788/cjl202118.1507001', 'E2-5459/E2-5878'),
    ('conflicto_3', '10.3788/cjl202148.1507001', 'control: articulo 1507001 del mismo numero'),
    ('aplicacion', '10.1016/j.mlwa.2021.100104', ''),
    ('aplicacion', '10.1109/tai.2023.3298588', ''),
    ('aplicacion', '10.1109/ssrr.2018.8468639', ''),
    ('aplicacion', '10.2352/issn.2470-1173.2019.7.iriacv-463', ''),
    ('aplicacion', '10.1109/globalsip.2013.6736996', ''),
    ('aplicacion', '10.2352/issn.2470-1173.2016.11.imawm-459', ''),
    ('aplicacion', '10.48550/arXiv.2007.06392', 'DeepHAZMAT preprint'),
    ('arquitectura', '10.1109/CVPR.2015.7298965', 'FCN'),
    ('arquitectura', '10.1109/TPAMI.2016.2644615', 'SegNet'),
    ('arquitectura', '10.1007/978-3-319-24574-4_28', 'U-Net'),
    ('arquitectura', '10.1007/978-3-030-01234-2_49', 'DeepLabv3+'),
    ('arquitectura', '10.1109/TMI.2019.2959609', 'UNet++'),
    ('arquitectura', '10.1109/CVPR52729.2023.01871', 'PIDNet'),
    ('arquitectura', '10.1145/3767748', 'VM-UNet'),
    ('arquitectura', '10.48550/arXiv.2209.08575', 'SegNeXt'),
    ('arquitectura', '10.48550/arXiv.2105.15203', 'SegFormer'),
    ('arquitectura', '10.48550/arXiv.2102.04306', 'TransUNet'),
    ('arquitectura', '10.48550/arXiv.2402.02491', 'VM-UNet preprint'),
    ('mecanismo', '10.48550/arXiv.1706.03762', 'Transformer'),
    ('mecanismo', '10.48550/arXiv.2312.00752', 'Mamba'),
    ('mecanismo', '10.48550/arXiv.1511.07122', 'Yu y Koltun'),
    ('mecanismo', '10.48550/arXiv.1609.03499', 'WaveNet'),
]


def get(url, follow=True):
    req = urllib.request.Request(url, headers=UA, method='GET')
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(req, timeout=30) as r:
            return r.status, r.read(), r.headers.get('Location')
    except urllib.error.HTTPError as e:
        return e.code, b'', e.headers.get('Location')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def crossref_meta(m):
    return {'doi': m.get('DOI'), 'titulo': (m.get('title') or [''])[0],
            'titulo_original': (m.get('original-title') or [''])[0],
            'autores': [' '.join(filter(None, [a.get('given'), a.get('family')])) or a.get('name', '')
                        for a in m.get('author', [])],
            'contenedor': (m.get('container-title') or [''])[0], 'volumen': m.get('volume'),
            'numero': m.get('issue'), 'paginas': m.get('page'), 'articulo': m.get('article-number'),
            'anio': (m.get('issued', {}).get('date-parts') or [[None]])[0][0], 'tipo': m.get('type'),
            'editorial': m.get('publisher')}


def datacite_meta(d):
    a = d['data']['attributes']
    return {'doi': a.get('doi'), 'titulo': (a.get('titles') or [{}])[0].get('title'),
            'autores': [c.get('name') for c in a.get('creators', [])],
            'contenedor': a.get('publisher'), 'anio': a.get('publicationYear'),
            'tipo': (a.get('types') or {}).get('resourceTypeGeneral')}


def main():
    out = []
    for grupo, doi, nota in DOIS:
        q = urllib.parse.quote(doi, safe='/')
        cr_code, cr_body, cr_loc = get(f'https://api.crossref.org/works/{q}', follow=False)
        dc_code, dc_body, _ = get(f'https://api.datacite.org/dois/{q}')
        org_code, _, org_loc = get(f'https://doi.org/{q}', follow=False)
        meta = None
        if cr_code == 200:
            meta = crossref_meta(json.loads(cr_body)['message'])
        elif dc_code == 200:
            meta = datacite_meta(json.loads(dc_body))
        out.append({'grupo': grupo, 'doi': doi, 'nota': nota,
                    'consultado': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                    'crossref_http': cr_code, 'crossref_location': cr_loc,
                    'datacite_http': dc_code, 'doi_org_http': org_code, 'doi_org_destino': org_loc,
                    'metadatos': meta})
        print(f'{doi:45s} crossref={cr_code} datacite={dc_code} doi.org={org_code} -> {org_loc}')
    (ROOT / 'evidencia_doi_registro.json').write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                     encoding='utf-8')


if __name__ == '__main__':
    main()
