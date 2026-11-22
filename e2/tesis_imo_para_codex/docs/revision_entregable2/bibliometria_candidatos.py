"""Bibliometría de las candidatas puntuadas: citas en OpenAlex y licencia del código oficial en GitHub.

Uso: python docs/revision_entregable2/bibliometria_candidatos.py
  - Citas: OpenAlex, misma base y misma fecha para todas (regla del protocolo, componente de citación).
    Cada candidata se busca por todos sus DOI (versión publicada y preprint de arXiv). Si OpenAlex tiene
    más de una obra, se registran todas y se usa el MÁXIMO, no la suma: OpenAlex fusiona a veces versiones
    y sumar contaría dos veces las mismas citas.
  - Licencia: campo `license` del repositorio oficial que declara el PDF (API de GitHub, vía `gh api`).
  - Año: el de la versión citada en el benchmark, fijado aquí (OpenAlex da a veces el año de publicación
    en línea, p. ej. 2019 para UNet++ de TMI 2020).
Salida: docs/revision_entregable2/bibliometria_candidatos.json
No asigna cuartiles: los de revistas se registran en puntuar_candidatos.py con su fuente.
"""
import datetime as dt
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# clave: (año de la versión citada, [DOI conocidos], repo oficial declarado en el PDF)
CANDIDATAS = {
    'DeepLabv3+':      (2018, ['10.1007/978-3-030-01234-2_49', '10.48550/arxiv.1802.02611'], 'tensorflow/models'),
    'UNet++':          (2020, ['10.1109/tmi.2019.2959609', '10.48550/arxiv.1912.05074'], 'MrGiovanni/UNetPlusPlus'),
    'PIDNet-S':        (2023, ['10.1109/cvpr52729.2023.01871', '10.48550/arxiv.2206.02066'], 'XuJiacong/PIDNet'),
    'SegNeXt-T':       (2022, ['10.48550/arxiv.2209.08575'], 'Visual-Attention-Network/SegNeXt'),
    'SegFormer-B0':    (2021, ['10.48550/arxiv.2105.15203'], 'NVlabs/SegFormer'),
    'TransUNet':       (2021, ['10.48550/arxiv.2102.04306'], 'Beckschen/TransUNet'),
    'VM-UNet':         (2024, ['10.1145/3767748', '10.48550/arxiv.2402.02491'], 'JCruan519/VM-UNet'),
    'Attention U-Net': (2018, ['10.48550/arxiv.1804.03999'], 'ozan-oktay/Attention-Gated-Networks'),
    'Swin-Unet':       (2022, ['10.1007/978-3-031-25066-8_9', '10.48550/arxiv.2105.05537'], 'HuCaoFighting/Swin-Unet'),
}


def openalex(doi):
    url = f'https://api.openalex.org/works/doi:{doi}?select=id,display_name,cited_by_count,publication_year,doi'
    for intento in range(5):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'bibliometria-e2/1.0'})
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(10 * (intento + 1))
    raise RuntimeError(f'OpenAlex no respondió para {doi}')


def licencia(repo):
    out = subprocess.run(['gh', 'api', f'repos/{repo}', '--jq', '[.license.spdx_id, .license.name, .archived, .pushed_at] | @json'],
                         capture_output=True, text=True)
    if out.returncode:
        return {'repo': repo, 'error': out.stderr.strip()[:200]}
    spdx, nombre, archivado, ultimo = json.loads(out.stdout)
    return {'repo': repo, 'spdx': spdx, 'nombre': nombre, 'archivado': archivado, 'ultimo_push': ultimo}


def main():
    hoy = dt.date.today()
    filas = {}
    for red, (anio, dois, repo) in CANDIDATAS.items():
        obras = []
        for doi in dois:
            w = openalex(doi)
            if w and w['id'] not in {o['openalex'] for o in obras}:
                obras.append({'doi_consultado': doi, 'openalex': w['id'], 'titulo': w['display_name'],
                              'citas': w['cited_by_count'], 'anio_openalex': w['publication_year']})
            time.sleep(1)
        citas = max((o['citas'] for o in obras), default=None)
        edad = max(1, hoy.year - anio + 1)
        filas[red] = {'anio_version_citada': anio, 'obras_openalex': obras, 'citas_max': citas,
                      'citas_por_anio': round(citas / edad, 1) if citas is not None else None,
                      'codigo': licencia(repo)}
        print(red, citas, filas[red]['citas_por_anio'], filas[red]['codigo'].get('spdx'))
    salida = {'fecha_consulta': hoy.isoformat(), 'fuente_citas': 'OpenAlex (api.openalex.org), máximo entre obras',
              'regla': 'citas/max(1, año_consulta - año + 1)', 'candidatas': filas}
    (ROOT / 'bibliometria_candidatos.json').write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
