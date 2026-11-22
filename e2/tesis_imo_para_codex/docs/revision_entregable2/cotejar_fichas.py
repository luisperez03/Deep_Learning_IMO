"""Coteja contra el PDF de arXiv cada campo de las fichas de extracción y genera FICHAS_COTEJADAS.md.

Uso: python docs/revision_entregable2/cotejar_fichas.py <carpeta_pdfs>
  - Entrada: fichas_extraccion.json (un objeto por artículo: campo -> valor + evidencia [página, fragmento]).
  - Si un PDF falta en la carpeta, lo descarga de https://arxiv.org/pdf/<id> (los PDF no se versionan).
  - Texto con `pdftotext -enc UTF-8 -raw`; páginas separadas por salto de página; "página" es la del PDF.
  - Un campo queda COTEJADO si todos sus fragmentos aparecen literalmente en la página indicada; "No reportado"
    sin evidencia queda como AUSENCIA DECLARADA (la ausencia se juzgó leyendo el texto, no se prueba con
    búsqueda literal). Cualquier fragmento que no aparezca hace fallar el script.
Salidas: FICHAS_COTEJADAS.md y cotejo_fichas.json en esta carpeta.
"""
import datetime as dt
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

NOMBRES = {
    'tarea_modalidad': 'Tarea y modalidad', 'mecanismo': 'Arquitectura y mecanismo',
    'datos_particion': 'Datos, partición y unidad', 'preprocesamiento': 'Preprocesamiento, aumentos y preentrenamiento',
    'entrenamiento': 'Pérdida, optimizador, duración y checkpoint', 'metricas': 'Métricas y agregación',
    'repeticiones_incertidumbre': 'Semillas, repeticiones e incertidumbre', 'ablaciones': 'Ablaciones',
    'dificultad_compartida': 'Dificultad compartida con HAZMAT-13', 'costo': 'Hardware, parámetros, FLOPs y latencia',
    'codigo': 'Código declarado', 'seleccion_por_test': 'Riesgo de selección por test',
    'limitaciones': 'Limitaciones reconocidas',
}


def paginas(carpeta, arxiv):
    pdf = carpeta / f'{arxiv}.pdf'
    if not pdf.exists():
        req = urllib.request.Request(f'https://arxiv.org/pdf/{arxiv}', headers={'User-Agent': 'cotejo/1.0'})
        pdf.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    out = subprocess.run(['pdftotext', '-enc', 'UTF-8', '-raw', str(pdf), '-'], capture_output=True, check=True)
    return out.stdout.decode('utf-8', errors='replace').split('\f')


def main(carpeta):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    fichas = json.loads((ROOT / 'fichas_extraccion.json').read_text(encoding='utf-8'))
    resultado, fallas = [], []
    md = ['# Fichas de extracción cotejadas con el PDF', '',
          f'**{dt.date.today():%d/%m/%Y}.** Generado por `cotejar_fichas.py` desde `fichas_extraccion.json`. Cada valor '
          'lleva la página del PDF de arXiv y el fragmento literal que lo respalda; el script comprueba que el fragmento '
          'está en esa página. "No reportado" es una ausencia declarada tras leer el artículo. Las cifras de cabecera '
          '(dataset, resultado, varianza, costo) están cotejadas aparte en `RESULTADOS_ORIGINALES_7REDES.md`. La '
          'evaluación de calidad por componente está en `PUNTUACION_DEFINITIVA.md`.', '']
    for f in fichas:
        pags = paginas(carpeta, f['arxiv'])
        md += [f"## {f['red']} ({f['referencia']}; arXiv {f['arxiv']})", '', '| Campo | Valor | Evidencia (p. del PDF) |',
               '|---|---|---|']
        for clave, nombre in NOMBRES.items():
            campo = f['campos'][clave]
            evid = []
            for e in campo['evidencia']:
                ok = e['fragmento'] in pags[e['pagina'] - 1]
                resultado.append({'red': f['red'], 'campo': clave, 'pagina': e['pagina'], 'fragmento': e['fragmento'],
                                  'confirmado': ok})
                if not ok:
                    fallas.append((f['red'], clave, e['pagina'], e['fragmento']))
                evid.append(f"p. {e['pagina']}: «{e['fragmento']}»")
            estado = '; '.join(evid) if evid else 'ausencia declarada'
            md.append(f"| {nombre} | {campo['valor'].replace('|', '/')} | {estado.replace('|', '/')} |")
        md.append('')
    resumen = {'fecha': dt.date.today().isoformat(), 'articulos': len(fichas),
               'fragmentos': len(resultado), 'confirmados': sum(r['confirmado'] for r in resultado),
               'campos_sin_evidencia': sum(1 for f in fichas for c in f['campos'].values() if not c['evidencia']),
               'detalle': resultado}
    (ROOT / 'cotejo_fichas.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'FICHAS_COTEJADAS.md').write_text('\n'.join(md), encoding='utf-8')
    print(f"{resumen['confirmados']} de {resumen['fragmentos']} fragmentos confirmados en {len(fichas)} artículos; "
          f"{resumen['campos_sin_evidencia']} campos con ausencia declarada")
    if fallas:
        for x in fallas:
            print('FALLA', *x)
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Indique una carpeta FUERA del repositorio para los PDF (no se versionan).')
    main(sys.argv[1])
