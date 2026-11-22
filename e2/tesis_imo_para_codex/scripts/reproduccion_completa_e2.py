"""Reproducción completa en un clon limpio desde GitHub, con los datos externos declarados (checklist 55, 68, B20, B21).

  python scripts/reproduccion_completa_e2.py --destino CARPETA_NUEVA --voc RUTA/VOC2012 --test RUTA/resultados_test_e2_final
         --python PYTHON [--python-alt PYTHON2] [--historias RUTA/tesis_imo_e2_arquitecturas_reales_v2]
         [--jupyter-book RUTA/jupyter-book] [--url URL] [--rama cierre-e2] [--sin-locales]

Qué hace, en orden:
1. Clona la rama (por defecto `cierre-e2`) de GitHub con `git clone --depth 1` en CARPETA_NUEVA/repo. Solo lo publicado:
   nada del disco de trabajo entra al clon.
2. Pone los datos externos como HERMANOS del proyecto, donde los buscan los scripts: `e2/VOC2012` y
   `e2/resultados_test_e2_final` (enlaces de directorio) y, si se dan, las historias de entrenamiento. Calcula el
   SHA-256 de cada archivo del test cerrado (50 JSON, resúmenes y 2.180 máscaras) y la huella del conjunto.
3. Reconstruye las 1.451 máscaras binarias (verificadas por SHA-256 contra el manifiesto).
4. Corre, con un intérprete aislado (-E -s), la cadena que produce las tablas y figuras del test:
   auditar_test_e2.py (verifica firmas, hashes y recalcula las métricas de las 2.180 máscaras de semilla 11),
   graficar_test_e2.py, analisis_integracion_e2.py (contrastes, Friedman, estratos, figuras), integrar_entregable2.py,
   completar_analisis_locales_e2.py (ya con las máscaras de test) y todos los archivos de tests.
   Sin historias de entrenamiento, analisis_integracion_e2.py corre con las dos figuras de curvas en blanco; esas dos
   figuras se restauran desde git y se reportan como NO reproducidas.
5. Compara cada archivo que la cadena reescribió con su versión publicada (`git show HEAD:...` en el clon):
   JSON, discrepancia numérica máxima; Markdown, líneas distintas; PNG, píxeles distintos.
6. Recalcula de forma INDEPENDIENTE (sin importar código del proyecto) el Dice por imagen y por grupo de las 10 redes
   de semilla 11 desde las máscaras predichas y las máscaras de referencia originales de VOC2012, y la tabla de test
   (media, DE entre semillas, diferencias contra FCN) desde los 50 JSON; lo contrasta con los JSON, con
   RESUMEN_TEST.json, con la auditoría publicada y con resultados/integracion_e2/analisis.json. Añade la discrepancia de
   la reinferencia en GPU ya publicada (resultados/diagnosticos_gpu_e2/RESUMEN_DIAGNOSTICOS.json, etapa clean).
7. Compila el libro con jupyter-book (-W --keep-going) y ejecuta los cuadernos que no piden GPU ni Drive; clasifica el
   resto leyendo su código. Antes genera resultados/inicio_voc y revision_voc (no versionados), que leen los cuadernos
   de EDA. Las celdas se ejecutan en orden con el intérprete limpio (sin kernel de Jupyter): el kernel disponible del
   libro trae un Pillow sin `load_default(size=)`. Con `--solo-cuadernos CLON JUPYTER_BOOK [PYTHON]` se repite solo
   este paso sobre un clon ya hecho.

Escribe resultados/reproduccion_completa_e2/REPORTE.json en el proyecto ORIGINAL. No borra nada: si CARPETA_NUEVA
existe, se detiene. Para borrar el clon después, quitar antes los enlaces con `rmdir` (no siguen al destino).
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

P = Path(__file__).resolve().parents[1]
REL = 'e2/tesis_imo_para_codex'
OUT = P / 'resultados/reproduccion_completa_e2'
URL = 'https://github.com/luisperez03/Deep_Learning_IMO.git'
HIST = 'tesis_imo_e2_arquitecturas_reales_v2'
CURVAS = ['libro/figuras/e2/curvas_validacion.png', 'libro/figuras/e2/curvas_perdida.png']


# ---------------------------------------------------------------- utilidades
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def correr(cmd, cwd, env, timeout=7200):
    t = time.time()
    try:
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=timeout)
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = 'timeout', '', ''
    return {'orden': ' '.join(str(c) for c in cmd[1:]), 'codigo_salida': rc, 'segundos': round(time.time() - t, 1),
            'salida_final': out.strip().splitlines()[-6:],
            'error_final': [x for x in err.strip().splitlines() if x.strip()][-8:], '_stdout': out, '_stderr': err}


def limpio(r):
    return {k: v for k, v in r.items() if not k.startswith('_')}


def hojas(x, pre=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from hojas(v, f'{pre}/{k}')
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from hojas(v, f'{pre}[{i}]')
    else:
        yield pre, x


def comparar_json(a, b):
    ha, hb = dict(hojas(a)), dict(hojas(b))
    dmax, donde, otras, ejemplos = 0.0, None, 0, []
    for k in sorted(set(ha) | set(hb)):
        x, y = ha.get(k), hb.get(k)
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        if num(x) and num(y):
            d = abs(float(x) - float(y))
            if d != d:  # NaN en los dos lados
                continue
            if d > dmax:
                dmax, donde = d, k
        elif x != y:
            otras += 1
            if len(ejemplos) < 3:
                ejemplos.append({'hoja': k, 'nuevo': str(x)[:120], 'publicado': str(y)[:120]})
    return {'tipo': 'json', 'discrepancia_max': dmax, 'donde': donde, 'hojas': len(ha),
            'hojas_no_numericas_distintas': otras, 'ejemplos': ejemplos}


def comparar_texto(a, b):
    la, lb = a.splitlines(), b.splitlines()
    distintas = [(i, x, y) for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    return {'tipo': 'texto', 'lineas_nuevo': len(la), 'lineas_publicado': len(lb), 'lineas_distintas': len(distintas),
            'ejemplos': [{'linea': i + 1, 'nuevo': x[:160], 'publicado': y[:160]} for i, x, y in distintas[:3]]}


def comparar_png(a, b):
    import io
    import numpy as np
    from PIL import Image
    x = np.asarray(Image.open(io.BytesIO(a)).convert('RGBA'), dtype=np.int16)
    y = np.asarray(Image.open(io.BytesIO(b)).convert('RGBA'), dtype=np.int16)
    if x.shape != y.shape:
        return {'tipo': 'png', 'misma_forma': False, 'forma_nuevo': list(x.shape), 'forma_publicado': list(y.shape)}
    d = np.abs(x - y).max(axis=2)
    return {'tipo': 'png', 'misma_forma': True, 'forma': list(x.shape[:2]), 'pixeles_distintos': int((d > 0).sum()),
            'fraccion_pixeles_distintos': float((d > 0).mean()), 'diferencia_max_canal': int(d.max())}


# ------------------------------------------- recálculo independiente (sin código del proyecto)
def recalculo_independiente(copia, test, publicado):
    """Dice por imagen y grupo desde máscaras; tabla y contrastes desde los 50 JSON. Solo numpy y Pillow."""
    import csv
    import numpy as np
    from PIL import Image
    size = 256
    with open(copia / 'resultados/preentrenamiento/test.csv', encoding='utf-8-sig') as f:
        filas = list(csv.DictReader(f))
    raiz_datos = copia.parent

    def referencia(fila):
        """Máscara original de VOC: todo color distinto de negro es etiqueta. Letterbox a 256 con vecino más cercano."""
        with Image.open(raiz_datos / fila['original_mask_path']) as im:
            rgb = np.asarray(im.convert('RGB'))
        y = (rgb != 0).any(axis=2)
        h, w = y.shape
        s = min(size / w, size / h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        r = np.asarray(Image.fromarray(y.astype(np.uint8) * 255).resize((nw, nh), Image.Resampling.NEAREST)) > 0
        lienzo, valido = np.zeros((size, size), bool), np.zeros((size, size), bool)
        x0, y0 = (size - nw) // 2, (size - nh) // 2
        lienzo[y0:y0 + nh, x0:x0 + nw] = r
        valido[y0:y0 + nh, x0:x0 + nw] = True
        return lienzo, valido

    def dice(p, t):
        tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0

    refs = {f['id']: referencia(f) for f in filas}
    grupo = {f['id']: f['group_id'] for f in filas}
    leer = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    corridas = {p.stem: leer(p) for p in sorted(test.glob('*_seed*.json'))}
    modelos = sorted({r['model'] for r in corridas.values()})
    semillas = sorted({r['seed'] for r in corridas.values()})
    R = {'imagenes_test': len(filas), 'grupos_test': len(set(grupo.values())), 'corridas': len(corridas),
         'mascaras': {}, 'tabla': {}, 'contrastes_vs_fcn': {}}

    # 1. Máscaras de semilla 11: Dice por imagen y por grupo, contra lo registrado en los JSON.
    dmax_img = dmax_grp = 0.0
    n_masc = 0
    for m in modelos:
        run = corridas[f'{m}_seed11']
        por_img = {}
        for d in run['details']:
            with Image.open(test / 'mascaras_seed11' / m / f"{d['id']}.png") as im:
                pred = np.asarray(im) > 0
            t, v = refs[d['id']]
            por_img[d['id']] = dice(pred[v], t[v])
            n_masc += 1
        g = defaultdict(list)
        for i, x in por_img.items():
            g[grupo[i]].append(x)
        por_grp = {k: float(np.mean(v)) for k, v in g.items()}
        di = max(abs(por_img[d['id']] - d['Dice']) for d in run['details'])
        dg = max(abs(por_grp[k] - run['groups'][k]['Dice']) for k in por_grp)
        dmax_img, dmax_grp = max(dmax_img, di), max(dmax_grp, dg)
        R['mascaras'][m] = {'imagenes': len(por_img), 'grupos': len(por_grp), 'disc_max_imagen': di,
                            'disc_max_grupo': dg, 'Dice_grupos_recalculado': float(np.mean(list(por_grp.values()))),
                            'Dice_grupos_registrado': run['means']['Dice']}
    R['mascaras_total'] = n_masc
    R['mascaras_disc_max_imagen'] = dmax_img
    R['mascaras_disc_max_grupo'] = dmax_grp

    # 2. Tabla de test desde los 50 JSON (por imagen -> grupo -> media de grupos -> semillas).
    G = sorted(set(grupo.values()))
    media_grupo = {}
    dmax_agr = 0.0
    for m in modelos:
        filas_m = []
        for s in semillas:
            run = corridas[f'{m}_seed{s}']
            g = defaultdict(list)
            for d in run['details']:
                g[d['group_id']].append(d['Dice'])
            gm = {k: float(np.mean(v)) for k, v in g.items()}
            dmax_agr = max(dmax_agr, max(abs(gm[k] - run['groups'][k]['Dice']) for k in gm))
            filas_m.append([gm[k] for k in G])
        a = np.array(filas_m)
        media_grupo[m] = a.mean(axis=0)
        R['tabla'][m] = {'Dice_test': float(a.mean()), 'DE_semillas': float(a.mean(axis=1).std(ddof=1))}
    R['agregacion_imagen_a_grupo_disc_max'] = dmax_agr
    for m in modelos:
        if m != 'fcn':
            R['contrastes_vs_fcn'][m] = float((media_grupo[m] - media_grupo['fcn']).mean())

    # 3. Contra lo publicado.
    resumen = leer(test / 'RESUMEN_TEST.json')
    auditoria = json.loads(publicado('resultados/auditoria_test_e2/AUDITORIA.json'))  # versión de GitHub, no la regenerada
    analisis = json.loads(publicado('resultados/integracion_e2/analisis.json'))
    disc = {
        'tabla_vs_RESUMEN_TEST_media': max(abs(R['tabla'][r['model']]['Dice_test'] - r['Dice_mean']) for r in resumen['summary']),
        'tabla_vs_RESUMEN_TEST_DE': max(abs(R['tabla'][r['model']]['DE_semillas'] - r['Dice_sd_seeds']) for r in resumen['summary']),
        'tabla_vs_AUDITORIA_publicada_media': max(abs(R['tabla'][r['model']]['Dice_test'] - r['Dice_test']) for r in auditoria['table']),
        'tabla_vs_AUDITORIA_publicada_DE': max(abs(R['tabla'][r['model']]['DE_semillas'] - r['Dice_sd_seeds']) for r in auditoria['table']),
        'contrastes_vs_RESUMEN_TEST': max(abs(R['contrastes_vs_fcn'][c['model']] - c['Dice_difference']) for c in resumen['contrasts']),
        'contrastes_vs_analisis_json': max(abs(R['contrastes_vs_fcn'][c['model']] - c['difference']) for c in analisis['contrasts']),
    }
    # INFORME_TEST.md redondea a 4 decimales: se compara la cifra impresa.
    informe = (test / 'INFORME_TEST.md').read_text(encoding='utf-8')
    impresas = {m: (float(a), float(b)) for m, a, b in re.findall(r'^\| (\w+) \| ([0-9.]+) ± ([0-9.]+) \|', informe, re.M)}
    disc['tabla_vs_INFORME_TEST_4_decimales'] = {
        'cifras': 2 * len(impresas),
        'distintas_al_redondear': sum(f'{R["tabla"][m]["Dice_test"]:.4f}' != f'{a:.4f}' or f'{R["tabla"][m]["DE_semillas"]:.4f}' != f'{b:.4f}'
                                      for m, (a, b) in impresas.items())}
    R['discrepancias_vs_publicado'] = disc

    # 4. Reinferencia en GPU (etapa `clean` de diagnosticos_gpu_e2.py: otra sesión de Colab, mismos 50 checkpoints):
    #    Dice medio de test por red contra el test cerrado. Es la única repetición de la inferencia de que se dispone.
    try:
        diag = json.loads(publicado('resultados/diagnosticos_gpu_e2/RESUMEN_DIAGNOSTICOS.json'))
        cerr = {r['model']: r['Dice_mean'] for r in resumen['summary']}
        dif = {x['model']: x['test']['Dice']['mean'] - cerr[x['model']] for x in diag['clean']}
        R['reinferencia_gpu'] = {'por_red': dif, 'disc_max': max(abs(v) for v in dif.values()), 'redes': len(dif),
                                 'redes_identicas': sum(v == 0 for v in dif.values())}
    except Exception as e:
        R['reinferencia_gpu'] = {'error': repr(e)}
    return R


# ------------------------------------------- análisis sin historias (se llama a sí mismo en el intérprete limpio)
def _analisis_sin_historias(copia):
    copia = Path(copia)
    sys.path.insert(0, str(copia / 'scripts'))
    os.chdir(copia)
    import analisis_integracion_e2 as A
    original = A.read

    def leer(path):
        path = Path(path)
        if HIST in path.parts and not path.exists():
            return []  # curva vacía; la figura se restaura desde git y se reporta como no reproducida
        return original(path)
    A.read = leer
    A.main()


# ------------------------------------------------------------------ cuadernos
def clasificar_cuaderno(path):
    nb = json.loads(path.read_text(encoding='utf-8'))
    codigo = '\n'.join(''.join(c.get('source', '')) for c in nb.get('cells', []) if c.get('cell_type') == 'code')
    motivos = [m for m, pat in [('Google Drive', r'/content/drive|drive\.mount'), ('Colab', r'google\.colab|/content/'),
                                ('GPU (cuda)', r'cuda'), ('instala paquetes', r'pip install')] if re.search(pat, codigo)]
    return {'celdas_codigo': sum(c.get('cell_type') == 'code' for c in nb.get('cells', [])),
            'celdas_con_salida_guardada': sum(bool(c.get('outputs')) for c in nb.get('cells', []) if c.get('cell_type') == 'code'),
            'requiere': motivos}


def _ejecutar_cuaderno(nb_path, salida):
    """Corre en el intérprete del libro: ejecuta con nbclient, guarda el cuaderno y compara salidas de texto."""
    import nbformat
    from nbclient import NotebookClient
    nb_path, salida = Path(nb_path), Path(salida)
    nb = nbformat.read(nb_path, as_version=4)
    guardadas = [c.get('outputs', []) for c in nb.cells if c.cell_type == 'code']
    for c in nb.cells:  # se vacían para que ninguna salida guardada pase por ejecutada
        if c.cell_type == 'code':
            c.outputs, c.execution_count = [], None
    texto = lambda outs: [''.join(o.get('text', '')) if o.get('output_type') == 'stream' else
                          ''.join(o.get('data', {}).get('text/plain', '')) for o in outs
                          if o.get('output_type') in ('stream', 'execute_result')]
    error = None
    try:
        NotebookClient(nb, timeout=1200, kernel_name='python3', resources={'metadata': {'path': str(nb_path.parent)}}).execute()
    except Exception as e:
        error = repr(e)[:600]
    nbformat.write(nb, salida)
    nuevas = [c.get('outputs', []) for c in nb.cells if c.cell_type == 'code']
    comparables = [(i, texto(g), texto(n)) for i, (g, n) in enumerate(zip(guardadas, nuevas)) if g]
    print(json.dumps({'error': error, 'celdas_ejecutadas': sum(c.get('execution_count') is not None for c in nb.cells if c.cell_type == 'code'),
                      'celdas_codigo': len(nuevas), 'celdas_con_error': sum(any(o.get('output_type') == 'error' for o in n) for n in nuevas),
                      'imagenes_producidas': sum(sum('image/png' in o.get('data', {}) for o in n) for n in nuevas),
                      'celdas_con_salida_guardada': len(comparables),
                      'celdas_texto_igual_a_lo_guardado': sum(g == n for _, g, n in comparables),
                      'celdas_texto_distinto': [i for i, g, n in comparables if g != n]}))


def _ejecutar_celdas(nb_path, salida):
    """Sin kernel de Jupyter: ejecuta las celdas de código en orden, en un mismo espacio de nombres, con el intérprete
    limpio; captura la salida de texto de cada celda y la compara con la guardada. `display` cuenta las figuras."""
    import contextlib
    import io
    import traceback
    nb_path = Path(nb_path)
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    os.chdir(nb_path.parent)
    figuras = []
    ns = {'__name__': '__main__', 'display': lambda obj, *x, **k: figuras.append(type(obj).__name__)}
    celdas = []
    for c in nb['cells']:
        if c.get('cell_type') != 'code':
            continue
        guardado = ''.join(''.join(o.get('text', '')) for o in c.get('outputs', []) if o.get('output_type') == 'stream')
        buf, error, antes = io.StringIO(), None, len(figuras)
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(''.join(c.get('source', '')), f'<celda {len(celdas)}>', 'exec'), ns)
        except Exception:
            error = traceback.format_exc().strip().splitlines()[-1][:300]
        celdas.append({'error': error, 'figuras': len(figuras) - antes, 'texto_igual': buf.getvalue() == guardado,
                       'con_salida_guardada': bool(c.get('outputs')),
                       'lineas_distintas': [x for x, y in zip(buf.getvalue().splitlines(), guardado.splitlines()) if x != y][:3]})
        if error:
            break
    Path(salida).write_text(json.dumps(celdas, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'error': next((c['error'] for c in celdas if c['error']), None), 'celdas_codigo': sum(c.get('cell_type') == 'code' for c in nb['cells']),
                      'celdas_ejecutadas_sin_error': sum(c['error'] is None for c in celdas), 'figuras_mostradas': len(figuras),
                      'celdas_con_salida_guardada': sum(c['con_salida_guardada'] for c in celdas),
                      'celdas_texto_igual_a_lo_guardado': sum(c['texto_igual'] and c['con_salida_guardada'] for c in celdas),
                      'lineas_distintas': [l for c in celdas for l in c['lineas_distintas']]}, ensure_ascii=False))


def ejecutar_cuadernos(copia, jb, env, salida_nb, py_datos=None, env_datos=None):
    """Clasifica cada cuaderno por su código; ejecuta los que no piden GPU, Colab ni Drive con el intérprete del libro.

    Los cuadernos de EDA leen resultados/inicio_voc/ y resultados/revision_voc/, que no se versionan: antes se generan
    con scripts/inicio_voc.py y scripts/revisar_voc.py.
    """
    py_libro = jb.parent.parent / ('python.exe' if sys.platform == 'win32' else 'bin/python')
    salida_nb.mkdir(parents=True, exist_ok=True)
    cuadernos = {}
    for script, marca in [('inicio_voc', 'resultados/inicio_voc/resumen.json'), ('revisar_voc', 'resultados/revision_voc/resumen.json')]:
        if not (copia / marca).exists():
            cuadernos[f'_previo_{script}'] = limpio(correr([py_datos or py_libro, *(['-E', '-s'] if py_datos else []),
                                                            f'scripts/{script}.py'], copia, env_datos or env))
    for nb in sorted(copia.rglob('*.ipynb')):
        if '_build' in nb.parts:
            continue
        rel = nb.relative_to(copia).as_posix()
        info = clasificar_cuaderno(nb)
        info['ejecutado'] = not info['requiere']  # la clasificación lee el código; no se ejecuta lo que pide Drive o GPU
        if info['ejecutado']:
            if py_datos:  # celdas en orden con el intérprete limpio (el kernel del libro trae otro Pillow)
                rr = correr([py_datos, '-E', '-s', str(Path(__file__).resolve()), '--_ejecutar-celdas', nb,
                             salida_nb / (rel.replace('/', '__') + '.json')], nb.parent, env_datos)
            else:
                rr = correr([py_libro, str(Path(__file__).resolve()), '--_ejecutar-cuaderno', nb,
                             salida_nb / (rel.replace('/', '__'))], nb.parent, env)
            info['ejecucion'] = limpio(rr)
            try:
                info['resultado'] = json.loads(rr['_stdout'].strip().splitlines()[-1])
            except Exception:
                info['resultado'] = None
        cuadernos[rel] = info
    return cuadernos


# ----------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--destino', type=Path, required=True)
    ap.add_argument('--voc', type=Path, required=True)
    ap.add_argument('--test', type=Path, required=True, help='carpeta resultados_test_e2_final')
    ap.add_argument('--historias', type=Path, help='carpeta tesis_imo_e2_arquitecturas_reales_v2 (opcional)')
    ap.add_argument('--python', type=Path, required=True)
    ap.add_argument('--python-alt', type=Path, help='intérprete para los tests que piden paquetes que el principal no tiene')
    ap.add_argument('--jupyter-book', type=Path)
    ap.add_argument('--url', default=URL)
    ap.add_argument('--rama', default='cierre-e2')
    ap.add_argument('--sin-locales', action='store_true', help='salta completar_analisis_locales_e2.py (unos 6 minutos)')
    a = ap.parse_args(argv)
    for nombre, ruta, marca in [('--voc', a.voc, 'labelmap.txt'), ('--test', a.test, 'RESUMEN_TEST.json')]:
        if not (ruta / marca).exists():
            raise SystemExit(f'{nombre} {ruta}: falta {marca}')
    if a.destino.exists():
        raise SystemExit(f'{a.destino} ya existe; usar una carpeta nueva')
    a.destino.mkdir(parents=True)
    t0 = time.time()

    # 1. Clon limpio.
    repo = a.destino / 'repo'
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', a.rama, '--single-branch', a.url, str(repo)],
                   check=True, capture_output=True)
    git = lambda *x: subprocess.run(['git', *x], cwd=repo, capture_output=True, check=True).stdout
    commit = git('rev-parse', 'HEAD').decode().strip()
    copia = repo / REL
    e2 = copia.parent

    # 2. Datos externos como hermanos.
    enlaces = {'VOC2012': a.voc, 'resultados_test_e2_final': a.test}
    if a.historias:
        enlaces[HIST] = a.historias
    for nombre, destino in enlaces.items():
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(e2 / nombre), str(destino.resolve())], check=True, capture_output=True)
    archivos_test = sorted(p for p in a.test.rglob('*') if p.is_file())
    hashes = {p.relative_to(a.test).as_posix(): sha256(p) for p in archivos_test}
    huella = hashlib.sha256('\n'.join(f'{k} {v}' for k, v in sorted(hashes.items())).encode()).hexdigest()

    env = {k: os.environ[k] for k in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'USERPROFILE', 'COMSPEC', 'APPDATA', 'LOCALAPPDATA') if k in os.environ}
    ruta = lambda py: os.pathsep.join([str(py.parent), str(py.parent / 'Library/bin'), str(py.parent / 'Scripts'),
                                        os.path.join(os.environ.get('SYSTEMROOT', r'C:\Windows'), 'System32')])
    env.update({'PATH': ruta(a.python), 'PYTHONHASHSEED': '0', 'PYTHONIOENCODING': 'utf-8', 'MPLBACKEND': 'Agg'})
    py = [str(a.python), '-E', '-s']
    vers = lambda p, e: subprocess.run([str(p), '-E', '-s', '-c', 'import sys,numpy,scipy,PIL,matplotlib;'
                                        'print(sys.version.split()[0],numpy.__version__,scipy.__version__,PIL.__version__,matplotlib.__version__)'],
                                       env=e, capture_output=True, text=True).stdout.split()
    v = vers(a.python, env)
    R = {'estado': 'Clon limpio desde GitHub; datos externos como hermanos; intérprete con -E -s.',
         'url': a.url, 'rama': a.rama, 'commit': commit, 'destino': str(a.destino),
         'entorno': dict(zip(['python', 'numpy', 'scipy', 'pillow', 'matplotlib'], v), plataforma=platform.platform()),
         'datos_externos': {'VOC2012': str(a.voc.resolve()), 'resultados_test_e2_final': str(a.test.resolve()),
                            'archivos_test': len(hashes), 'mascaras_seed11': sum(k.startswith('mascaras_seed11/') for k in hashes),
                            'huella_sha256_test': huella, 'historias': str(a.historias) if a.historias else None},
         'pasos': {}}
    (a.destino / 'hashes_test.json').write_text(json.dumps(hashes, indent=0), encoding='utf-8')

    # 3. Máscaras binarias.
    R['pasos']['mascaras_binarias'] = limpio(correr(py + ['scripts/reconstruir_mascaras_binarias_e2.py', '--voc', e2 / 'VOC2012'], copia, env))

    # 4. Cadena del test.
    pasos = [('auditar_test', ['scripts/auditar_test_e2.py']),
             ('graficar_test', ['scripts/graficar_test_e2.py'])]
    if a.historias:
        pasos.append(('analisis_integracion', ['scripts/analisis_integracion_e2.py']))
    else:
        pasos.append(('analisis_integracion', [str(Path(__file__).resolve()), '--_analisis-sin-historias', str(copia)]))
    pasos.append(('integrar_entregable2', ['scripts/integrar_entregable2.py']))
    if not a.sin_locales:
        pasos.append(('completar_analisis_locales', ['scripts/completar_analisis_locales_e2.py']))
    for nombre, cmd in pasos:
        r = correr(py + cmd, copia, env)
        R['pasos'][nombre] = limpio(r)
        print(nombre, r['codigo_salida'], r['segundos'], flush=True)
    if not a.historias:
        for c in CURVAS:
            git('checkout', '--', f'{REL}/{c}')
        R['no_reproducido'] = {c: 'faltan las historias de entrenamiento (history.json de las 50 corridas, solo en Drive)' for c in CURVAS}

    # Tests: con el intérprete principal y, si falla un archivo, con el alternativo.
    R['pasos']['tests'] = {}
    for t in sorted((copia / 'tests').glob('test_*.py')):
        r = limpio(correr(py + ['-m', 'unittest', 'discover', '-s', 'tests', '-p', t.name], copia, env))
        if r['codigo_salida'] != 0 and a.python_alt:
            env2 = {**env, 'PATH': ruta(a.python_alt)}
            r2 = limpio(correr([str(a.python_alt), '-E', '-s', '-m', 'unittest', 'discover', '-s', 'tests', '-p', t.name], copia, env2))
            r = {'principal': r, 'alternativo': r2, 'codigo_salida': r2['codigo_salida']}
        R['pasos']['tests'][t.name] = r
    if a.python_alt:
        env2 = {**env, 'PATH': ruta(a.python_alt)}
        R['entorno_alternativo'] = dict(zip(['python', 'numpy', 'scipy', 'pillow', 'matplotlib'], vers(a.python_alt, env2)))

    # 5. Comparación de lo regenerado con lo publicado.
    estado = git('status', '--porcelain', '--untracked-files=all', '--', REL).decode('utf-8', 'replace').splitlines()
    comp = {}
    for linea in estado:
        marca, ruta_rel = linea[:2], linea[3:].strip().strip('"')
        rel = ruta_rel[len(REL) + 1:] if ruta_rel.startswith(REL + '/') else ruta_rel
        nuevo = copia / rel
        if marca.strip() == '??':
            comp[rel] = {'tipo': 'nuevo', 'nota': 'sin versión publicada'}
            continue
        if not nuevo.exists():
            comp[rel] = {'tipo': 'borrado'}
            continue
        pub = git('show', f'HEAD:{ruta_rel}')
        suf = nuevo.suffix.lower()
        try:
            if suf == '.json':
                comp[rel] = comparar_json(json.loads(nuevo.read_bytes()), json.loads(pub))
            elif suf == '.png':
                comp[rel] = comparar_png(nuevo.read_bytes(), pub)
            elif suf in ('.md', '.txt', '.csv', '.yml', '.inc', '.bib'):
                comp[rel] = comparar_texto(nuevo.read_text(encoding='utf-8'), pub.decode('utf-8'))
            else:
                comp[rel] = {'tipo': suf, 'bytes_iguales': nuevo.read_bytes() == pub}
        except Exception as e:  # se reporta, no se esconde
            comp[rel] = {'tipo': 'error', 'error': repr(e)}
    R['archivos_modificados_respecto_a_lo_publicado'] = comp
    R['disc_max_json'] = max((c['discrepancia_max'] for c in comp.values() if c.get('tipo') == 'json'), default=0.0)

    # 6. Recálculo independiente.
    try:
        R['recalculo_independiente'] = recalculo_independiente(copia, a.test, lambda rel: git('show', f'HEAD:{REL}/{rel}'))
    except Exception as e:
        R['recalculo_independiente'] = {'error': repr(e)}

    # 7. Libro y cuadernos.
    if a.jupyter_book:
        jb = a.jupyter_book
        envjb = {**env, 'PATH': os.pathsep.join([str(jb.parent), str(jb.parent.parent), str(jb.parent.parent / 'Library/bin'), env['PATH']])}
        r = correr([jb, 'build', 'libro', '-W', '--keep-going'], copia, envjb)
        avisos = [x for x in (r['_stdout'] + r['_stderr']).splitlines() if 'WARNING' in x or 'ERROR' in x]
        r = limpio(r)
        r['avisos'] = len(avisos)
        r['avisos_ejemplos'] = avisos[:10]
        r['html'] = (copia / 'libro/_build/html/index.html').exists()
        r['paginas_html'] = len(list((copia / 'libro/_build/html').glob('*.html')))
        R['pasos']['libro'] = r
        R['pasos']['cuadernos'] = ejecutar_cuadernos(copia, jb, envjb, a.destino / 'cuadernos_ejecutados', py[0], env)

    R['segundos_total'] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'REPORTE.json').write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    ri = R['recalculo_independiente']
    print(json.dumps({'commit': commit, 'entorno': R['entorno'], 'disc_max_json': R['disc_max_json'],
                      'mascaras': ri.get('mascaras_disc_max_imagen'), 'grupos': ri.get('mascaras_disc_max_grupo'),
                      'vs_publicado': ri.get('discrepancias_vs_publicado'), 'reinferencia_gpu': ri.get('reinferencia_gpu', {}).get('disc_max'),
                      'libro': R['pasos'].get('libro', {}).get('codigo_salida')}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--_analisis-sin-historias':
        _analisis_sin_historias(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == '--_ejecutar-celdas':
        _ejecutar_celdas(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == '--_ejecutar-cuaderno':
        _ejecutar_cuaderno(sys.argv[2], sys.argv[3])
    elif len(sys.argv) in (4, 5) and sys.argv[1] == '--solo-cuadernos':
        # python scripts/reproduccion_completa_e2.py --solo-cuadernos CARPETA_DEL_CLON RUTA/jupyter-book [PYTHON_DATOS]
        destino, jb = Path(sys.argv[2]), Path(sys.argv[3])
        copia = destino / 'repo' / REL
        env = {k: os.environ[k] for k in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'USERPROFILE', 'COMSPEC', 'APPDATA', 'LOCALAPPDATA') if k in os.environ}
        env.update({'PATH': os.pathsep.join([str(jb.parent), str(jb.parent.parent), str(jb.parent.parent / 'Library/bin'),
                                             os.path.join(os.environ.get('SYSTEMROOT', r'C:\Windows'), 'System32')]),
                    'PYTHONIOENCODING': 'utf-8', 'MPLBACKEND': 'Agg'})
        R = json.loads((OUT / 'REPORTE.json').read_text(encoding='utf-8'))
        pyd = sys.argv[4] if len(sys.argv) == 5 else None
        envd = {**env, 'PATH': os.pathsep.join([str(Path(pyd).parent), str(Path(pyd).parent / 'Library/bin'), env['PATH']]),
                'PYTHONHASHSEED': '0'} if pyd else None
        R['pasos']['cuadernos'] = ejecutar_cuadernos(copia, jb, env, destino / 'cuadernos_ejecutados', pyd, envd)
        (OUT / 'REPORTE.json').write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
        print(json.dumps({k: v.get('resultado', v.get('codigo_salida')) for k, v in R['pasos']['cuadernos'].items()
                          if v.get('ejecutado', True)}, indent=1, ensure_ascii=False))
    else:
        main()
