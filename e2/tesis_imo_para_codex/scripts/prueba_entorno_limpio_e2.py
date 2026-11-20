"""Prueba de reproducción en un entorno limpio (checklist 55 y 68). Exploratoria.

  python scripts/prueba_entorno_limpio_e2.py --destino CARPETA --voc RUTA/VOC2012 --python PYTHON
         [--jupyter-book RUTA/jupyter-book] [--embeddings RUTA/embeddings_resnet50.npz] [--ref HEAD] [--rapido]

1. Copia limpia: `git archive <ref>` del proyecto a CARPETA/e2/tesis_imo_para_codex (solo lo versionado: sin
   cachés, entornos ni datos locales). Los datos van como HERMANOS del proyecto, como piden los CSV
   (`VOC2012/...` relativo a e2/): CARPETA/e2/VOC2012 es un enlace (junction) a --voc, de solo lectura en la práctica.
2. Intérprete aislado: se llama con -E -s (sin PYTHONPATH ni site de usuario) y un entorno de variables mínimo.
3. Pasos, cada uno con su código de salida y las últimas líneas de error:
   a. `scripts/reproducir_entregable2.py` tal cual (la orden de reproducción del proyecto).
   b. Cada archivo de tests por separado.
   c. Reconstrucción de máscaras binarias (verificada por SHA-256).
   d. Re-ejecución desde datos de `completar_analisis_locales_e2.py` (EDA, descriptores, histogramas y adversarial
      train/test con 200 permutaciones); se detiene donde pide máscaras de test de seed11 que el snapshot no trae.
      Se compara cada JSON regenerado con el versionado: discrepancia numérica máxima.
   e. Los complementos de datos nuevos (fuga, shift, ruido, EDA por clase, gamma), comparados igual.
   f. Construcción del libro con jupyter-book (si se da la ruta).
4. Escribe resultados/entorno_limpio_e2/REPORTE.json en el proyecto ORIGINAL (no en la copia).

--rapido salta d (unos 6 minutos de permutaciones).
"""
import argparse
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

P = Path(__file__).resolve().parents[1]
REPO = P.parent.parent
OUT = P / 'resultados/entorno_limpio_e2'
FALTAN_CONOCIDOS = {
    'resultados_test_e2_final/': '50 JSON de test por corrida y mascaras_seed11/ (2.180 PNG); los lee analisis_integracion_e2.py, '
                                 'integrar_entregable2.py, auditar_test_e2.py y completar_analisis_locales_e2.py',
    'tesis_imo_e2_arquitecturas_reales_v2/': '50 carpetas de corrida con history.json y checkpoints; los lee analisis_integracion_e2.py',
    'resultados/preentrenamiento/masks_binary/': '1.451 PNG binarios; reconstruibles con reconstruir_mascaras_binarias_e2.py',
}


def correr(cmd, cwd, env, timeout=3600):
    t = time.time()
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = 'timeout', '', ''
    return {'orden': ' '.join(str(c) for c in cmd[1:]) if len(cmd) > 1 else str(cmd[0]), 'codigo_salida': rc,
            'segundos': round(time.time() - t, 1), 'salida_final': out.strip().splitlines()[-4:],
            'error_final': [x for x in err.strip().splitlines() if x.strip()][-6:]}


def hojas(x, pre=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from hojas(v, f'{pre}/{k}')
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from hojas(v, f'{pre}[{i}]')
    else:
        yield pre, x


def comparar(a, b):
    """Discrepancia numérica máxima y número de hojas no numéricas distintas entre dos JSON."""
    ha, hb = dict(hojas(a)), dict(hojas(b))
    dmax, donde, otras = 0.0, None, 0
    for k in set(ha) | set(hb):
        x, y = ha.get(k), hb.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
            d = abs(float(x) - float(y))
            if d > dmax:
                dmax, donde = d, k
        elif x != y:
            otras += 1
    return {'discrepancia_max': dmax, 'donde': donde, 'hojas': len(ha), 'hojas_no_numericas_distintas': otras}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destino', type=Path, required=True)
    ap.add_argument('--voc', type=Path, required=True)
    ap.add_argument('--python', type=Path, required=True)
    ap.add_argument('--jupyter-book', type=Path)
    ap.add_argument('--embeddings', type=Path)
    ap.add_argument('--ref', default='HEAD')
    ap.add_argument('--rapido', action='store_true')
    a = ap.parse_args(argv)
    a.voc = a.voc.resolve()
    if not (a.voc / 'labelmap.txt').exists() or not (a.voc / 'JPEGImages').is_dir():
        raise SystemExit(f'{a.voc} no parece VOC2012 (falta labelmap.txt o JPEGImages)')
    raiz = a.destino / 'e2'
    if a.destino.exists():  # no se borra nada: la carpeta lleva un enlace a los datos originales
        raise SystemExit(f'{a.destino} ya existe; usar otra carpeta (para borrarla, quitar antes el enlace e2/VOC2012 con rmdir)')
    raiz.mkdir(parents=True)
    rel = P.relative_to(REPO).as_posix()
    tar = subprocess.run(['git', 'archive', '--format=tar', a.ref, rel], cwd=REPO, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(a.destino, filter='data')
    copia = a.destino / rel
    commit = subprocess.run(['git', 'rev-parse', a.ref], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(['cmd', '/c', 'mklink', '/J', str(raiz / 'VOC2012'), str(a.voc)], check=True, capture_output=True)
    if not (raiz / 'VOC2012' / 'labelmap.txt').exists():
        raise SystemExit('El enlace a los datos no resuelve')
    env = {k: os.environ[k] for k in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'USERPROFILE', 'COMSPEC') if k in os.environ}
    env['PATH'] = os.pathsep.join([str(a.python.parent), str(a.python.parent / 'Library/bin'), str(a.python.parent / 'Scripts'),
                                   os.path.join(os.environ.get('SYSTEMROOT', r'C:\Windows'), 'System32')])
    env['PYTHONHASHSEED'] = '0'; env['PYTHONIOENCODING'] = 'utf-8'; env['MPLBACKEND'] = 'Agg'
    py = [str(a.python), '-E', '-s']
    version = subprocess.run(py + ['-c', 'import sys,numpy,scipy,PIL;print(sys.version.split()[0],numpy.__version__,scipy.__version__,PIL.__version__)'],
                             env=env, capture_output=True, text=True).stdout.split()
    R = {'estado': 'Exploratorio. Copia limpia desde git; datos como hermanos; intérprete con -E -s.',
         'ref': a.ref, 'commit': commit, 'copia': str(copia), 'datos_hermanos': str(raiz / 'VOC2012'),
         'entorno': {'python': version[0] if version else None, 'numpy': version[1] if len(version) > 1 else None,
                     'scipy': version[2] if len(version) > 2 else None, 'pillow': version[3] if len(version) > 3 else None,
                     'plataforma': platform.platform()},
         'faltan_en_el_snapshot': {k: {'descripcion': v, 'presente_en_copia': (a.destino / 'e2' / k).exists() if not k.startswith('resultados')
                                       else (copia / k).exists()} for k, v in FALTAN_CONOCIDOS.items()},
         'pasos': {}}

    R['pasos']['a_reproducir_entregable2'] = correr(py + ['scripts/reproducir_entregable2.py'], copia, env)
    R['pasos']['b_tests'] = {t.name: correr(py + ['-m', 'unittest', 'discover', '-s', 'tests', '-p', t.name], copia, env)
                             for t in sorted((copia / 'tests').glob('test_*.py'))}
    R['pasos']['c_mascaras_binarias'] = correr(py + ['scripts/reconstruir_mascaras_binarias_e2.py', '--voc', str(raiz / 'VOC2012')], copia, env)
    R['pasos']['b2_tests_con_mascaras'] = {t.name: correr(py + ['-m', 'unittest', 'discover', '-s', 'tests', '-p', t.name], copia, env)
                                           for t in sorted((copia / 'tests').glob('test_*.py'))}

    def cmp_json(nombres, sub):
        res = {}
        for n in nombres:
            nuevo, viejo = copia / sub / n, P / sub / n
            if not nuevo.exists():
                res[n] = 'no se regeneró'; continue
            res[n] = comparar(json.loads(nuevo.read_text(encoding='utf-8')), json.loads(viejo.read_text(encoding='utf-8')))
        return res

    if not a.rapido:
        R['pasos']['d_completar_analisis_locales'] = correr(py + ['scripts/completar_analisis_locales_e2.py'], copia, env)
        R['pasos']['d_completar_analisis_locales']['comparacion'] = cmp_json(
            ['eda_por_imagen.json', 'descriptores_imagen.json', 'histogramas_rgb_por_clase.json', 'validacion_adversarial.json'],
            'resultados/complementos_locales_e2')

    voc = str(raiz / 'VOC2012')
    nuevos = [('fuga', ['scripts/fuga_embeddings_e2.py', '--voc', voc] + (['--embeddings', str(a.embeddings)] if a.embeddings else ['--embeddings', 'no_existe.npz']),
               'resultados/fuga_embeddings_e2', ['RESUMEN.json', 'similitud_test_train.json', 'grupos_casi_duplicado.json']),
              ('sensibilidad', ['scripts/sensibilidad_casi_duplicados_e2.py', '--fuente', 'bordes_seed11'],
               'resultados/fuga_embeddings_e2', ['sensibilidad_bordes_seed11.json']),
              ('eda_clases', ['scripts/eda_clases_formas_e2.py', '--voc', voc], 'resultados/eda_clases_e2', ['RESUMEN.json']),
              ('gamma', ['scripts/techo_reanotacion_e2.py', 'gamma'], 'resultados/techo_e2', ['gamma.json'])]
    if not a.rapido:
        nuevos.append(('shift', ['scripts/shift_train_val_e2.py'], 'resultados/shift_train_val_e2', ['RESUMEN.json']))
    R['pasos']['e_complementos_datos'] = {}
    for nombre, cmd, sub, archivos in nuevos:
        r = correr(py + cmd, copia, env)
        r['comparacion'] = cmp_json(archivos, sub)
        R['pasos']['e_complementos_datos'][nombre] = r

    if a.jupyter_book:
        r = correr([str(a.jupyter_book), 'build', 'libro', '-W', '--keep-going'], copia, {**env, 'PATH': os.pathsep.join(
            [str(a.jupyter_book.parent), str(a.jupyter_book.parent.parent), str(a.jupyter_book.parent.parent / 'Library/bin'), env['PATH']])})
        r['html'] = (copia / 'libro/_build/html/index.html').exists()
        R['pasos']['f_libro'] = r

    comparaciones = [c for paso in ('d_completar_analisis_locales',) if paso in R['pasos'] for c in R['pasos'][paso]['comparacion'].values()
                     if isinstance(c, dict)]
    comparaciones += [c for r in R['pasos']['e_complementos_datos'].values() for c in r['comparacion'].values() if isinstance(c, dict)]
    R['discrepancia_maxima_global'] = max((c['discrepancia_max'] for c in comparaciones), default=None)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'REPORTE.json').write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'commit': commit, 'entorno': R['entorno'], 'discrepancia_maxima_global': R['discrepancia_maxima_global'],
                      'a': R['pasos']['a_reproducir_entregable2']['codigo_salida']}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
