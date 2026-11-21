"""Compila el libro desde cero y lo deja en <repo>/docs, la carpeta que GitHub Pages publica desde main.

  python scripts/publicar_libro_docs.py --jupyter-book <ruta a jupyter-book>

Borra libro/_build, compila con --keep-going y se detiene si hay advertencias (una figura faltante no debe llegar al
sitio). Luego reemplaza el contenido de docs/ por libro/_build/html y conserva docs/.nojekyll. No hace commit ni push:
eso queda en manos de quien publica, con el mensaje que cuente qué cambió.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

PROYECTO = Path(__file__).resolve().parents[1]
REPO = PROYECTO.parents[1]
DOCS = REPO / 'docs'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--jupyter-book', default='jupyter-book')
    a = ap.parse_args()
    shutil.rmtree(PROYECTO / 'libro' / '_build', ignore_errors=True)
    r = subprocess.run([a.jupyter_book, 'build', str(PROYECTO / 'libro'), '--keep-going'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    salida = r.stdout + r.stderr
    avisos = [l for l in salida.splitlines() if 'WARNING' in l]
    if r.returncode or avisos or 'build succeeded' not in salida:
        print('\n'.join(avisos[:40]) or salida[-3000:])
        raise SystemExit('La compilación no quedó limpia: no se toca docs/')
    html = PROYECTO / 'libro' / '_build' / 'html'
    for p in DOCS.iterdir():
        if p.name == '.nojekyll':
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    shutil.copytree(html, DOCS, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.buildinfo', '.doctrees'))
    (DOCS / '.nojekyll').touch()
    print(f'docs/ actualizado desde {html}: {sum(1 for _ in DOCS.rglob("*") if _.is_file())} archivos')


if __name__ == '__main__':
    main()
