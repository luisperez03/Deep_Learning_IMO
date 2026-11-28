"""Empaqueta la secuencia verificada en un .py y un notebook autocontenidos."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
TRAIN=ROOT/'e3/training'


def main():
    notebooks={p.name:json.loads(p.read_text(encoding='utf-8'))
               for p in sorted((TRAIN/'colab/secuencia_v4').glob('*.ipynb'))}
    assert len(notebooks)==10
    orchestrator=(TRAIN/'orquestador_e3.py').read_text(encoding='utf-8')
    source=orchestrator+'\nNOTEBOOKS = '+repr(notebooks)+"\nif __name__ == '__main__':\n    run(NOTEBOOKS, os.environ.get('E3_CARRIL', 'todo'))\n"
    ast.parse(source)
    (ROOT/'E3_CONTINUAR_EN_ESTA_MAQUINA.py').write_text(source,encoding='utf-8')
    def cell(kind,s):
        c=dict(cell_type=kind,metadata={},source=s.splitlines(True))
        if kind=='code':c.update(execution_count=None,outputs=[])
        return c
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
            cells=[cell('markdown','# E3: toda la secuencia en una máquina\n\nSelecciona GPU y ejecuta todo. '
                       'No cambies de notebook. Guarda checkpoints en Drive y reanuda al repetir. '
                       'Entradas: paquete_colab_e2_v2.zip, compactas_a0498_e3.zip y la carpeta histórica del benchmark '
                       'en MyDrive. No evalúa test. Si ya estás entrenando en otra celda, espera que termine antes de iniciar.\n'),
                   cell('code',"import os\nos.environ['E3_CARRIL'] = 'todo'\n# Una máquina: todo. Dos: original_y_compactas y rivales. Al acabar ambas: cierre.\n# Sin entrenar el original: sin_original (rivales y recepción, sin test).\n# También: original, compactas, rivales_a, rivales_b, rivales_c\n"),
                   cell('code',source)])
    (ROOT/'E3_TODO_EN_UNA_MAQUINA.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    print('Lanzador y notebook creados; SHA256:',hashlib.sha256(source.encode()).hexdigest())


if __name__=='__main__':main()
