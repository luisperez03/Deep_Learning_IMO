"""No admitir campañas incompletas ni checkpoints alterados como listos para revisión."""
import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

E3 = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E3/'experiments'))
import auditar_campana as audit


@pytest.mark.parametrize('alteracion', [None, 'sha', 'config', 'arquitectura'])
def test_auditor_detecta_incompletos_y_alterados(tmp_path, monkeypatch, alteracion):
    run=tmp_path/'compactas/fase2/fcn_a0498_s11';run.mkdir(parents=True)
    (run/'best.pt').write_bytes(b'contenido de prueba para verificar SHA')
    result=dict(completed=True, fingerprint='fp', familia='fcn', semilla=11, arq='a0498',
                best_file='best.pt',best_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
                best_val_group_Dice=.8)
    config=dict(fingerprint='fp',tarea=dict(w=[64,64,64],b=[0,1,0]))
    if alteracion=='sha': (run/'best.pt').write_bytes(b'alterado')
    if alteracion=='config': config['fingerprint']='otro'
    if alteracion=='arquitectura': config['tarea']['w']=[16,32,64]
    (run/'config.json').write_text(json.dumps(config))
    (run/'result.json').write_text(json.dumps(result))
    monkeypatch.setattr(sys,'argv',['audit','--campana',str(tmp_path/'campana'),
                                  '--benchmark',str(tmp_path/'benchmark'),'--compactas',str(tmp_path/'compactas')])
    assert audit.main()==2
    report=json.loads((tmp_path/'campana/estado_campana.json').read_text())
    assert report['esperadas']==120 and report['verificadas']==(1 if alteracion is None else 0)
    assert not report['lista_para_revision_pretest']


def test_cuadernos_fuentes_y_orden():
    folder=E3/'training/colab/secuencia_v4'
    notebooks=sorted(folder.glob('*.ipynb'))
    assert len(notebooks)==10
    for p in notebooks:
        nb=json.loads(p.read_text(encoding='utf-8'))
        for c in nb['cells']:
            if c['cell_type']!='code':continue
            assert c['execution_count'] is None and not c['outputs']
            tree=ast.parse(''.join(c['source']))
            for node in tree.body:
                if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='CONTENIDO':
                    for rel,code in ast.literal_eval(node.value).items():
                        assert code==(E3/rel).read_text(encoding='utf-8')
                        if rel.endswith('.py'): ast.parse(code)
    first=json.loads(notebooks[0].read_text(encoding='utf-8'))
    code='\n'.join(''.join(c['source']) for c in first['cells'] if c['cell_type']=='code')
    assert code.index("'-m', 'pytest'") < code.index('entrenar([11])')
    assert 'opencv-python-headless==4.11.0.86' in code
    assert '--pause-after-epoch' not in code.split('env = dict(os.environ')[-1]
