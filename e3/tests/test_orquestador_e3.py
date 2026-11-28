import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('orquestador',ROOT/'e3/training/orquestador_e3.py')
O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)


def test_carriles_no_se_pisan():
    groups=[set(O.LANES[n]) for n in ['original','compactas','rivales_a','rivales_b','rivales_c']]
    assert sum(len(g) for g in groups)==len(set.union(*groups))==8
    assert set(O.LANES['todo'])==set.union(*groups)|{'09','10'}
    assert not set(O.LANES['original_y_compactas']) & set(O.LANES['rivales'])
    assert set(O.LANES['original_y_compactas']) | set(O.LANES['rivales']) == set.union(*groups)
    assert set(O.LANES['sin_original']) == {'05','06','07','08','09'}


def test_entorno_rivales_aislado_y_descarga_solo_al_final():
    s="ENV=Path('/content/imo-e2-env')\nfiles.download(str(archive))\n"
    assert 'imo-e3-rivales-env' in O.transformed(s,'07')
    assert 'imo-e3-rivales-env' not in O.transformed(s,'01')
    assert 'files.download' not in O.transformed(s,'07')


@pytest.mark.parametrize('fails',[False,True])
def test_ejecucion_secuencial_y_reanudacion(tmp_path,monkeypatch,fails):
    colab=types.ModuleType('google.colab')
    colab.drive=types.SimpleNamespace(mount=lambda *_:None)
    colab.files=types.SimpleNamespace(download=lambda *_:None)
    monkeypatch.setitem(sys.modules,'google.colab',colab)
    monkeypatch.setattr(O,'ROOT',tmp_path)
    monkeypatch.setitem(O.LANES,'prueba',['01','02'])
    monkeypatch.setattr(O,'completed',lambda stage:stage=='01')
    monkeypatch.setattr(O,'targets',lambda _:[])
    path=tmp_path/'ejecutado'
    code="raise RuntimeError('fallo simulado')" if fails else f"from pathlib import Path\nPath({str(path)!r}).write_text('ok')"
    notebooks={'01_x':{'cells':[{'cell_type':'code','source':["raise RuntimeError('no repetir')"]}]},
               '02_x':{'cells':[{'cell_type':'code','source':[code]}]}}
    if fails:
        with pytest.raises(RuntimeError,match='simulado'):O.run(notebooks,'prueba')
    else:
        O.run(notebooks,'prueba');assert path.read_text()=='ok'
    state=json.loads((tmp_path/'progreso_prueba.json').read_text())
    assert state['etapas']['01']['estado']=='completa_previa_verificada'
    assert state['etapas']['02']['estado']==('interrumpida_o_fallida' if fails else 'completa')


def test_lanzador_autocontenido_y_transformaciones():
    launcher=ROOT/'E3_CONTINUAR_EN_ESTA_MAQUINA.py'
    if not launcher.exists():launcher=ROOT/'e3/continuidad/E3_CONTINUAR_EN_ESTA_MAQUINA.py'
    source=launcher.read_text(encoding='utf-8')
    tree=ast.parse(source)
    notebooks=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                   and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='NOTEBOOKS')
    assert len(notebooks)==10
    for name,nb in notebooks.items():
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(O.transformed(''.join(c['source']),name[:2]))
