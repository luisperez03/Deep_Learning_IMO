import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import threading
import types
import zipfile
from pathlib import Path

import pytest

E3=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('orq_fix',E3/'training/orquestador_e3.py')
O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)


@pytest.fixture
def project(tmp_path):
    root=tmp_path/'base';p=root/'project';p.mkdir(parents=True)
    files={'scripts/a.py':b'A=1\n','third_party/e2/net.py':b'NET=1\n',
           'third_party/e2/sources.lock.json':b'{}','requirements-e2.txt':b'numpy==1.26.4\n'}
    archive=tmp_path/'base.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for rel,data in files.items():
            q=p/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
            z.writestr('project/'+rel,data)
    extra={'scripts/busqueda_arquitectura_e2.py':'B=2\n','scripts/complemento_entreno_e2.py':'C=3\n'}
    notebooks={'05_compactas':{'cells':[{'cell_type':'code','source':['FUENTES_E2 = '+repr(extra)+"\nraise RuntimeError('NO ENTRENAR')"]}]}}
    scope={'PROJECT':p,'OUTPUT':tmp_path/'salida','ZIP':archive,'BASE':root}
    return notebooks,scope,extra


def test_05_omitida_restaura_06_antes_de_entrenar(project,tmp_path,monkeypatch):
    notebooks,scope,extra=project
    signature=O.prepare_sources(notebooks,scope,'06')['source']
    run=scope['OUTPUT']/'modelo_seed89';run.mkdir(parents=True)
    record={'fingerprint':'historico-intocable','identity':{'source':signature}}
    (run/'config.json').write_text(json.dumps(record));checkpoint=run/'last.pt';checkpoint.write_bytes(b'intocable')
    for rel in extra:(scope['PROJECT']/rel).unlink()  # máquina nueva, sin efectos de 05
    assert O.source_digest(scope['PROJECT'])!=signature
    config_before=(run/'config.json').read_bytes()
    setup='from pathlib import Path\n'+'\n'.join(f'{k}=Path({str(v)!r})' for k,v in scope.items())
    verify=f"assert (PROJECT/'scripts/busqueda_arquitectura_e2.py').read_bytes()=={extra['scripts/busqueda_arquitectura_e2.py'].encode()!r}\nprint('entrenamiento habilitado')"
    notebooks['06_rivales']={'cells':[{'cell_type':'code','source':[setup]},{'cell_type':'code','source':[verify]}]}
    colab=types.ModuleType('google.colab');colab.drive=types.SimpleNamespace(mount=lambda *_:None)
    colab.files=types.SimpleNamespace(download=lambda *_:None)
    monkeypatch.setitem(sys.modules,'google.colab',colab);monkeypatch.setattr(O,'ROOT',tmp_path/'drive')
    monkeypatch.setitem(O.LANES,'regresion',['05','06']);monkeypatch.setattr(O,'completed',lambda s:s=='05')
    monkeypatch.setattr(O,'targets',lambda _:[])
    O.run(notebooks,'regresion')
    assert O.source_digest(scope['PROJECT'])==signature
    assert (run/'config.json').read_bytes()==config_before and checkpoint.read_bytes()==b'intocable'
    assert 'entrenamiento habilitado' in next((O.ROOT/'logs').rglob('06.stdout.log')).read_text()


def test_recupera_perfil_base_historico_y_rechaza_desconocido(project):
    notebooks,scope,extra=project;base=O.source_digest(scope['PROJECT'])
    O.prepare_sources(notebooks,scope,'06')
    scope['OUTPUT'].mkdir();preflight=scope['OUTPUT']/'preflight.json'
    preflight.write_text(json.dumps({'identity':{'source':base}}))
    assert O.prepare_sources(notebooks,scope,'08')['source']==base
    assert not (scope['PROJECT']/next(iter(extra))).exists()
    preflight.write_text(json.dumps({'identity':{'source':'firma-no-recuperable'}}));before=preflight.read_bytes()
    with pytest.raises(RuntimeError,match='No se pudo recuperar'):O.prepare_sources(notebooks,scope,'06')
    assert preflight.read_bytes()==before


def test_transformaciones_no_cambian_payloads_historicos():
    for path in (E3/'training/colab/secuencia_v4').glob('*.ipynb'):
        nb=json.loads(path.read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type']!='code':continue
            original=''.join(cell['source']);changed=O.transformed(original,path.name[:2])
            def payloads(code):
                return {node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(code).body
                        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                        and node.targets[0].id in ['CONTENIDO','FUENTES_E2','FIRMAS','CONFIG_E3']}
            assert payloads(original)==payloads(changed)
            if path.name.startswith('06') and 'def pip_install' in original:
                assert 'setuptools==84.2.1' in changed


def test_preflight_regenerado_no_invalida_config_checkpoint(project):
    notebooks,scope,extra=project;base=O.source_digest(scope['PROJECT'])
    wanted=O.prepare_sources(notebooks,scope,'06')['source']
    run=scope['OUTPUT']/'transunet_seed89';run.mkdir(parents=True)
    cfg=run/'config.json';cfg.write_text(json.dumps({'fingerprint':'no-alterar','identity':{'source':wanted}}))
    checkpoint=run/'last.pt';checkpoint.write_bytes(b'checkpoint-original')
    preflight=scope['OUTPUT']/'preflight.json'
    preflight.write_text(json.dumps({'identity':{'source':base}}));before=preflight.read_bytes();cfg_before=cfg.read_bytes()
    for rel in extra:(scope['PROJECT']/rel).unlink()
    assert O.prepare_sources(notebooks,scope,'06')['source']==wanted
    assert cfg.read_bytes()==cfg_before and checkpoint.read_bytes()==b'checkpoint-original'
    assert preflight.read_bytes()==before
    assert next((scope['OUTPUT']/'logs').glob('preflight_anterior_*.json')).read_bytes()==before


def test_conflicto_real_entre_corridas_no_se_oculta(project):
    notebooks,scope,extra=project
    before=O.source_digest(scope['PROJECT'])
    for i,value in enumerate(['firma-1','firma-2']):
        folder=scope['OUTPUT']/f'red_seed{i}';folder.mkdir(parents=True)
        (folder/'config.json').write_text(json.dumps({'identity':{'source':value}}))
    with pytest.raises(RuntimeError,match='config.json de corridas tienen firmas distintas'):
        O.prepare_sources(notebooks,scope,'06')
    assert O.source_digest(scope['PROJECT'])==before


def test_stdout_stderr_en_vivo_y_error_persistente(tmp_path):
    received=threading.Event();release=tmp_path/'continuar'
    class Watch(io.StringIO):
        def write(self,s):
            value=super().write(s)
            if 'ANTES' in self.getvalue():received.set()
            return value
    out=Watch();err=io.StringIO();results=[]
    code="import sys,time,pathlib;print('ANTES',flush=True);print('ERROR-VIVO',file=sys.stderr,flush=True);p=pathlib.Path(sys.argv[1]);\nwhile not p.exists(): time.sleep(.01)\nsys.exit(7)"
    def runner():
        with (tmp_path/'stdout.log').open('w') as log1,(tmp_path/'stderr.log').open('w') as log2:
            with contextlib.redirect_stdout(O.Tee(out,log1)),contextlib.redirect_stderr(O.Tee(err,log2)):
                try:O.stream_run([sys.executable,'-c',code,str(release)],check=True,capture_output=True,text=True,timeout=15)
                except O.subprocess.CalledProcessError as e:results.append(e)
    thread=threading.Thread(target=runner);thread.start()
    try:
        assert received.wait(10), 'stdout solo apareció al terminar'
        assert thread.is_alive() and 'ANTES' in (tmp_path/'stdout.log').read_text()
    finally:release.touch();thread.join(20)
    assert not thread.is_alive() and results[0].returncode==7
    assert 'ERROR-VIVO' in results[0].stderr and 'ANTES' in results[0].stdout
    assert 'ERROR-VIVO' in (tmp_path/'stderr.log').read_text()
