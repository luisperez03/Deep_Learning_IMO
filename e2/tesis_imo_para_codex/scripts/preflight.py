"""Valida preparación; nunca inicia entrenamiento, incluso si todos los checks pasan."""
import ctypes,importlib.util,json,sys
from pathlib import Path
from inicio_voc import PROJECT,sha
from preparar_base import read_csv

def memory_available():
    if sys.platform!='win32':return None
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),*[(name,ctypes.c_ulonglong) for name in ['total','available','totalpage','availablepage','totalvirtual','availablevirtual','extended']]]
    state=Memory();state.length=ctypes.sizeof(Memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):return None
    return state.available/1024**3

def main():
    out=PROJECT/'resultados/preentrenamiento';frozen=json.loads((out/'freeze_manifest.json').read_text())
    issues=[]
    for filename,digest in frozen['artifacts'].items():
        if sha(out/filename)!=digest:issues.append('Artefacto cambió: '+filename)
    for filename,digest in frozen['inputs'].items():
        if sha(PROJECT/filename)!=digest:issues.append('Entrada cambió: '+filename)
    if sha(PROJECT/'scripts/preparar_base.py')!=frozen['script_sha256']:issues.append('Código de preparación cambió')
    if sha(PROJECT/'scripts/inicio_voc.py')!=frozen['helper_sha256']:issues.append('Helper de preparación cambió')
    manifest=read_csv(out/'manifest.csv');groups={}
    for row in manifest:
        if not row['split']:continue
        if row['group_id'] in groups and groups[row['group_id']]!=row['split']:issues.append('Grupo dividido')
        groups[row['group_id']]=row['split']
        if sha(PROJECT/row['binary_mask_path'])!=row['binary_mask_sha256']:issues.append('Máscara derivada cambió: '+row['id'])
    cfg=json.loads((PROJECT/'config/preentrenamiento.json').read_text())
    experiment=json.loads((PROJECT/'config/experimento_propuesto.json').read_text())
    pending=[k for k in ['hardware_approved_by_user','protocol_approved_by_user','literature_selection_finalized'] if not cfg[k]]
    if not experiment['architectures_finalized']:pending.append('architectures_finalized')
    ram=memory_available()
    if ram is None:pending.append('live_free_memory_not_available')
    elif ram<experiment['local_profile']['minimum_free_ram_gib']:pending.append('less_than_2_GiB_free_RAM')
    if not importlib.util.find_spec('torch'):pending.append('training_environment_not_installed_or_validated')
    report={'data_checks_pass':not issues,'data_errors':issues,'pending_before_training':pending,
      'live_free_ram_gib':ram,'training_allowed':False,'training_started':False,
      'note':'Validador de lectura. El entrenamiento sigue deshabilitado por solicitud del investigador.'}
    (out/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 2 if issues else 0

if __name__=='__main__':raise SystemExit(main())
