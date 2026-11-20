"""Espera hasta 180 s por RAM y ejecuta solo cinco pasos FCN, guardando log."""
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root/'resultados/preentrenamiento'
status = out/'lanzador_local.json'

class Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                *[(n, ctypes.c_ulonglong) for n in ['total', 'available', 'totalpage',
                  'availablepage', 'totalvirtual', 'availablevirtual', 'extended']]]

def available():
    state = Memory(); state.length = ctypes.sizeof(Memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        return 0
    return state.available/2**30

def record(**values):
    status.write_text(json.dumps(values, indent=2), encoding='utf-8')

record(status='waiting_for_memory', max_wait_seconds=180)
time.sleep(15)
for attempt in range(34):
    ram = available()
    if ram >= 2:
        record(status='running_five_steps', free_ram_gib=ram)
        with (out/'prueba_local.log').open('w', encoding='utf-8') as log:
            try:
                result = subprocess.run([sys.executable, str(root/'scripts/prueba_carga.py'),
                                         '--model', 'fcn'], cwd=root, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=180)
                record(status='finished', returncode=result.returncode)
            except subprocess.TimeoutExpired:
                record(status='stopped_timeout', max_run_seconds=180)
        break
    record(status='waiting_for_memory', free_ram_gib=ram, attempt=attempt)
    time.sleep(5)
else:
    record(status='not_started_low_memory', free_ram_gib=available())
