"""Una orden para regenerar el informe desde artefactos locales, sin GPU."""
import subprocess
import sys
from pathlib import Path
P=Path(__file__).resolve().parents[1]
for args in [
    ['-m','unittest','discover','-s','tests','-p','test_integracion_e2.py'],
    # Auditoría y figura del test antes de integrar: integrar_entregable2.py copia
    # resultados/auditoria_test_e2/comparacion_validacion_test.png, que no existe en un clon nuevo.
    ['scripts/auditar_test_e2.py'],
    ['scripts/graficar_test_e2.py'],
    ['scripts/analisis_integracion_e2.py'],
    ['scripts/integrar_entregable2.py'],
]:
    print('Ejecutando:', ' '.join(args),flush=True)
    subprocess.run([sys.executable,*args],cwd=P,check=True)
book=Path(sys.executable).with_name('jupyter-book.exe' if sys.platform=='win32' else 'jupyter-book')
subprocess.run([str(book),'build','libro','-W','--keep-going'],cwd=P,check=True)
print('Informe: libro/_build/html/index.html')
