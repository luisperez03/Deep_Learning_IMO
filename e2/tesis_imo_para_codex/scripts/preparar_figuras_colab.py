"""Prepara un paquete pequeño para inspeccionar tres ejemplos sin repetir el test."""
import ast
import csv
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"colab"
IDS = ["image_1656","image_1241","image_1118"]


def main():
    with zipfile.ZipFile(ROOT/"IMO_test_final.zip") as archive:
        provenance = json.loads(archive.read("procedencia.json"))
    prep = ROOT/"resultados/preentrenamiento"
    records = (prep/"test.csv").read_bytes()
    assert hashlib.sha256(records).hexdigest() == provenance["frozen_files"]["test.csv"]
    rows = {r["id"]:r for r in csv.DictReader(io.StringIO(records.decode("utf-8-sig")))}
    stream = io.StringIO()
    writer = csv.DictWriter(stream,fieldnames=rows[IDS[0]])
    writer.writeheader()
    writer.writerows(rows[i] for i in IDS)
    paths = {}
    for source in provenance["checkpoints"]:
        if source["seed"] == 11:
            paths[source["model"]+".pt"] = (ROOT/source["file"],source["sha256"])
    norm = "resultados/preentrenamiento/normalization_train.json"
    paths[norm] = ROOT/norm,provenance["frozen_files"]["normalization_train.json"]
    for identifier in IDS:
        row = rows[identifier]
        for name,digest in [("../"+row["image_path"],row["image_sha256"]),
                            (row["binary_mask_path"],row["binary_mask_sha256"])]:
            paths[name] = ROOT/name,digest
    result = OUT/"IMO_figuras_colab.zip"
    with tempfile.TemporaryDirectory(dir=OUT) as temporary:
        bundle = Path(temporary)/result.name
        with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as archive:
            for name,(file,digest) in paths.items():
                data = file.read_bytes()
                assert hashlib.sha256(data).hexdigest() == digest
                member = file.resolve().relative_to(ROOT.parent).as_posix() if name.startswith("../") else ROOT.name+"/"+name
                archive.writestr(member,data)
            for name in ["figuras_colab","modelos_base","dataset_ready","preparar_base","inicio_voc"]:
                file = ROOT/"scripts"/(name+".py")
                archive.write(file,ROOT.name+"/scripts/"+file.name)
            archive.writestr(ROOT.name+"/casos.csv",stream.getvalue())
            archive.writestr(ROOT.name+"/huellas.json",json.dumps({k:v[1] for k,v in paths.items()}))
        bundle.replace(result)
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    code = f"""from pathlib import Path
import hashlib, shutil, subprocess, sys, zipfile
bundle = Path('/content/IMO_figuras_colab.zip')
if hashlib.sha256(bundle.read_bytes()).hexdigest() != '{digest}':
    raise ValueError('Sube el ZIP que acompaña a este cuaderno.')
destination = Path('/content/imo_cualitativo')
with zipfile.ZipFile(bundle) as archive:
    archive.extractall(destination)
subprocess.run([sys.executable, str(destination/'tesis_imo_para_codex/scripts/figuras_colab.py')], check=True)
shutil.make_archive('/content/paneles_IMO', 'zip', '/content/IMO_figuras')
from IPython.display import Image, display
display(Image(filename='/content/IMO_figuras/comparacion.png'))
"""
    ast.parse(code)
    cells = [dict(cell_type="markdown",id="uso",metadata={},source=["# Aciertos y errores de segmentación\n","En VS Code elige un kernel Colab con GPU. Usa Upload to Colab para subir IMO_figuras_colab.zip a /content. Ejecuta la celda y descarga /content/paneles_IMO.zip. El paquete contiene tres fotos locales para revisión privada y los mejores checkpoints de semilla 11."]),
             dict(cell_type="code",id="panel",metadata={},source=code.splitlines(True),outputs=[],execution_count=None)]
    notebook = dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={"kernelspec":{"name":"python3","language":"python","display_name":"Python 3"},"accelerator":"GPU"})
    (OUT/"04_IMO_Figuras_Colab.ipynb").write_text(json.dumps(notebook,indent=2,ensure_ascii=False),encoding="utf-8")


if __name__ == "__main__":
    main()
