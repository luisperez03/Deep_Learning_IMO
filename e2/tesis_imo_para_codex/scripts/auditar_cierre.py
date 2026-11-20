"""Auditoría de solo lectura del experimento congelado. No carga las redes."""
import hashlib
import json
import tarfile
import zipfile
from itertools import product

import numpy as np

from analisis_entregable import METRICS, MODELS, ROOT, SEEDS, csv_rows
from inicio_voc import sha
from preparar_base import make_groups, read_csv


def main():
    prep = ROOT/"resultados/preentrenamiento"
    with zipfile.ZipFile(ROOT/"IMO_test_final.zip") as archive:
        meta = json.loads(archive.read("procedencia.json"))
        detail = csv_rows(archive.read("por_imagen.csv"))
        runs = csv_rows(archive.read("por_corrida.csv"))
    freeze = json.loads((prep/"freeze_manifest.json").read_text())
    for catalog in [meta["frozen_files"],freeze["artifacts"]]:
        for filename,expected in catalog.items():
            assert sha(prep/filename)==expected, filename
    for filename,expected in freeze["inputs"].items():
        assert sha(ROOT/filename)==expected, filename

    manifest = {r["id"]:r for r in read_csv(prep/"manifest.csv")}
    edges = read_csv(ROOT/"resultados/revision_voc/candidatos_similares.csv")
    groups = make_groups(manifest,edges)
    assert all(manifest[i]["group_id"]==g for g,ids in groups.items() for i in ids)
    used_ids, used_groups = set(), set()
    for k, counts in [("train",(1015,567)),("val",(218,122)),("test",(218,122))]:
        rows = read_csv(prep/f"{k}.csv")
        ids = {r["id"] for r in rows}
        group_ids = {r["group_id"] for r in rows}
        assert len(ids)==len(rows) and (len(rows),len(group_ids))==counts
        assert not (ids & used_ids or group_ids & used_groups)
        used_ids.update(ids)
        used_groups.update(group_ids)
        for r in rows:
            assert manifest[r["id"]]==r and r["split"]==k
            for column,root in [("image",ROOT.parent),("original_mask",ROOT.parent),("binary_mask",ROOT)]:
                assert sha(root/r[column+"_path"])==r[column+"_sha256"], r["id"]

    test_ids = [r["id"] for r in read_csv(prep/"test.csv")]
    keys = {(r["model"],int(r["seed"]),r["id"]) for r in detail}
    assert keys==set(product(MODELS,SEEDS,test_ids)) and len(keys)==len(detail)
    run_keys = set(product(MODELS,SEEDS))
    assert {(r["model"],int(r["seed"])) for r in runs}==run_keys and len(runs)==15
    for row in detail:
        assert manifest[row["id"]]["group_id"]==row["group_id"]
        assert all(0<=float(row[k])<=1 for k in METRICS)
        iou = float(row["IoU"])
        assert np.isclose(float(row["Dice"]),2*iou/(1+iou),rtol=0,atol=1e-12)

    tarpath = ROOT/"piloto_T4_final.tar.gz"
    assert sha(tarpath)==meta["archive_sha256"]
    assert {(c["model"],c["seed"]) for c in meta["checkpoints"]}==run_keys
    assert len(meta["checkpoints"])==15
    with tarfile.open(tarpath) as tar:
        for ck in meta["checkpoints"]:
            name = ck["file"]
            path = ROOT/name
            actual = sha(path) if path.exists() else hashlib.sha256(tar.extractfile(name).read()).hexdigest()
            assert actual==ck["sha256"], name
    print(f"Verificados {len(used_ids)} pares elegibles, {len(edges)} enlaces, "
          f"{len(keys)} evaluaciones y 15 checkpoints.")


if __name__ == "__main__":
    main()
