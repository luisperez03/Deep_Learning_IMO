"""Pruebas automatizadas con datos sintéticos; NO se usan los datasets del usuario."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

MODULE_FILE = Path(__file__).resolve().parents[1] / "scripts" / "auditar_datasets.py"
spec = importlib.util.spec_from_file_location("auditar_datasets", MODULE_FILE)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


class AuditoriaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config").mkdir()
        (self.root / "config" / "datasets.json").write_text(json.dumps({
            "datasets": [
                {"nombre": "Hazmat", "ruta": "Hazmat Pacards.v5i.coco"},
                {"nombre": "VOC", "ruta": "VOC2012"},
                {"nombre": "SeaFront", "ruta": "SeaFront_v1_0_0"}
            ], "directorio_resultados": "resultados/auditoria"
        }), encoding="utf-8")
        # COCO: caja y polígono real.
        c_root = self.root / "Hazmat Pacards.v5i.coco" / "train"
        (c_root / "images").mkdir(parents=True)
        Image.new("RGB", (32, 24), "red").save(c_root / "images" / "haz.jpg")
        coco = {
            "images": [{"id": 10, "file_name": "haz.jpg", "width": 32, "height": 24}],
            "categories": [{"id": 1, "name": "danger-class"}],
            "annotations": [
                {"id": 1, "image_id": 10, "category_id": 1, "bbox": [3, 3, 10, 10],
                 "segmentation": [[3, 3, 13, 3, 13, 13, 3, 13]]},
                {"id": 2, "image_id": 10, "category_id": 1, "bbox": [8, 6, 9, 8]},
            ]
        }
        (c_root / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
        # VOC: objeto y máscara de segmentación.
        v_root = self.root / "VOC2012"
        (v_root / "JPEGImages").mkdir(parents=True)
        (v_root / "Annotations").mkdir()
        (v_root / "SegmentationClass").mkdir()
        Image.new("RGB", (32, 24), "blue").save(v_root / "JPEGImages" / "voc.jpg")
        Image.new("P", (32, 24), 1).save(v_root / "SegmentationClass" / "voc.png")
        (v_root / "Annotations" / "voc.xml").write_text(
            "<annotation><filename>voc.jpg</filename><object><name>container</name>"
            "<bndbox><xmin>1</xmin><ymin>1</ymin><xmax>20</xmax><ymax>20</ymax>"
            "</bndbox></object></annotation>", encoding="utf-8")
        # YOLO: dos instancias en una imagen y duplicado intencional entre splits.
        y_root = self.root / "SeaFront_v1_0_0"
        (y_root / "images" / "train").mkdir(parents=True)
        (y_root / "images" / "val").mkdir(parents=True)
        (y_root / "labels" / "train").mkdir(parents=True)
        Image.new("RGB", (32, 24), "green").save(y_root / "images" / "train" / "sea.jpg")
        (y_root / "images" / "val" / "sea_duplicado.jpg").write_bytes(
            (y_root / "images" / "train" / "sea.jpg").read_bytes())
        (y_root / "labels" / "train" / "sea.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n1 0.7 0.7 0.2 0.2\n", encoding="utf-8")
        (y_root / "data.yaml").write_text("names:\n  0: placard\n  1: other\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_detecta_tres_formatos_y_mascaras(self):
        exit_code = auditor.main(["--root", str(self.root), "--todas-imagenes", "--hash-duplicados"])
        self.assertEqual(exit_code, 0)
        output = self.root / "resultados" / "auditoria"
        report = json.loads((output / "resumen.json").read_text(encoding="utf-8"))
        haz, voc, sea = report["datasets"]
        self.assertEqual(haz["imagenes"], 1)
        self.assertEqual(haz["anotaciones"]["coco"]["objetos"], 2)
        self.assertEqual(haz["anotaciones"]["coco"]["poligonos"], 1)
        self.assertEqual(haz["anotaciones"]["coco"]["imagenes_no_encontradas"], 0)
        self.assertEqual(voc["mascaras_archivos"], 1)
        self.assertEqual(voc["anotaciones"]["voc"]["objetos"], 1)
        self.assertEqual(sea["anotaciones"]["yolo"]["objetos"], 2)
        self.assertEqual(sea["anotaciones"]["yolo"]["nombres"]["0"], "placard")
        self.assertGreaterEqual(report["duplicados"]["grupos_entre_splits"], 1)
        self.assertTrue((output / "INFORME_AUDITORIA.md").exists())
        self.assertTrue((output / "clases.csv").exists())

    def test_muestrarios_visualizables(self):
        spec_visual = importlib.util.spec_from_file_location(
            "crear_muestrarios", Path(__file__).resolve().parents[1] / "scripts" / "crear_muestrarios.py")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        visual = importlib.util.module_from_spec(spec_visual)
        spec_visual.loader.exec_module(visual)
        self.assertEqual(visual.main(["--root", str(self.root), "--limite-por-dataset", "4"]), 0)
        folder = self.root / "resultados" / "auditoria" / "muestrarios"
        self.assertTrue((folder / "Hazmat_muestrario.jpg").exists())
        self.assertTrue((folder / "VOC_muestrario.jpg").exists())
        self.assertTrue((folder / "SeaFront_muestrario.jpg").exists())

    def test_ninguna_modificacion_de_originales(self):
        c = self.root / "Hazmat Pacards.v5i.coco" / "train" / "_annotations.coco.json"
        before = c.read_bytes()
        auditor.main(["--root", str(self.root), "--muestra-imagenes", "1"])
        self.assertEqual(c.read_bytes(), before)

    def test_particiones_y_detector_yolo(self):
        self.assertEqual(auditor.split_name(Path("images/valid/a.jpg")), "val")
        self.assertEqual(auditor.split_name(Path("annotations/test.json")), "test")
        self.assertTrue(auditor.is_yolo_label(Path("labels/train/x.txt")))
        self.assertFalse(auditor.is_yolo_label(Path("ImageSets/Main/train.txt")))


if __name__ == "__main__":
    unittest.main()
