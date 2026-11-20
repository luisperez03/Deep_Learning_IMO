"""Pruebas del flujo inicial sin depender de los datasets reales."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
from PIL import Image

spec=importlib.util.spec_from_file_location('inicio_voc',Path(__file__).resolve().parents[1]/'scripts/inicio_voc.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class MaskTests(unittest.TestCase):
    def test_unknown_color_is_not_foreground(self):
        im=Image.new('RGB',(3,1)); im.putdata([(0,0,0),(128,0,0),(255,255,255)])
        known,unknown,fraction=module.mask_stats(im,{(0,0,0):'background',(128,0,0):'label'})
        self.assertEqual(known,{'background':1,'label':1})
        self.assertEqual(sum(unknown.values()),1)
        self.assertAlmostEqual(fraction,1/3)

    def test_palette_mask_decoded_as_rgb(self):
        im=Image.new('P',(2,1)); pal=[0]*768;pal[3:6]=[64,0,0];im.putpalette(pal);im.putdata([0,1])
        known,unknown,fraction=module.mask_stats(im,{(0,0,0):'background',(64,0,0):'organic-peroxide'})
        self.assertFalse(unknown);self.assertEqual(known['organic-peroxide'],1);self.assertEqual(fraction,0.5)

    def test_empty_mask(self):
        known,unknown,fraction=module.mask_stats(Image.new('RGB',(8,8)),{(0,0,0):'background'})
        self.assertEqual(fraction,0);self.assertEqual(known['background'],64)

    def test_pipeline_reproducible_and_preserves_sources(self):
        original_project=module.PROJECT
        temp_root=original_project/'resultados/tests_tmp'
        temp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            assert Path(tmp).resolve().is_relative_to(temp_root.resolve())
            root=Path(tmp); project=root/'project'; dataset=root/'VOC2012'
            for p in [project/'config',dataset/'JPEGImages',dataset/'SegmentationClass',dataset/'ImageSets',dataset/'bboxes/annotations']:p.mkdir(parents=True)
            (project/'config/inicio_voc.json').write_text(json.dumps({'dataset':'VOC2012','seed':42,'sample_size':1,'mask_directory':'SegmentationClass','output_directory':'results'}))
            (dataset/'labelmap.txt').write_text('background:0,0,0::\nlabel:128,0,0::\n')
            for split,text in [('train','a\n'),('val',''),('trainval','a\n')]:
                (dataset/'ImageSets'/f'{split}.txt').write_text(text)
            Image.new('RGB',(8,8),'white').save(dataset/'JPEGImages/a.jpg')
            Image.new('RGB',(8,8),(128,0,0)).save(dataset/'SegmentationClass/a.png')
            before={str(p):module.sha(p) for p in dataset.rglob('*') if p.is_file()}
            module.PROJECT=project
            try:
                module.main(['--data-root',str(root)])
                first=(project/'results/inventario.csv').read_bytes()
                first_summary=(project/'results/resumen.json').read_bytes()
                module.main(['--data-root',str(root)])
                self.assertEqual(first,(project/'results/inventario.csv').read_bytes())
                self.assertEqual(first_summary,(project/'results/resumen.json').read_bytes())
                self.assertEqual(before,{str(p):module.sha(p) for p in dataset.rglob('*') if p.is_file()})
            finally:module.PROJECT=original_project

if __name__=='__main__':unittest.main()
