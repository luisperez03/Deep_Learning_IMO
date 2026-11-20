"""Casos que impiden interpretar similitud visual como identidad exacta."""
from pathlib import Path
import sys,unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from revisar_voc import visual_hashes,pixel_hash,candidate_pairs

class SimilarityTests(unittest.TestCase):
    def test_uniform_images_collide_perceptually_but_not_exactly(self):
        # Fondo oscuro y claro pueden compartir dHash y aHash. Es una limitación
        # real: el programa debe conservar el resultado como candidato.
        a=Image.new('RGB',(20,20),(10,10,10)); b=Image.new('RGB',(20,20),(200,200,200))
        self.assertEqual(visual_hashes(a),visual_hashes(b))
        self.assertNotEqual(pixel_hash(a),pixel_hash(b))

    def test_pixel_hash_includes_size(self):
        a=Image.new('RGB',(8,4),'black');b=Image.new('RGB',(4,8),'black')
        self.assertEqual(a.tobytes(),b.tobytes())
        self.assertNotEqual(pixel_hash(a),pixel_hash(b))

    def test_pairs_include_cross_split_and_threshold_boundary(self):
        def row(key,split,dh):
            return {'id':key,'original_split':split,'source_folder_xml':'test','dhash':dh,'ahash':0}
        pairs=candidate_pairs([row('a','train',0),row('b','val',3),row('c','train',255)],2,0)
        self.assertEqual(len(pairs),1)
        self.assertTrue(pairs[0]['cross_train_val'])
        self.assertEqual(pairs[0]['dhash_distance'],2)

    def test_hash_repeatable(self):
        image=Image.fromarray(np.arange(16*16*3,dtype=np.uint8).reshape(16,16,3))
        self.assertEqual(visual_hashes(image),visual_hashes(image.copy()))
        self.assertEqual(pixel_hash(image),pixel_hash(image.copy()))

if __name__=='__main__':unittest.main()
