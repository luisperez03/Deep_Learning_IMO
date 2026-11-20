from pathlib import Path
import sys,unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from preparar_base import make_groups,grouped_split,binary_mask,letterbox
from dataset_ready import metrics,partition

class PreparationTests(unittest.TestCase):
    def test_transitive_group(self):
        groups=make_groups(['a','b','c','d'],[{'id_a':'a','id_b':'b'},{'id_a':'b','id_b':'c'}])
        self.assertEqual(sorted(map(len,groups.values())),[1,3])
        self.assertTrue(any(set(v)=={'a','b','c'} for v in groups.values()))
    def test_deterministic_group_split(self):
        groups={str(i):[str(i)] for i in range(40)};f={'train':.7,'val':.15,'test':.15}
        a=grouped_split(groups,f,42);self.assertEqual(a,grouped_split(groups,f,42))
        self.assertEqual(set(a),set(groups));self.assertEqual(set(a.values()),set(f))
    def test_unknown_color_fails(self):
        with self.assertRaises(ValueError):binary_mask(Image.new('RGB',(3,3),'white'),{(0,0,0):'background'})
    def test_letterbox_binary_and_geometry(self):
        im=Image.new('RGB',(20,10),'white');mask=Image.new('L',(20,10),1)
        a,v=letterbox(im,32);b,w=letterbox(mask,32,True)
        self.assertTrue(np.array_equal(v,w));self.assertEqual(int(v.sum()),32*16)
        self.assertEqual(set(np.unique(b)),{0,1});self.assertEqual(a.size,b.size)
    def test_metrics_padding_ignored_and_empty(self):
        target=np.array([[1,0],[0,0]]);pred=np.array([[1,0],[1,1]]);valid=np.array([[1,1],[0,0]],dtype=bool)
        self.assertEqual(metrics(pred,target,valid)['Dice'],1)
        self.assertEqual(metrics(np.zeros((2,2)),np.zeros((2,2)))['Dice'],1)
        self.assertEqual(metrics(np.zeros((2,2)),target)['Dice'],0)
    def test_test_partition_locked(self):
        with self.assertRaises(PermissionError):partition('test')

if __name__=='__main__':unittest.main()
