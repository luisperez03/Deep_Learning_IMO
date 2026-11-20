import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from completar_analisis_locales_e2 import border
class Borders(unittest.TestCase):
    def test_identical_and_padding(self):
        y=np.zeros((10,10),bool);y[3:6,3:6]=1;v=np.ones_like(y);v[0]=0
        p=y.copy();p[0]=1;r=border(p,y,v)
        self.assertEqual(r['HD95'],0);self.assertEqual(r['ASSD'],0);self.assertEqual(r['precision'],1)
    def test_empty_is_explicit_penalty(self):
        y=np.ones((3,4),bool);r=border(np.zeros_like(y),y,y)
        self.assertEqual(r['HD95'],5);self.assertEqual(r['recall'],0);self.assertTrue(r['empty_prediction'])
    def test_translated_single_point(self):
        y=np.zeros((8,8),bool);p=y.copy();y[2,2]=1;p[2,5]=1
        r=border(p,y,np.ones_like(y));self.assertEqual(r['HD95'],3);self.assertEqual(r['ASSD'],3)
if __name__=='__main__':unittest.main()
