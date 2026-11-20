import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analisis_integracion_e2 import holm,signflip

class StatisticalChecks(unittest.TestCase):
    def test_holm_reference_and_input_order(self):
        np.testing.assert_allclose(holm([.03,.001,.04]),[.06,.003,.06])
    def test_five_pairs_cannot_reject_exact_two_sided(self):
        self.assertEqual(signflip(np.ones(5)),.0625)
    def test_zero_effect(self):
        self.assertEqual(signflip(np.zeros(7)),1)
    def test_sign_symmetry(self):
        d=np.array([.1,.2,-.4,.3,.7])
        self.assertEqual(signflip(d),signflip(-d))

if __name__=='__main__':unittest.main()
