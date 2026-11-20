import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metricas_diagnostico_e2 import (reliability,probability_metrics,fit_temperature,perturb,label_set,border_metrics,
    conformal_quantile,conformal_scores_by_class,conformal_calibrate,conformal_sets)
class Diagnostics(unittest.TestCase):
    def test_perfect_and_padding(self):
        p=np.array([[0.,1.],[.9,.9]]);y=np.array([[0,1],[0,0]]);v=np.array([[1,1],[0,0]],bool)
        r=probability_metrics(p,y,v);self.assertEqual(r['Brier'],0);self.assertEqual(r['ECE15'],0)
    def test_adaptive_does_not_split_ties_by_pixel_order(self):
        self.assertEqual(reliability(np.full(100,.5),np.r_[np.zeros(50),np.ones(50)],adaptive=True),0)
    def test_temperature_reduces_miscalibration(self):
        z=np.full(100,4.);y=np.r_[np.ones(75),np.zeros(25)]
        r=fit_temperature(z,y,np.ones(100));self.assertGreater(r['temperature'],1)
        self.assertLess(r['validation_NLL_after'],r['validation_NLL_before'])
    def test_perturbation_preserves_padding_and_is_deterministic(self):
        x=np.ones((3,8,8),np.float32)*.5;v=np.ones((8,8),bool);v[0]=False;x[:,0]=0
        norm={'mean':[0,0,0],'std':[1,1,1]}
        a=perturb(x,v,norm,'noise',.1,42);b=perturb(x,v,norm,'noise',.1,42)
        np.testing.assert_array_equal(a,b);self.assertTrue(np.all(a[:,0]==0))
    def test_label_sets_are_empirical(self):
        p=np.array([[.95,.5,.05]]);y=np.array([[1,1,1]]);v=np.ones_like(y,bool)
        r=label_set(p,y,v);self.assertAlmostEqual(r['coverage'],2/3);self.assertAlmostEqual(r['mean_set_size'],4/3)
    def test_border_empty_penalty(self):
        y=np.ones((3,4),bool);self.assertEqual(border_metrics(np.zeros_like(y),y,y)['HD95'],5)
    def test_conformal_quantile_finite_sample(self):
        s=np.arange(1,9)/10.;self.assertEqual(conformal_quantile(s,.1),float('inf'))
        s=np.arange(1,20)/20.;self.assertAlmostEqual(conformal_quantile(s,.1),.9)
    def test_mondrian_protects_minority_class(self):
        rng=np.random.default_rng(0);store={0:{},1:{}};v=np.ones((64,64),bool)
        def scene():
            y=np.zeros((64,64),np.uint8);y[:8,:8]=1
            p=np.where(y==1,rng.uniform(.2,1,(64,64)),rng.uniform(0,.05,(64,64)))
            return p,y
        for g in range(40):
            p,y=scene();conformal_scores_by_class(p,y,v,f'g{g}',store)
        q,n=conformal_calibrate(store,.1,10**6,0);self.assertEqual(n['label'],40*64)
        cov={'m':[0,0],'mond':[0,0]}
        for _ in range(40):
            p,y=scene()
            a=conformal_sets(p,y,v,q['label'],q['background'])['label'];b=conformal_sets(p,y,v,q['marginal'],q['marginal'])['label']
            cov['mond'][0]+=a['covered'];cov['mond'][1]+=a['pixels'];cov['m'][0]+=b['covered'];cov['m'][1]+=b['pixels']
        self.assertGreater(cov['mond'][0]/cov['mond'][1],.87)
        self.assertLess(cov['m'][0]/cov['m'][1],.5)
    def test_conformal_sets_ignore_padding(self):
        p=np.array([[.9,.1,.6]]);y=np.array([[1,0,0]]);v=np.array([[1,1,0]],bool)
        r=conformal_sets(p,y,v,.5,.5);self.assertEqual(r['all']['pixels'],2);self.assertEqual(r['all']['covered'],2)
if __name__=='__main__':unittest.main()
