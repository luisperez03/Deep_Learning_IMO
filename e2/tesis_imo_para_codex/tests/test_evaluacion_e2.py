import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluar_e2_cerrado import grouped_scores,summarize
from experimento_e2_colab import digest

class EvaluationTests(unittest.TestCase):
    def test_groups_not_images_equal_weight(self):
        rows=[dict(id=str(i),group_id=g,**{k:v for k in ['Dice','IoU','mIoU','accuracy']})
              for i,(g,v) in enumerate([('a',1.),('a',1.),('b',0.)])]
        means,_=grouped_scores(rows)
        self.assertEqual(means['Dice'],.5)

    def test_complete_summary_paired_difference_and_integrity(self):
        seeds=[11,23,37,53,71];models=['fcn','transunet_2d'];runs=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model in models:
                for seed in seeds:
                    name=f'{model}_seed{seed}';runs.append({'folder':name})
                    value=.6 if model=='fcn' else .7
                    metrics={k:value for k in ['Dice','IoU','mIoU','accuracy']}
                    result={'signature':'fixed','model':model,'seed':seed,'means':metrics,
                            'groups':{'a':metrics,'b':metrics},'seconds_inference':1}
                    result['payload_sha256']=digest(result)
                    (root/(name+'.json')).write_text(json.dumps(result))
            protocol={'runs':runs,'seeds':seeds,'test_order':['x','y']}
            out=summarize(root,protocol,'fixed')
            self.assertAlmostEqual(out['contrasts'][0]['Dice_difference'],.1)
            self.assertAlmostEqual(out['contrasts'][0]['CI95_groups_fixed_seeds'][0],.1)
            bad=root/'fcn_seed11.json';r=json.loads(bad.read_text());r['signature']='changed';bad.write_text(json.dumps(r))
            with self.assertRaises(RuntimeError):summarize(root,protocol,'fixed')

if __name__=='__main__':unittest.main()
