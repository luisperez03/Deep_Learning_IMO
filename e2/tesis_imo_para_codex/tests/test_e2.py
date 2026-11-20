"""Pruebas sintéticas del contrato E2, sin entrenar con imágenes del proyecto."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import experimento_e2_colab as runner

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.conv=torch.nn.Conv2d(3,1,1); self.drop=torch.nn.Dropout(.2)
    def forward(self,x): return self.conv(self.drop(x))

class Contracts(unittest.TestCase):
    def test_padding_loss(self):
        z=torch.randn(2,1,8,8,requires_grad=True); y=torch.zeros_like(z)
        valid=torch.ones_like(z,dtype=torch.bool);valid[:,:,:2]=False
        first=runner.binary_loss(z,y,valid)
        altered=z.detach().clone();altered[~valid]=1000
        self.assertTrue(torch.equal(first.detach(),runner.binary_loss(altered,y,valid)))
        first.backward();self.assertEqual(z.grad[~valid].abs().sum().item(),0)

    def test_boundary_padding(self):
        y=torch.ones(2,1,8,8);valid=torch.ones_like(y,dtype=torch.bool);valid[:,:,:2]=False
        edge,core=runner.boundary_target(y,valid)
        self.assertFalse(core[:,:,:3].any());self.assertEqual(edge[core].sum().item(),0)

    def test_batch_tail_no_loss_or_duplicate(self):
        batches=runner.training_batches(list(range(5)),11,0,2)
        self.assertEqual(sorted(sum(batches,[])),list(range(5)))
        self.assertEqual([len(b) for b in batches],[2,3])

    def test_resume_matches_and_rejects_changed_identity(self):
        config=json.loads((runner.PROJECT/'config/experimento_e2.json').read_text())
        config.update(epochs=3,patience=10)
        rows=[{'id':str(i),'group_id':str(i)} for i in range(4)]
        def tensor_batch(batch,device):
            x=torch.stack([torch.full((3,8,8),.1+int(r['id'])/10) for r in batch]).to(device)
            y=torch.zeros(len(batch),1,8,8,device=device);y[:,:,2:6,2:6]=1
            return x,y,torch.ones_like(y,dtype=torch.bool)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(runner.MODELS_E2,{'fcn':Tiny}), \
             patch.object(runner,'partition',return_value=rows), patch.object(runner,'batch_tensors',side_effect=tensor_batch):
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            runner.train_model('fcn',11,a,config,{'data':'same'},torch.device('cpu'))
            runner.train_model('fcn',11,b,config,{'data':'same'},torch.device('cpu'),stop_after=1)
            runner.train_model('fcn',11,b,config,{'data':'same'},torch.device('cpu'))
            sa=torch.load(a/'fcn_seed11/last.pt',weights_only=False)
            sb=torch.load(b/'fcn_seed11/last.pt',weights_only=False)
            for key in sa['model']: self.assertTrue(torch.equal(sa['model'][key],sb['model'][key]))
            with self.assertRaises(RuntimeError):
                runner.train_model('fcn',11,b,config,{'data':'changed'},torch.device('cpu'))

if __name__=='__main__': unittest.main()
