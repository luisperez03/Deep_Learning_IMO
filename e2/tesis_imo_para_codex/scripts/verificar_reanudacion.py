"""Prueba numérica pequeña del checkpoint: continuo versus interrumpido.

Usa imágenes sintéticas de 8x8 y una sola convolución; no entrena baselines.
Se ejecuta antes del piloto, con la misma reserva de memoria.
"""
from pathlib import Path
import json
import tempfile


def main():
    from piloto_local import guard, ROOT
    guard(2)
    import torch
    from unittest.mock import patch
    import colab_runner as runner
    import numpy as np
    temp = ROOT/'resultados/tests_tmp'
    temp.mkdir(exist_ok=True)
    rows = [{'id': 'a', 'group_id': 'g_a'}, {'id': 'b', 'group_id': 'g_b'}]
    def sample(row, device):
        x = torch.arange(192, dtype=torch.float32).reshape(1,3,8,8)/192
        y = torch.zeros(1,1,8,8); y[:,:,2:6,2:6] = 1
        return x.to(device), y.to(device), torch.ones_like(y, dtype=torch.bool).to(device)
    with tempfile.TemporaryDirectory(dir=temp) as folder:
        folder = Path(folder)
        with patch.dict(runner.MODELS, {'tiny': lambda: torch.nn.Conv2d(3,1,1)}), \
             patch.object(runner, 'partition', return_value=rows), \
             patch.object(runner, 'tensors', side_effect=sample):
            kwargs = dict(epochs=2, patience=5, identity='synthetic-resume-test', force_cpu=True)
            runner.run('tiny', 11, folder/'continuous', **kwargs)
            first = runner.run('tiny', 11, folder/'resume', max_epochs_this_call=1, **kwargs)
            assert first['completed'] is False
            second = runner.run('tiny', 11, folder/'resume', **kwargs)
            assert second['completed'] is True and second['epochs_completed'] == 2
            a = torch.load(folder/'continuous/tiny_seed11_lr0.001/last.pt', weights_only=False)
            b = torch.load(folder/'resume/tiny_seed11_lr0.001/last.pt', weights_only=False)
            for name in a['model']:
                assert torch.equal(a['model'][name], b['model'][name]), name
            for left, right in zip(a['history'], b['history']):
                assert left['train_loss'] == right['train_loss']
                assert left['val_group_Dice'] == right['val_group_Dice']
            try:
                runner.run('tiny', 11, folder/'resume', epochs=3, identity='synthetic-resume-test', force_cpu=True)
            except ValueError:
                pass
            else:
                raise AssertionError('No se rechazó cambio de configuración')
    report = {'status':'passed', 'resume_matches_continuous':True,
              'changed_config_rejected':True, 'torch':str(torch.__version__), 'numpy':np.__version__,
              'scope':'Modelo sintético; no prueba equivalencia determinista de SegNet'}
    (ROOT/'resultados/verificacion_reanudacion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
