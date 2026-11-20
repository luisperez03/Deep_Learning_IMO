"""Adaptaciones binarias de arquitecturas reales, sin pesos ni fallbacks.
SMP: DeepLabv3+/UNet++ R18 y SegFormer MiT-B0. Los demás usan fuentes
de autores fijadas en third_party/e2/sources.lock.json.
"""
import copy
import importlib
import sys
import types
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from modelos_base import SmallFCN, SmallSegNet, SmallUNet

SOURCES = Path(__file__).resolve().parents[1] / 'third_party/e2'

def official_package(name, directory):
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
    return name

def resize(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class SMPAdapter(nn.Module):
    def __init__(self, architecture, encoder):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.model = getattr(smp, architecture)(encoder_name=encoder,
            encoder_weights=None, in_channels=3, classes=1, activation=None)

    def forward(self, x):
        return self.model(x)

class DeepLabV3PlusAdapter(SMPAdapter):
    def __init__(self): super().__init__('DeepLabV3Plus', 'resnet18')

class UNetPlusPlusAdapter(SMPAdapter):
    def __init__(self): super().__init__('UnetPlusPlus', 'resnet18')

class SegFormerB0Adapter(SMPAdapter):
    def __init__(self): super().__init__('Segformer', 'mit_b0')

class PIDNetAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        pkg = official_package('e2_pidnet', SOURCES / 'pidnet/models')
        cls = importlib.import_module(pkg + '.pidnet').PIDNet
        self.model = cls(m=2, n=3, num_classes=1, planes=32,
                         ppm_planes=96, head_planes=128, augment=True)

    def forward(self, x):
        p, main, boundary = self.model(x)
        return {'logits': resize(main, x.shape[-2:]),
                'aux': resize(p, x.shape[-2:]),
                'boundary': resize(boundary, x.shape[-2:])}

class TransUNetAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        pkg = official_package('e2_transunet', SOURCES / 'transunet/networks')
        module = importlib.import_module(pkg + '.vit_seg_modeling')
        config = copy.deepcopy(module.CONFIGS['R50-ViT-B_16'])
        config.n_classes, config.n_skip = 1, 3
        config.patches.grid = (16, 16)
        self.model = module.VisionTransformer(config, img_size=256, num_classes=1)
        # No load_from: inicialización aleatoria.

    def forward(self, x): return resize(self.model(x), x.shape[-2:])

class SegNeXtAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        from segnext_compat import build_segnext
        self.encoder, self.decoder = build_segnext(SOURCES / 'segnext')

    def forward(self, x): return resize(self.decoder(self.encoder(x)), x.shape[-2:])

class VMUNetAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError('VM-UNet requiere GPU CUDA y mamba-ssm; no hay fallback a GRU.')
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        pkg = official_package('e2_vmunet', SOURCES / 'vmunet/models/vmunet')
        module = importlib.import_module(pkg + '.vmamba')
        module.selective_scan_fn = selective_scan_fn
        self.model = module.VSSM(in_chans=3, num_classes=1,
            depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2], drop_path_rate=0.2)
        # VSSM expone logits; el wrapper VMUNet aplicaría sigmoid.

    def forward(self, x): return self.model(x)

MODELS_E2 = {'fcn': SmallFCN, 'segnet': SmallSegNet, 'unet': SmallUNet,
    'deeplabv3plus': DeepLabV3PlusAdapter, 'unetplusplus': UNetPlusPlusAdapter,
    'pidnet_s': PIDNetAdapter, 'segnext_t': SegNeXtAdapter,
    'segformer_b0': SegFormerB0Adapter, 'transunet_2d': TransUNetAdapter,
    'vmunet': VMUNetAdapter}

def main_logits(output):
    return output['logits'] if isinstance(output, dict) else output
