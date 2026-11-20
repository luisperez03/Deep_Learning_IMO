"""Infraestructura mínima para fuentes intactas MSCAN/LightHamHead Apache-2.0.
Reemplaza registro/MMCV, SyncBN por BN de un dispositivo, y dispositivo NMF.
No cambia bloques ni profundidades. No es una instalación de MMSegmentation.
"""
import ast
import types
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath

class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        if init_cfg is not None: raise ValueError('No se permite inicialización externa')
        self.init_cfg = None

def build_norm_layer(cfg, channels):
    if cfg['type'] in ('BN', 'SyncBN'):
        layer = nn.BatchNorm2d(channels, eps=cfg.get('eps', 1e-5), momentum=cfg.get('momentum', .1))
    elif cfg['type'] == 'GN':
        layer = nn.GroupNorm(cfg['num_groups'], channels, eps=cfg.get('eps', 1e-5))
    else: raise ValueError(cfg)
    for p in layer.parameters(): p.requires_grad_(cfg.get('requires_grad', True))
    return 'norm', layer

def constant_init(module, val, bias=0.):
    nn.init.constant_(module.weight, val)
    if module.bias is not None: nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0., std=1., bias=0.):
    nn.init.normal_(module.weight, mean, std)
    if module.bias is not None: nn.init.constant_(module.bias, bias)

def trunc_normal_init(module, std=.02, bias=0.):
    nn.init.trunc_normal_(module.weight, std=std)
    if module.bias is not None: nn.init.constant_(module.bias, bias)

class ConvModule(nn.Module):
    def __init__(self, a, b, kernel, conv_cfg=None, norm_cfg=None, act_cfg=dict(type='ReLU')):
        super().__init__()
        if conv_cfg is not None: raise ValueError(conv_cfg)
        self.conv = nn.Conv2d(a, b, kernel, bias=norm_cfg is None)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None: nn.init.zeros_(self.conv.bias)
        self.norm = build_norm_layer(norm_cfg, b)[1] if norm_cfg else nn.Identity()
        if act_cfg is not None and act_cfg['type'] != 'ReLU': raise ValueError(act_cfg)
        self.activation = nn.ReLU(inplace=True) if act_cfg else nn.Identity()

    def forward(self, x): return self.activation(self.norm(self.conv(x)))

class BaseDecodeHead(nn.Module):
    def __init__(self, in_channels, in_index, channels, num_classes, norm_cfg,
                 dropout_ratio=.1, align_corners=False, input_transform='multiple_select', **kwargs):
        super().__init__()
        if kwargs or input_transform != 'multiple_select': raise ValueError(kwargs)
        self.in_channels, self.in_index, self.channels = in_channels, in_index, channels
        self.norm_cfg, self.conv_cfg = norm_cfg, None
        self.act_cfg, self.align_corners = dict(type='ReLU'), align_corners
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
        normal_init(self.conv_seg, std=.01)

    def _transform_inputs(self, inputs): return [inputs[i] for i in self.in_index]
    def cls_seg(self, x): return self.conv_seg(self.dropout(x))

def load_classes(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    kept = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or '').startswith(('mmcv', 'mmseg'))): continue
        if isinstance(node, ast.ClassDef): node.decorator_list = []
        kept.append(node)
    tree.body = kept
    namespace = dict(BaseModule=BaseModule, BaseDecodeHead=BaseDecodeHead,
        ConvModule=ConvModule, build_norm_layer=build_norm_layer, DropPath=DropPath,
        constant_init=constant_init, normal_init=normal_init,
        trunc_normal_init=trunc_normal_init, resize=F.interpolate)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace

def build_segnext(root):
    mscan = load_classes(root / 'mmseg/models/backbones/mscan.py')
    ham = load_classes(root / 'mmseg/models/decode_heads/ham_head.py')
    encoder = mscan['MSCAN'](embed_dims=[32,64,160,256], mlp_ratios=[8,8,4,4],
        depths=[3,3,5,2], drop_rate=0., drop_path_rate=.1, norm_cfg=dict(type='BN'))
    encoder.init_weights()
    decoder = ham['LightHamHead'](in_channels=[64,160,256], in_index=[1,2,3],
        channels=256, ham_channels=256, ham_kwargs=dict(MD_R=16),
        dropout_ratio=.1, num_classes=1, norm_cfg=dict(type='GN', num_groups=32), align_corners=False)
    nmf = decoder.hamburger.ham
    nmf.register_buffer('_device_anchor', torch.empty(0), persistent=False)
    def device_bases(self, B, S, D, R, cuda=False):
        return F.normalize(torch.rand(B*S, D, R, device=self._device_anchor.device), dim=1)
    nmf._build_bases = types.MethodType(device_bases, nmf)
    return encoder, decoder
