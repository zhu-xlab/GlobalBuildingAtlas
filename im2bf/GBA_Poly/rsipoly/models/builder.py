# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import pdb

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry
from mmcv.utils import ConfigDict, print_log

MODELS = Registry('models', parent=MMCV_MODELS)
ATTENTION = Registry('attention', parent=MMCV_ATTENTION)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS
GAN_MODULE = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)

def build_gan_module(cfg):
    """Build head."""
    return GAN_MODULE.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    model = SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

    model.init_weights()
    # freeze parameters by prefix
    frozen_parameters = cfg.pop('frozen_parameters', None)
    if frozen_parameters is not None:
        print(f'Frozen parameters: {frozen_parameters}')
        for name, param in model.named_parameters():
            for frozen_prefix in frozen_parameters:
                if frozen_prefix in name:
                    param.requires_grad = False
            if param.requires_grad:
                print(f'Training parameters: {name}')
    return model
