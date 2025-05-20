# Copyright (c) OpenMMLab. All rights reserved.
from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .dpt_head import DPTHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .isa_head import ISAHead
from .knet_head import IterativeDecodeHead, KernelUpdateHead, KernelUpdator
from .lraspp_head import LRASPPHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .segformer_head import SegformerHead
from .segmenter_mask_head import SegmenterMaskTransformerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .setr_mla_head import SETRMLAHead
from .setr_up_head import SETRUPHead
from .stdc_head import STDCHead
from .uper_head import UPerHead
from .matching_head import OptimalMatching
# from .matching_head_v2 import OptimalMatchingV2
# from .matching_head_v3 import OptimalMatchingV3
# from .matching_head_v4 import OptimalMatchingV4
# from .matching_head_v5 import OptimalMatchingV5
# from .matching_head_v6 import OptimalMatchingV6
# from .matching_head_v7 import OptimalMatchingV7
# from .matching_head_v9 import OptimalMatchingV9
# from .matching_head_v10 import OptimalMatchingV10
# from .matching_head_v11 import OptimalMatchingV11
# from .matching_head_v12 import OptimalMatchingV12
# from .matching_head_v13 import OptimalMatchingV13
# from .matching_head_v14 import OptimalMatchingV14
# from .point_head_v1 import PointHeadV1
# from .matching_head import NonMaxSuppression
# from .point_det_head import PointDetHead
# from .point_reg_head import PointRegHead
# from .post_processor import PostProcessor
# from .crf_head import DenseCRFHead
"""
from .deep_q_net_v1 import DeepQNetV1
from .deep_q_net_v2 import DeepQNetV2
from .deep_q_net_v3 import DeepQNetV3
from .deep_q_net_v4 import DeepQNetV4
from .deep_q_net_v5 import DeepQNetV5
from .deep_q_net_v6 import DeepQNetV6
from .deep_q_net_v7 import DeepQNetV7
from .deep_q_net_v8 import DeepQNetV8
from .deep_q_net_v9 import DeepQNetV9
from .deep_q_net_v12 import DeepQNetV12
from .deep_q_net_v10 import DeepQNetV10
from .deep_q_net_v13 import DeepQNetV13
from .deep_q_net_v14 import DeepQNetV14
"""

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'APCHead', 'DMHead', 'LRASPPHead', 'SETRUPHead',
    'SETRMLAHead', 'DPTHead', 'SETRMLAHead', 'SegmenterMaskTransformerHead',
    'SegformerHead', 'ISAHead', 'STDCHead', 'IterativeDecodeHead',
    'KernelUpdateHead', 'KernelUpdator',
    # 'OptimalMatching',
    # 'NonMaxSuppression',
    # 'PointDetHead', 'PointRegHead', 'OptimalMatchingV3', 'OptimalMatchingV5',
    # 'OptimalMatchingV4', 'PostProcessor', 'OptimalMatchingV6', 'OptimalMatchingV7',
    # 'OptimalMatchingV9',
    # 'DenseCRFHead',
    # 'OptimalMatchingV10', 'OptimalMatchingV11',
    # 'OptimalMatchingV12', 'OptimalMatchingV13', 'PointHeadV1',
    # 'DeepQNetV1', 'DeepQNetV2', 'DeepQNetV3', 'DeepQNetV4', 'OptimalMatchingV14', 'DeepQNetV5',
    # 'DeepQNetV6', 'DeepQNetV7', 'DeepQNetV8', 'DeepQNetV9', 'DeepQNetV10',
    # 'DeepQNetV12', 'DeepQNetV13', 'DeepQNetV14'
]
