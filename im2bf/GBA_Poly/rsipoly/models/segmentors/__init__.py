# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .encoder_decoder import EncoderDecoder
from .regular_encoder_decoder import RegularEncoderDecoder
from .poly_encoder_decoder import PolyEncoderDecoder
# from .poly_encoder_decoder_v2 import PolyEncoderDecoderV2
# from .poly_encoder_decoder_v3 import PolyEncoderDecoderV3
# from .poly_encoder_decoder_v4 import PolyEncoderDecoderV4
# from .poly_encoder_decoder_v5 import PolyEncoderDecoderV5
# from .poly_encoder_decoder_v6 import PolyEncoderDecoderV6
# from .poly_encoder_decoder_v7 import PolyEncoderDecoderV7
# from .poly_encoder_decoder_v8 import PolyEncoderDecoderV8
# from .poly_encoder_decoder_v9 import PolyEncoderDecoderV9
# from .poly_encoder_decoder_v11 import PolyEncoderDecoderV11
# from .poly_dqn_encoder_decoder_v1 import PolyDQNEncoderDecoderV1
# from .poly_dqn_encoder_decoder_v4 import PolyDQNEncoderDecoderV4
# from .poly_dqn_encoder_decoder_v5 import PolyDQNEncoderDecoderV5
# from .poly_dqn_encoder_decoder_v8 import PolyDQNEncoderDecoderV8
# from .poly_dqn_encoder_decoder_v9 import PolyDQNEncoderDecoderV9
# from .poly_dqn_encoder_decoder_v10 import PolyDQNEncoderDecoderV10
# from .poly_dqn_encoder_decoder_v11 import PolyDQNEncoderDecoderV11
# from .poly_dqn_encoder_decoder_v14 import PolyDQNEncoderDecoderV14
# from .poly_simplifier_v1 import PolySimplifierV1
# from .poly_simplifier_v2 import PolySimplifierV2
# from .poly_simplifier_v3 import PolySimplifierV3
# from .poly_simplifier_v4 import PolySimplifierV4
# from .poly_simplifier_v5 import PolySimplifierV5
# from .poly_simplifier_v6 import PolySimplifierV6
# from .poly_simplifier_v7 import PolySimplifierV7
# from .poly_simplifier_v8 import PolySimplifierV8
# from .poly_simplifier_v9 import PolySimplifierV9
# from .poly_simplifier_v10 import PolySimplifierV10
# from .poly_simplifier_v11 import PolySimplifierV11
# from .poly_simplifier_v12 import PolySimplifierV12
# from .poly_simplifier_v13 import PolySimplifierV13
# from .poly_simplifier_v14 import PolySimplifierV14
# from .poly_simplifier_v15 import PolySimplifierV15
# from .poly_simplifier_v16 import PolySimplifierV16
# from .poly_simplifier_v17 import PolySimplifierV17 # train on 103 shapes
# from .poly_simplifier_v18 import PolySimplifierV18 # clustering
# from .poly_regularizer_v2 import PolyRegularizerV2
# from .poly_regularizer_v3 import PolyRegularizerV3
# from .poly_regularizer_v4 import PolyRegularizerV4
from .poly_regularizer_v5 import PolyRegularizerV5
# from .poly_regularizer_v6 import PolyRegularizerV6
# from .poly_regularizer_v7 import PolyRegularizerV7
# from .poly_regularizer_v8 import PolyRegularizerV8 # end to end
# from .poly_regularizer_v9 import PolyRegularizerV9 # end to end
# from .polygonizer_v1 import PolygonizerV1
# from .polygonizer_v2 import PolygonizerV2
# from .polygonizer_v3 import PolygonizerV3
# from .polygonizer_v4 import PolygonizerV4
# from .polygonizer_v5 import PolygonizerV5 # end-to-end segmentation + polygonization
# from .polygonizer_v6 import PolygonizerV6 # end-to-end segmentation + polygonization
# from .polygonizer_v7 import PolygonizerV7 # end-to-end segmentation + polygonization
# from .polygonizer_v9 import PolygonizerV9 # train segmentation + polygonization separately
from .polygonizer_v10 import PolygonizerV10 # train segmentation + polygonization separately
# from .dp_polygonizer import DPPolygonizer # train segmentation + polygonization separately

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder', 'PolyEncoderDecoder',
           # 'PolyEncoderDecoderV2', 'PolyEncoderDecoderV3', 'PolyEncoderDecoderV4',
           # 'PolyEncoderDecoderV5', 'PolyEncoderDecoderV6', 'PolyEncoderDecoderV7',
           # 'PolyEncoderDecoderV8', 'PolyEncoderDecoderV9', 'RegularEncoderDecoder',
           # 'PolyEncoderDecoderV11', 'PolyDQNEncoderDecoderV1', 'PolyDQNEncoderDecoderV4',
           # 'PolyDQNEncoderDecoderV5', 'PolyDQNEncoderDecoderV8', 'PolyDQNEncoderDecoderV9',
           # 'PolyDQNEncoderDecoderV10', 'PolyDQNEncoderDecoderV11', 'PolyDQNEncoderDecoderV14',
           # 'PolySimplifierV1', 'PolySimplifierV2', 'PolySimplifierV3', 'PolySimplifierV4',
           # 'PolySimplifierV5', 'PolyRegularizerV2', 'PolyRegularizerV3', 'PolySimplifierV6',
           # 'PolySimplifierV7', 'PolyRegularizerV4', 'PolySimplifierV8',
           'PolyRegularizerV5',
           # 'PolygonizerV1', 'PolySimplifierV9', 'PolygonizerV2', 'PolygonizerV3',
          #  'PolyRegularizerV6', 'PolySimplifierV10', 'PolySimplifierV11', 'PolygonizerV4',
          #  'PolyRegularizerV7', 'PolygonizerV5', 'PolyRegularizerV8', 'PolySimplifierV12',
          #  'PolyRegularizerV9', 'PolygonizerV6', 'PolySimplifierV13', 'PolySimplifierV14',
           # 'PolySimplifierV15', 'PolygonizerV7', 'PolySimplifierV17', 'PolygonizerV9',
           'PolygonizerV10',
           # 'DPPolygonizer'
          ]
