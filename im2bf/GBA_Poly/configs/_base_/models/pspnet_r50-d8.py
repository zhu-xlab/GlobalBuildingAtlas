# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    #pretrained='/home/xshadow/RSI-Segmentation/pretrained/resnet50_v1c-sp.pth',
    #pretrained='/home/xshadow/RSI-Segmentation/pretrained/resnet_50_sp2.pth',
    #pretrained='/home/xshadow/RSI-Segmentation/pretrained/resnet50_v1c_layer.pth',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        in_channels=14, # newly add
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,class_weight=[0.6, 1.180])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[0.6, 1.180])),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
