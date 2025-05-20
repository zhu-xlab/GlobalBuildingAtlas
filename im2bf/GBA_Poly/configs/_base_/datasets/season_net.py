# dataset settings
dataset_type = 'EODataset'
datapipe = 'season_net'
datapipe_cfg = {'season': 'spring'}
data_root = '../../Datasets/Dataset4EO/SeasonNet'
# data_root = '../data/Dataset4EO/SeasonNet'
reduce_zero_label = True
gt_seg_map_loader_cfg=dict(imdecode_backend = 'tifffile',
                           reduce_zero_label = reduce_zero_label)

img_norm_cfg = dict(
    mean=[817.83099309,817.90637517,613.89910777], std=[1152.3451639,1081.4451218,1107.54732507], to_rgb=True)


crop_size = (128, 128)
train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='LoadAnnotations', reduce_zero_label=reduce_zero_label, imdecode_backend='tifffile'),
    # dict(type='PercentileNormalize', **img_norm_cfg),
    dict(type='ClipNormalize', **img_norm_cfg),
    # dict(type='MultiDomainClipNormalize', norm_dict=img_norm_cfg, key_name='season'),
    dict(type='Resize', img_scale=(120, 120), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomRotate90', prob=1.0),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    # dict(type='PercentileNormalize', **img_norm_cfg),
    dict(type='ClipNormalize', **img_norm_cfg),
    # dict(type='MultiDomainClipNormalize', norm_dict=img_norm_cfg, key_name='season'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(128, 128),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=32,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        reduce_zero_label=reduce_zero_label,
        split='train',
        gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        reduce_zero_label=reduce_zero_label,
        split='val_10k',
        # datapipe_cfg = {'season': 'spring'},
        gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        reduce_zero_label=reduce_zero_label,
        # split='val_10k',
        split='test',
        gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
        pipeline=test_pipeline),
    train_dataloader=dict(
        persistent_workers=True),
    val_dataloader=dict(
        persistent_workers=True)
)
