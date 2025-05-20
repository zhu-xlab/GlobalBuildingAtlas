# dataset settings
dataset_type = 'CrowdAIDataset'
datapipe = 'crowd_ai'
data_root = '../../Datasets/Dataset4EO/CrowdAI'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFileV2'),
    dict(type='LoadFeatures'),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    # dict(type='Resize', img_scale=[(224, 224), (256, 256), (288, 288), (320, 320), (352, 352)]),
    # dict(type='Resize', img_scale=(320, 320)),
    dict(type='Resize', img_scale=(224, 224)),
    # dict(type='RandomCrop', crop_size=crop_size),
    # dict(type='CreatePolyAnn', num_max_vertices=512),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='PolyFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], cpu_keys=['features']),
]
test_pipeline = [
    dict(type='LoadImageFromFileV2'),
    dict(type='Resize', img_scale=(320, 320)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', meta_keys=['filename', 'ori_shape'], keys=['img']),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        reduce_zero_label=True,
        split='train_small',
        gt_seg_map_loader_cfg=None,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        reduce_zero_label=True,
        split='val_small',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        reduce_zero_label=True,
        split='val_small',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline),
    train_dataloader=dict(
        persistent_workers=False),
    val_dataloader=dict(
        persistent_workers=False)
)
