# dataset settings
dataset_type = 'SRTrainingDataset'
datapipe = 'planet_building_paired_v2'
# data_root = '../../Datasets/Dataset4EO/sr_training'
data_root = '../../Datasets/Dataset4EO/Planet'
# window_dir = '../../Datasets/Dataset4EO/sr_training/*/data'
window_dir = '../../Datasets/Dataset4EO/sr_training/sr_training/*/data'
window_to_copy_dirs = ['../../Datasets/Dataset4EO/sr_training/sr_training/*/seg']
num_bands=4

img_norm_cfg = dict(
    # mean=[0.54845886, 0.57936498, 0.47839623, 0.51780941],
    mean = [628.28175185, 929.9814896, 1075.76097327, 2363.76032923],
    # std=[0.18698201, 0.17037536, 0.20319796, 0.19128074],
    std=[303.92523313, 319.16280294, 395.9882922, 662.51779089],
    to_rgb=False
)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFileV3', imdecode_backend='tifffile'),
    # dict(type='LoadFeatures'),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    # dict(type='Resize', img_scale=[(224, 224), (256, 256), (288, 288), (320, 320), (352, 352)]),
    # dict(type='Resize', img_scale=(320, 320)),
    dict(type='Resize', img_scale=(1024, 1024)),
    dict(type='RandomCrop', crop_size=crop_size),
    # dict(type='CreatePolyAnn', num_max_vertices=512),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='PolyFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], cpu_keys=[]),
]
test_pipeline = [
    dict(type='LoadImageFromFileV3', imdecode_backend='tifffile'),
    # dict(type='LoadFeaturesV2'),
    dict(type='Resize', img_scale=(256, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        # meta_keys=['filename', 'ori_shape'],
        keys=['img', 'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        window_root=window_dir,
        reduce_zero_label=False,
        split='train',
        gt_seg_map_loader_cfg=None,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        window_root=window_dir,
        reduce_zero_label=False,
        split='val',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        collect_keys=['img', 'seg', 'ann'],
        window_root = '../../Datasets/Dataset4EO/sr_training/sr_testing/data',
        window_to_copy_dirs = ['../../Datasets/Dataset4EO/sr_training/sr_testing/seg'],
        num_bands=num_bands,
        reduce_zero_label=False,
        # split='test_1k',
        split='test',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline),
    train_dataloader=dict(
        persistent_workers=False),
    val_dataloader=dict(
        persistent_workers=False),
    test_dataloader=dict(
        persistent_workers=False),
)
