# dataset settings
dataset_type = 'SRTrainingDataset'
datapipe = 'planet_building_paired_v2'
data_root = '../../Datasets/Dataset4EO/Planet'
window_dir = '../../Datasets/Dataset4EO/sr_training/sr_training/*/data'
num_bands=8

img_norm_cfg = dict(
    mean=[638.07703902, 718.62986311, 832.99469179, 932.22725665, 997.31072304, 993.89744732,
          1244.5647796, 2290.81339935, 4.43860973],
    std=[386.96097664, 425.03383269, 438.72453041, 466.3736261, 543.3482596, 576.61652676,
         544.68372881, 888.55098706, 5.1230093],
    to_rgb=False
)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFileV3', imdecode_backend='tifffile'),
    dict(type='Resize', img_scale=(1024, 1024)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PolyFormatBundle'),
    dict(
        type='Collect', keys=['img', 'gt_semantic_seg', 'ndsm'], cpu_keys=[],
        meta_keys=(
             'filename', 'ori_filename', 'ori_shape',
             'img_shape', 'pad_shape', 'scale_factor', 'flip',
             'flip_direction', 'img_norm_cfg', 'use_ndsm'
        ),
    ),
]
test_pipeline = [
    dict(type='LoadImageFromFileV3', imdecode_backend='tifffile'),
    dict(type='Resize', img_scale=(256, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        # meta_keys=['filename', 'ori_shape'],
        meta_keys=['filename', 'ori_shape', 'city_name', 'continent_name'],
        keys=['img', 'gt_semantic_seg', 'ndsm']
    )
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,

        # new train
        num_bands=num_bands,
        window_root = ['../../Datasets/Dataset4EO/sr_training/sr_training/*/data'],
        window_to_copy_dirs = [
            ['../../Datasets/Dataset4EO/sr_training/sr_training/*/seg', {255:0, 0:1}],
           '../../Datasets/Dataset4EO/Planet/pred_ndsm_v2/sr_training/*/ndsm',
        ],
        split='train',
        collect_keys = ['8bands_img', 'seg', 'ndsm'],
        ignore_shp = True,

        reduce_zero_label=False,
        gt_seg_map_loader_cfg=None,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        collect_keys=['img', 'seg', 'ann', 'ndsm'],
        window_root = '../../Datasets/Dataset4EO/sr_training/sr_test/data',
        window_to_copy_dirs = ['../../Datasets/Dataset4EO/sr_training/sr_test/seg'],
        num_bands=num_bands,
        reduce_zero_label=False,
        # split='test_1k',
        split='test',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        collect_keys=['img', 'seg', 'ndsm'],
        ignore_shp=True,
        window_root = '../../Datasets/Dataset4EO/sr_training/sr_test/data',
        window_to_copy_dirs = ['../../Datasets/Dataset4EO/sr_training/sr_test/seg'],
        num_bands=num_bands,
        reduce_zero_label=False,
        # split='test_1k',
        split='test',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline
    ),
    train_dataloader=dict(
        persistent_workers=False,
        pin_memory=False
    ),
    val_dataloader=dict(
        persistent_workers=False,
        pin_memory=False
    ),
    test_dataloader=dict(
        persistent_workers=False,
        pin_memory=False
    ),
)
