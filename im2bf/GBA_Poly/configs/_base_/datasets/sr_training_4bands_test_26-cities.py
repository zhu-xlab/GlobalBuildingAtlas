# dataset settings
dataset_type = 'SRTrainingDataset'
datapipe = 'planet_building_paired_v2'
data_root = '../../Datasets/Dataset4EO/Planet'
# window_dir = '../../Datasets/Dataset4EO/sr_training/sr_training/*/data'
window_dir = None
num_bands=4

img_norm_cfg = dict(
    mean=[799.71214243, 959.26968473, 1003.91903734, 2170.10086864],
    std=[447.39048165, 452.96305473, 547.40357583, 853.5349271],
    to_rgb=False
)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFileV3', imdecode_backend='tifffile', to_float32=True),
    dict(type='Resize', img_scale=(1024, 1024)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PolyFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], cpu_keys=[]),
    # dict(type='Collect', keys=['img'], cpu_keys=[], meta_keys=['filename']),
    # dict(type='Collect', keys=['img'], cpu_keys=[], meta_keys=['filename']),
]
test_pipeline = [
    dict(type='LoadImageFromFileV3', imdecode_backend='tifffile', collect_transform=True),
    # dict(type='LoadFeaturesV2'),
    # dict(type='Resize', img_scale=(256, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        # meta_keys=['filename'],
        meta_keys=['filename', 'ori_shape', 'city_name', 'geo_transform', 'crs'],
        keys=['img']),
        # keys=['gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        collect_keys=['img', 'seg', 'ann'],
        window_root = '../../Datasets/Dataset4EO/sr_training/sr_training/*/data',
        window_to_copy_dirs = ['../../Datasets/Dataset4EO/sr_training/sr_training/*/seg'],
        split='train',
        # window_root = '../../Datasets/Dataset4EO/sr_training/sr_test/data',
        # window_to_copy_dirs = ['../../Datasets/Dataset4EO/sr_training/sr_test/seg'],
        # split='test',
        num_bands=num_bands,
        reduce_zero_label=False,
        gt_seg_map_loader_cfg=None,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        collect_keys=['img', 'seg'],
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
    test=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        collect_keys=['img'],
        ignore_shp=True,
        window_root = None,
        window_to_copy_dirs = [],
        crop_size=[9600 * 2, 9600 * 2],
        stride=[9600 * 2, 9600 * 2],
        num_bands=num_bands,
        reduce_zero_label=False,
        split='test',
        # split='test_100',
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
