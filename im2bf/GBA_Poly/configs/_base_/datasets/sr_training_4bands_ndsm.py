# dataset settings
dataset_type = 'SRTrainingDataset'
datapipe = 'planet_building_paired_v2'
data_root = '../../Datasets/Dataset4EO/Planet'
window_dir = '../../Datasets/Dataset4EO/sr_training/sr_training/*/data'
num_bands=4

img_norm_cfg = dict(
    mean=[799.71214243, 959.26968473, 1003.91903734, 2170.10086864, 4.43860973],
    std=[447.39048165, 452.96305473, 547.40357583, 853.5349271, 5.1230093],
    to_rgb=False
)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFileV3', imdecode_backend='tifffile'),
    dict(type='Resize', img_scale=(1024, 1024)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        # type='Collect', keys=['img', 'gt_semantic_seg'], cpu_keys=[],
        type='Collect', keys=['img', 'gt_semantic_seg'], cpu_keys=[],
        meta_keys=(
            'filename', 'ori_filename', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction', 'img_norm_cfg', 'use_ndsm'
        ),
    ),
]
test_pipeline = [
    dict(type='LoadImageFromFileV3', imdecode_backend='tifffile'),
    # dict(type='LoadFeaturesV2'),
    dict(type='Resize', img_scale=(256, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        meta_keys=['filename', 'ori_shape', 'city_name', 'continent_name'],
        keys=['img', 'gt_semantic_seg'],
    ),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
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
        crop_size=[256, 256],
        stride=[192, 192],
        collect_keys = ['4bands_img', 'seg', 'ndsm'],
        ignore_shp = True,

        reduce_zero_label=False,
        gt_seg_map_loader_cfg=None,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,

        num_bands=num_bands,
        window_root = ['../../Datasets/Dataset4EO/sr_training/sr_test/data',
                      '../../Datasets/Dataset4EO/SiningPlanet/bf_test_data/data'],
        window_to_copy_dirs = [
            ['../../Datasets/Dataset4EO/sr_training/sr_test/seg', {255:0, 0:1}],
            '../../Datasets/Dataset4EO/SiningPlanet/bf_test_data/seg',
            '../../Datasets/Dataset4EO/SiningPlanet/ndsm',
            '../../Datasets/Dataset4EO/Planet/pred_ndsm/sr_test/ndsm',
            # '../../Datasets/Dataset4EO/Planet/pred_ndsm/sr_test/ndsm',
        ],
        split='test',
        collect_keys = ['4bands_img', 'seg', 'ndsm'],
        ignore_shp = True,
        additional_raster_resource_names = None,

        reduce_zero_label=False,
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,

        num_bands=num_bands,
        window_root = [
            '../../Datasets/Dataset4EO/sr_training/sr_test/data',
            '../../Datasets/Dataset4EO/Planet/sr_test_gt_v2/*/*',
        ],
        window_to_copy_dirs = [
            '../../Datasets/Dataset4EO/Planet/sr_test_gt_v2/*/seg',
            '../../Datasets/Dataset4EO/Planet/pred_ndsm_v2/sr_test/ndsm',
            '../../Datasets/Dataset4EO/Planet/pred_ndsm_v2/extra_test_filtered/ndsm',
        ],
        split='test',
        collect_keys = ['4bands_img', 'seg', 'ndsm', 'majority_voting'],
        # collect_keys = [f'4bands_img', 'seg', 'ndsm', 'majority_voting'],
        ignore_shp = True,
        additional_raster_resource_names = [('majority_voting', {255:1, 0:0})],

        reduce_zero_label=False,
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline
    ),
    train_dataloader=dict(
        persistent_workers=False,
        pin_memory=True
    ),
    val_dataloader=dict(
        persistent_workers=False,
        pin_memory=True
    ),
    test_dataloader=dict(
        persistent_workers=False,
        pin_memory=True
    ),
)
