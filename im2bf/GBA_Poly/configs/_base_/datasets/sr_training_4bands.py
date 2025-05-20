# dataset settings
dataset_type = 'SRTrainingDataset'
datapipe = 'planet_building_paired_v2'
data_root = '../../Datasets/Dataset4EO/Planet'
window_dir = '../../Datasets/Dataset4EO/sr_training/sr_training/*/data'
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
    dict(type='LoadImageFromFileV3', imdecode_backend='tifffile'),
    # dict(type='LoadFeaturesV2'),
    dict(type='Resize', img_scale=(256, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        meta_keys=['filename', 'ori_shape', 'city_name', 'continent_name'],
        keys=['img', 'gt_semantic_seg', 'majority_voting']),
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
        collect_keys = ['4bands_img', 'seg', 'majority_voting'],
        # collect_keys = [f'4bands_img', 'seg', 'ndsm', 'majority_voting'],
        ignore_shp = True,
        additional_raster_resource_names = [('majority_voting', {255:1, 0:0})],

        reduce_zero_label=False,
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

