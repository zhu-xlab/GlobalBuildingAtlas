# dataset settings
dataset_type = 'PolyBuildingDatasetShape'
dataset_type_2 = 'GlobalBuildingRasterDataset'

datapipe = 'planet_building_shape'
datapipe_2 = 'global_building_raster'

data_root = '../../Datasets/Dataset4EO/PlanetBuildingV3/shapes'
in_root = '/home/fahong/Datasets/so2sat/parallelTest/'
in_base_dirs = ['africa/mosaic']
out_root = '/home/fahong/Datasets/ai4eo/Dataset4EO/GlobalBFV2/mask'

img_norm_cfg = dict(
    mean=[799.71214243, 959.26968473, 1003.91903734, 2170.10086864],
    std=[447.39048165, 452.96305473, 547.40357583, 853.5349271],
    to_rgb=False
)
crop_size = (256, 256)
train_pipeline = [
    dict(type='ParseShape', sample_type='all'),
    dict(type='RandomCropShape'),
    # dict(type='PolyFormatBundle'),
    dict(type='Collect', meta_keys=[], keys=[], cpu_keys=['gt_features']),
]
test_pipeline = [
    dict(type='LoadRasterWithWindow', binarize=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='Collect', keys=['img']),
    dict(type='Collect', meta_keys=['filename', 'geo_transform', 'geo_crs', 'ori_shape', 'in_root',
                                    'out_root', 'crop_boxes', 'out_path', 'img_norm_cfg'],
         keys=['img'], cpu_keys=[]),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        reduce_zero_label=True,
        split='train',
        gt_seg_map_loader_cfg=None,
        pipeline=train_pipeline,
        has_shape=True
    ),
    val=dict(
        type=dataset_type_2,
        data_root=in_root,
        in_base_dirs=in_base_dirs,
        out_root=out_root,
        datapipe=datapipe_2,
        has_shape=False,
        upscale = 1,
        pixel_width=3.,
        pixel_height=3.,
        thre=0.5,
        reduce_zero_label=True,
        split='train',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type_2,
        data_root=in_root,
        in_base_dirs=in_base_dirs,
        out_root=out_root,
        datapipe=datapipe_2,
        has_shape=False,
        patchify=True,
        filter_post_fix='sr_mosaic_tile.tif',
        crop_size=(10000, 10000),
        upscale = 1,
        pixel_width=3.,
        pixel_height=3.,
        thre=0.5,
        reduce_zero_label=True,
        split='train',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline,
    ),
    train_dataloader=dict(
        persistent_workers=False),
    val_dataloader=dict(
        persistent_workers=False),
    test_dataloader=dict(
        persistent_workers=False)
)
