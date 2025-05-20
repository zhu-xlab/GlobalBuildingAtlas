# dataset settings
dataset_type = 'PolyBuildingDatasetShape'
dataset_type_2 = 'GlobalBuildingRasterDataset'

datapipe = 'planet_building_shape'
datapipe_2 = 'global_building_raster'

data_root = '../../Datasets/Dataset4EO/PlanetBuildingV3/shapes'
in_root = '/home/Datasets/so2sat/planet_global_processing/Continents'
in_base_dir = 'EUROPE/glcv103_guf_wsf' # this prefix will be copied to the out_dir
out_root = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/polygon_outputs'

img_norm_cfg = dict(
    mean=[0.5, 0.5, 0.5], std=[1, 1, 1], to_rgb=True
)
crop_size = (256, 256)
train_pipeline = [
    dict(type='ParseShape', sample_type='all'),
    dict(type='RandomCropShape'),
    # dict(type='PolyFormatBundle'),
    dict(type='Collect', meta_keys=[], keys=[], cpu_keys=['gt_features']),
]
test_pipeline = [
    dict(type='LoadRasterWithWindow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='Collect', keys=['img']),
    dict(type='Collect', meta_keys=['filename', 'geo_transform', 'geo_crs', 'ori_shape', 'in_root', 'out_root', 'crop_boxes'],
         keys=[], cpu_keys=['img']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
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
        in_base_dir=in_base_dir,
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
        in_base_dir=in_base_dir,
        out_root=out_root,
        datapipe=datapipe_2,
        has_shape=False,
        patchify=True,
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
