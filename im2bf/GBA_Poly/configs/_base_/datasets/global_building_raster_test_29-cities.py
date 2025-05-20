# dataset settings
dataset_type = 'PolyBuildingDatasetShape'
dataset_type_2 = 'PolyBuildingDatasetShape'

datapipe = 'planet_building_shape'
datapipe_2 = 'global_building_raster'

data_root = '../../Datasets/Dataset4EO/PlanetBuildingV3/shapes'
# data_root = '../../Datasets/Dataset4EO/PlanetBuildingV3/samples'
# data_root_2 = '../../Datasets/Dataset4EO/PlanetBuildingV3/raster_samples_2'
in_root = '../../Datasets/Dataset4EO/PlanetBuildingV3'
in_base_dir = 'raster_samples_2'
out_root = './outputs/polygonizer_v1'

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
    dict(type='LoadRasterFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='Collect', keys=['img']),
    dict(
        type='Collect', meta_keys=['filename', 'geo_transform', 'geo_crs', 'ori_shape', 'in_root', 'out_root'],
        keys=[], cpu_keys=['img']
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
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
