# dataset settings
dataset_type = 'PolyBuildingDatasetShape'
# dataset_type_2 = 'PolyBuildingDatasetV2'
dataset_type_2 = 'PolyBuildingDatasetShape'

datapipe = 'planet_building_shape'
# datapipe_2 = 'planet_building_paired'
datapipe_2 = 'planet_building_raster'

# data_root = '../../Datasets/Dataset4EO/PlanetBuildingV3/shapes'
data_root = '../../Datasets/Dataset4EO/PlanetBuildingV3/samples'
# data_root_2 = '../../Datasets/Dataset4EO/PlanetBuildingV2'
data_root_2 = '../../Datasets/Dataset4EO/PlanetBuildingV3/raster_samples'
# data_root_2 = './outputs/regu-v4_29-cities'

img_norm_cfg = dict(
    mean=[0.5, 0.5, 0.5], std=[1, 1, 1], to_rgb=True
)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadShape', sample_type='random', to_pixel_coords=True, gsd_range=[2., 4.]),
    # dict(type='RandomCropShapeV2', gsd=0.3),
    # dict(type='PolyFormatBundle'),
    dict(type='Collect', meta_keys=[], keys=[], cpu_keys=['features']),
]

test_pipeline = [
    dict(type='LoadRasterFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='Collect', keys=['img']),
    dict(type='Collect', keys=[], cpu_keys=['img']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        has_shape=True,
        reduce_zero_label=True,
        split='train',
        gt_seg_map_loader_cfg=None,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type_2,
        data_root=data_root_2,
        datapipe=datapipe_2,
        has_shape=False,
        reduce_zero_label=True,
        split='train',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type_2,
        data_root=data_root_2,
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
