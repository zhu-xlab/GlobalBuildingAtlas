# dataset settings
dataset_type = 'PolyBuildingDatasetShape'
dataset_type_2 = 'PolyBuildingDatasetRasterV2'

datapipe = 'planet_building_shape'
datapipe_2 = 'planet_building_raster'

# data_root = '../../Datasets/Dataset4EO/PlanetBuildingV3/shapes'
data_root = '../../Datasets/Dataset4EO/PlanetBuildingV3/samples'
# data_root_2 = '../../Datasets/Dataset4EO/PlanetBuildingV3/raster_samples'
# data_root_2 = '../../Datasets/Dataset4EO/Planet/sr_4bands_v2'
data_root_2 = '../../Datasets/Dataset4EO/Planet/samples'
# data_root_2 = './outputs/regu-v4_29-cities'

img_norm_cfg = dict(
    mean=[799.71214243, 959.26968473, 1003.91903734, 2170.10086864],
    std=[447.39048165, 452.96305473, 547.40357583, 853.5349271],
    to_rgb=False
)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadShape', sample_type='random', to_pixel_coords=True, gsd_range=[0.5, 1], min_perimeter=None),
    # dict(type='RandomCropShapeV2', gsd=0.3),
    # dict(type='PolyFormatBundle'),
    dict(type='Collect', meta_keys=[], keys=[], cpu_keys=['features']),
]

test_pipeline = [
    # dict(type='LoadRasterFromFile'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='ImageToTensor', keys=['img']),
    # dict(type='Collect', keys=[], cpu_keys=['img']),

    dict(type='LoadImageFromFileV3', imdecode_backend='tifffile', collect_transform=True),
    # dict(type='Resize', img_scale=(256, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        meta_keys=['filename', 'geo_transform', 'ori_shape', 'city_name', 'continent_name', 'crs', 'seg_path'],
        keys=['img', 'gt_semantic_seg', 'majority_voting']
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
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
        # has_shape=False,
        reduce_zero_label=True,
        split='train',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type_2,
        data_root=data_root_2,
        datapipe=datapipe_2,
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
