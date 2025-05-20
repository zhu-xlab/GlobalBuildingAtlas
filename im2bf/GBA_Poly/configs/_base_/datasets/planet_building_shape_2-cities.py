# dataset settings
dataset_type = 'PolyBuildingDatasetShape'
dataset_type_2 = 'PolyBuildingDatasetV2'

datapipe = 'planet_building_shape'
datapipe_2 = 'planet_building_paired'

data_root = '../../Datasets/Dataset4EO/PlanetBuildingV3/samples'
data_root_2 = '../../Datasets/Dataset4EO/PlanetBuildingV2'
img_norm_cfg = dict(
    mean=[0.5, 0.5, 0.5], std=[1, 1, 1], to_rgb=True
)
crop_size = (256, 256)
train_pipeline = [
    dict(type='ParseShape'),
    # dict(type='PolyFormatBundle'),
    dict(type='Collect', meta_keys=[], keys=[], cpu_keys=['gt_features']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', use_mask_as_img=False, combine_mask_with_img=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(5742, 6383),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, img_interpolation_type='nearest'),
            dict(type='CreateContours'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'contours', 'contour_labels', 'comp_mask']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        reduce_zero_label=True,
        split='train',
        gt_seg_map_loader_cfg=None,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type_2,
        data_root=data_root_2,
        datapipe=datapipe_2,
        city_names=['munich'],
        img_type='osm_3m',
        mask_type='osm_075m',
        gdal_poly_type='gdal_polygon',
        gt_poly_type='osm_polygon',
        reduce_zero_label=True,
        split='train',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type_2,
        data_root=data_root_2,
        datapipe=datapipe_2,
        city_names=['munich'],
        # img_type='building_footprint',
        img_type='building_footprint_regu-v3',
        mask_type='osm_075m',
        # mask_scale=4,
        gdal_poly_type='gdal_polygon',
        gt_poly_type='osm_polygon',
        crop_size=[-1,-1],
        stride=[-1,-1],
        # crop_size=[1024 * 4, 1024 * 4],
        # stride=[1024, 1024],
        upscale = 1,
        pixel_width=3.,
        pixel_height=3.,
        thre=0.0,
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
