# dataset settings
dataset_type = 'PolyBuildingDatasetV2'
datapipe = 'planet_building_paired'
data_root = '../../Datasets/Dataset4EO/PlanetBuildingV2'
img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    mean=[0.5, 0.5, 0.5], std=[1, 1, 1], to_rgb=True
)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile', use_mask_as_img=False, combine_mask_with_img=False),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='LoadPolyAnnV2'),
    # dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.6, 1.4)),
    dict(type='Resize', img_scale=(1024, 1024), img_interpolation_type='nearest', keep_gt_semantic_seg_size=True),
    dict(type='Register'),
    # dict(type='RandomCrop', crop_size=crop_size),
    # dict(type='CreateGTPointsFromPolygons'),
    dict(type='CreateContours'),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='PolyFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'],
         cpu_keys=['contours', 'gdal_features', 'gt_features', 'contour_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', use_mask_as_img=False, combine_mask_with_img=False),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1024, 1024),
        img_scale=(5742, 6383),
        # img_scale=(1024 * 4, 1024 * 4),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, img_interpolation_type='nearest'),
            # dict(type='RandomCrop', crop_size=crop_size),
            # dict(type='LoadPolyAnn'),
            # dict(type='RandomFlip'),
            dict(type='CreateContours'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'contours', 'contour_labels', 'comp_mask']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        city_names=['munich'],
        img_type='building_footprint',
        mask_type='osm_075m',
        mask_scale=4,
        gdal_poly_type='gdal_polygon_microsoft3m',
        gt_poly_type='microsoft_polygon',
        reduce_zero_label=True,
        split='train',
        gt_seg_map_loader_cfg=None,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        city_names=['munich'],
        img_type='osm_3m',
        mask_type='osm_075m',
        gdal_poly_type='gdal_polygon_microsoft3m',
        gt_poly_type='microsoft_polygon',
        reduce_zero_label=True,
        split='train',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        city_names=['munich'],
        img_type='building_footprint',
        mask_type='osm_075m',
        # mask_scale=4,
        gdal_poly_type='gdal_polygon_microsoft3m',
        gt_poly_type='microsoft_polygon',
        # crop_size=[1024,1024],
        # stride=[1024,1024],
        crop_size=[-1,-1],
        stride=[-1,-1],
        upscale = 1,
        pixel_width=0.75,
        pixel_height=0.75,
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
