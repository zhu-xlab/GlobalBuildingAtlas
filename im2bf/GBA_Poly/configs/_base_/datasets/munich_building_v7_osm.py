# dataset settings
dataset_type = 'PolyBuildingDataset'
datapipe = 'planet_building'
data_root = '../../Datasets/Dataset4EO/PlanetBuilding'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='LoadPolyAnn'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.6, 1.4)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='CreateGTPointsFromPolygons'),
    dict(type='CreateContours'),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='PolyFormatBundle'),
    dict(type='Collect', keys=['img', 'mask'],
         cpu_keys=['contours', 'gt_points', 'polygons', 'contour_labels', 'gt_degrees']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomCrop', crop_size=crop_size),
            # dict(type='RandomFlip'),
            dict(type='CreateContours'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'contours', 'contour_labels']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        city_names=['munich'],
        img_type='planet_SR',
        mask_type='osm_3m',
        poly_type='osm_polygon',
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
        img_type='planet_SR',
        # mask_type='microsoft_3m',
        mask_type='building_footprint',
        poly_type='osm_polygon',
        reduce_zero_label=True,
        split='train',
        gt_seg_map_loader_cfg=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        city_names=['munich'],
        img_type='planet_SR',
        # mask_type='microsoft_3m',
        mask_type='building_footprint',
        poly_type='osm_polygon',
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
