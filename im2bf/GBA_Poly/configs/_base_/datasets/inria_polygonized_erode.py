# dataset settings
dataset_type = 'InriaPolygonizedDataset'
datapipe = 'inria_polygonized'
data_root = '../../Datasets/Dataset4EO/InriaPolygonized'
coco_ann_path = '../../Datasets/Dataset4EO/InriaPolygonized/ann_polygonized.json'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFileV3', imdecode_backend='tifffile', use_shp=True, to_float32=True,
         raster_shape=(1024, 1024), raster_downscale=1., collect_features=True),
    dict(type='ErodeGT', kernel_size=7),
    dict(type='Resize', img_scale=(1024, 1024)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'eroded_gt_semantic_seg'], cpu_keys=['features']),
    # dict(type='Collect', keys=['img'], cpu_keys=[], meta_keys=['filename']),
    # dict(type='Collect', keys=['img'], cpu_keys=[], meta_keys=['filename']),
]
test_pipeline = [
    dict(type='LoadImageFromFileV3', imdecode_backend='tifffile',
         use_shp=True, collect_features=True),
    # dict(type='LoadFeaturesV2'),
    # dict(type='Resize', img_scale=(1024, 1024)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        meta_keys=['filename', 'ori_shape', 'city_name', 'ann_path'],
        keys=['img', 'gt_semantic_seg'], cpu_keys=['features']),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        collect_keys=['img', 'seg', 'ann'],
        split='train',
        reduce_zero_label=False,
        gt_seg_map_loader_cfg=None,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        datapipe=datapipe,
        collect_keys=['img', 'seg', 'ann'],
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
        coco_ann_path = coco_ann_path,
        crop_size=[5000, 5000],
        stride=[5000, 5000],
        collect_keys=['img', 'seg', 'ann'],
        reduce_zero_label=False,
        split='test',
        # split='test_100',
        gt_seg_map_loader_cfg=None,
        gt_seg_map_loader_pipeline= [dict(
            type='LoadImageFromFileV3', imdecode_backend='tifffile', use_shp=False, to_float32=True,
            collect_features=True
        )],
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
