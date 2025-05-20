_base_ = [
    '../_base_/models/upernet_convnext.py',
    '../_base_/datasets/global_building_raster_test_samples.py',
    '../_base_/default_runtime.py', '../_base_/schedules/adamw_80k.py'
]
crop_size = (1024, 1024)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa
wandb_cfg=dict(
    init_kwargs=dict(
        project = 'GBA',
        entity = 'tum-tanmlh',
        name = 'polygonizer',
        resume = 'never',
        dir = 'work_dirs/'
    ),
    interval=200,
    scalar_interval=50,
    num_eval_images=0
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
# load_from = ('work_dirs/poly-dqn-v1_pretrain_20k_munich-microsoft/iter_20000.pth')
model = dict(
    type='PolygonizerV10',
    init_cfg=dict(
        regu_checkpoint='work_dirs/regu_upernet_convnext-t_80k/iter_80000.pth',
    ),
    ring_sample_conf=dict(
        interval=1.5, length=50, num_max_ring=128, noise_type='random',
        num_angles=32, max_offset=8, ring_stride=30
    ),
    regu_net = dict(
        type='PolyRegularizerV5',
        backbone=dict(
            type='mmcls.ConvNeXt',
            arch='tiny',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
        ),
        decode_head=dict(
            type='UPerHead',
            in_channels=[96, 192, 384, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            dropout_ratio=0.1,
            num_classes=2,
            channels=512,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=[
                dict(type='CrossEntropyLoss', class_weight=[1,1], use_sigmoid=False, loss_weight=1.0),
                dict(type='DiceLoss')
            ]
        ),
        auxiliary_head=dict(
            type='FCNHead',
            in_channels=384,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', class_weight=[1,1], use_sigmoid=False, loss_weight=0.0
            ),
        ),
        test_cfg=dict(
            mode='slide', crop_size=crop_size,
            stride=(992, 992), num_max_test_nodes=512,
            prob_thre=0.0, nms_width=11, scale_factor=4,
            upscale_output=False, filter_existed=True
        ),
        train_cfg=dict(mask_graph=True, num_min_proposals=512, crop_size=crop_size, disc_step=-1,
                       all_touched=False),
        ring_sample_conf=dict(interval=1.5, length=50, num_max_ring=128, noise_type='random')
    ),
    test_cfg=dict(
        mode='slide', crop_size=crop_size,
        # stride=(1024, 1024), num_max_test_nodes=512,
        stride=(992, 992), num_max_test_nodes=512,
        prob_thre=0.0, nms_width=11, scale_factor=4,
        upscale_output=False, filter_existed=True,
        save_regu_results=False
    ),
    train_cfg=dict(mask_graph=True, num_min_proposals=512, crop_size=crop_size, disc_step=-1,
                   all_touched=False),
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbVisualizer', wandb_cfg=wandb_cfg)
    ])

evaluation = dict(interval=8000, skip_evaluate=True, metric=None, pre_eval=False)
checkpoint_config = dict(by_epoch=False, interval=8000)
