# yapf:disable
init_kwargs = {
    'project': 'rsi-segmentation',
    'entity': 'tum-tanmlh',
}
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='MMSegWandbHook',
        #      init_kwargs={'project': 'rsi_poly'},
        #      interval=201,
        #      num_eval_images=10)
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
