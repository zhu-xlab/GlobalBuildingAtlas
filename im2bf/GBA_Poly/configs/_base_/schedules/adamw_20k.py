# optimizer
# runtime settings
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[10000])

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(interval=5000)
evaluation = dict(interval=5000, metric='mIoU', pre_eval=True)
