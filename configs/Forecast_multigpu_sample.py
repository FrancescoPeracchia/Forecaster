


modality = dict(
frame_sequence = [-4, -2, 0, 2, 3, 4],
target = [1,2,3]
)

predictor = dict (low = dict (kernel = (3,3,3),stride = (1,1,1), padding =(0,1,1),skip = True),
                    medium = dict (kernel = (3,3,3),stride = (1,1,1), padding =(0,1,1),skip = True),
                    high = dict (kernel = (3,3,3),stride = (1,1,1), padding =(0,1,1),skip = True),
                    huge = dict (kernel = (3,3,3),stride = (1,1,1), padding =(0,1,1),skip = True))


#_________________________________________________________________________
# model settings
model = dict(
    efficientPS_config = '/inference/efficientPS_cityscapes/config/efficientPS_multigpu_sample.py',
    efficientPS_checkpoint = '/inference/efficientPS_cityscapes/model/model.pth',
    multi_forecasting_modality = False,
    predictor_config = predictor,
    type='Forecaster'
    )

# model training and testing settings
train_cfg = dict()
test_cfg = dict()
#_________________________________________________________________________





#_________________________________________________________________________
img_norm_cfg = dict(
    mean=[106.433, 116.617, 119.559], std=[65.496, 67.6, 74.123], to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize_custom', img_scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=False)
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize_custom'),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


# dataset settings
dataset_type = 'RawDataset'
data_root = 'data/cityscapes/'

data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file ='/data/kitti_raw/train.json',
        data_root = '/data/kitti_raw/training',
        pipeline=train_pipeline),
    val=dict(),
    test=dict())

    
evaluation = dict(interval=1, metric=['panoptic'])
#_________________________________________________________________________



# optimizer
optimizer = dict(type='SGD', lr=0.07, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[120, 144])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 160
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
