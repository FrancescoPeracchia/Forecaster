'''
modality dict
'''


modality = dict(  frame_sequence = [-5, -3, 0, 3, 4, 5],
                target = [1,2,3]

)



'''
model predictor variants 
'''
#first
F2F_3DCONV = dict (  low = dict (kernel = (3,5,5),stride = (1,1,1), padding =(0,2,2),skip = True),
                    medium = dict (kernel = (3,3,3),stride = (1,1,1), padding =(0,1,1),skip = True),
                    high = dict (kernel = (3,3,3),stride = (1,1,1), padding =(0,1,1),skip = True),
                    huge = dict (kernel = (3,3,3),stride = (1,1,1), padding =(0,1,1),skip = True))

pre_trained = '/home/fperacch/Forecaster/saved/model_predictor.pth',
#change accordigly the desired predictor_config = dict (module_predictor,pre_trained weights),


#_________________________________________________________________________
# model settings
model = dict(
    efficientPS_config = '/inference/efficientPS_cityscapes/config/efficientPS_multigpu_sample.py',
    efficientPS_checkpoint = '/inference/efficientPS_cityscapes/model/model.pth',
    multi_forecasting_modality = False,
    predictor_config = dict (model = F2F_3DCONV, weights = pre_trained),
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
    dict(type='Resize_custom', img_scale=(1024, 512), ratio_range=(0.5, 2.0), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32)
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
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
        ann_file ='/data/KITTI/train.json',
        data_root = '/data/KITTI/training',
        pipeline=train_pipeline),
    validation=dict(
        type=dataset_type,
        ann_file ='/data/KITTI/validation.json',
        data_root = '/data/KITTI/validation',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file ='/data/KITTI/test.json',
        data_root = '/data/KITTI/test',
        pipeline=train_pipeline))

    
evaluation = dict(interval=1, metric=['panoptic'])
#_________________________________________________________________________




