_base_ = [
    '../../projects/Detic_new/configs/detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis-masa.py',
    '../datasets/got/got_dataset.py',
    '../default_runtime.py'
]

default_scope = 'mmdet'
detector = _base_.model
detector.pop('data_preprocessor')
detector['init_cfg'] = dict(
    type='Pretrained',
    checkpoint= 'saved_models/pretrain_weights/detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis-ec91245d.pth'
    # noqa: E501
)
detector['type'] = 'DeticMasa'
del _base_.model

model = dict(
    type='DeepSORT',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    detector=detector,
    reid=dict(
        type='BaseReID',
        data_preprocessor=dict(type='mmpretrain.ClsDataPreprocessor'),
        backbone=dict(
            type='mmpretrain.ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=380,
            loss_cls=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
            loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
        )),
    tracker=dict(
        type='SORTTracker',
        motion=dict(type='KalmanFilter', center_only=False),
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
        num_frames_retain=100))

train_dataloader = None
train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook')
]



val_dataloader = dict(
    dataset=dict(
        ann_file='data/got10k/annotations/got_coco_val.json',
    )
)
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file='data/got10k/annotations/got_coco_val.json',
    outfile_prefix='results/deep_sort_detic_got',
)
test_evaluator = val_evaluator
