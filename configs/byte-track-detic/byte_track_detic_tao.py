_base_ = [
    '../../projects/Detic_new/configs/detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis-masa.py',
    '../datasets/tao/tao_dataset_v1.py',
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
    type='ByteTrack',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    detector=detector,
    tracker=dict(
        type='ByteTracker',
        motion=dict(type='KalmanFilter'),
        obj_score_thrs=dict(high=0.1, low=0.0001),
        init_track_thr=0.0001, # original was 0.7
        weight_iou_with_det_scores=False, # init was true
        match_iou_thrs=dict(high=0.3, low=0.7, tentative=0.5), # is senetive to fps
        num_frames_retain=30))


train_dataloader = None
train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='MasaTrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook')
]
auto_scale_lr = dict(enable=False, base_batch_size=16)
val_dataloader = dict(
    dataset=dict(
        ann_file='data/tao/annotations/tao_val_lvis_v1_classes.json',
    )
)
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file='data/tao/annotations/tao_val_lvis_v1_classes.json',
    outfile_prefix='results/byte-track-detic-tao',
)
test_evaluator = val_evaluator

