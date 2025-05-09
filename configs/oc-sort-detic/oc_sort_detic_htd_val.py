_base_ = [
    '../../projects/Detic_new/configs/detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis-masa.py',
    '../datasets/hard_tracks/hard_tracks_dataset.py',
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
    type='OCSORT',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    detector=detector,
    tracker=dict(
        type='OCSORTTracker',
        motion=dict(type='KalmanFilter'),
        obj_score_thr=0.0001,
        init_track_thr=0.0001,
        weight_iou_with_det_scores=True,
        match_iou_thr=0.5,
        num_tentatives=3,
        vel_consist_weight=0.2,
        vel_delta_t=3,
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
        ann_file='data/htd/annotations/hard_tracks_dataset_coco_val.json',
        data_prefix=dict(img_path='data/htd/data'),
    )
)
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file='data/htd/annotations/hard_tracks_dataset_coco_val.json',
    outfile_prefix='results/oc_sort_detic_htd_val',
)
test_evaluator = val_evaluator



