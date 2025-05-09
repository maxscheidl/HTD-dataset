# data pipeline

test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadTrackAnnotations')
        ]),
    dict(type='PackTrackInputs')
]

# dataloader

test_dataset_tpye = 'GotDataset'

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    # Now we support two ways to test, image_based and video_based
    # if you want to use video_based sampling, you can use as follows
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type=test_dataset_tpye,
        ann_file='data/got10k/annotations/got_coco_val.json',
        #ann_file='hard_tracks/hard_tracks_annotations_got.json',
        data_prefix=dict(img_path='data/got10k/'),
        test_mode=True,
        pipeline=test_pipeline
    ))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='GotTETAMetric',
    dataset_type=test_dataset_tpye,
    format_only=False,
    ann_file='data/got10k/annotations/got_coco_val.json',
    #ann_file='hard_tracks/hard_tracks_annotations_got.json',
    metric=['TETA'])
test_evaluator = val_evaluator


