import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_json(json_path):
    """ Load JSON file """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compute_statistics(track_id, dataset_name, video_id, video_name, track_annotations, frame_indices, video_length, video_frame_range, video_dims, cat_name):
    """Compute statistics for a given track"""

    video_length = video_length  # Total frames in video
    v_width, v_height = video_dims
    v_diagonal = np.sqrt(v_width ** 2 + v_height ** 2)
    v_area = v_width * v_height

    # Extract bounding boxes
    bboxes = np.array([ann['bbox'] for ann in track_annotations])

    # Compute occlusion statistics (track appears/disappears) --> if the object is occluded in the first frame, it is not occluded
    frame_indices = np.sort(np.array(frame_indices))
    normalized_frame_indices = frame_indices / video_frame_range # Now the increments are 1
    frame_gaps = np.diff(normalized_frame_indices)  # Differences between consecutive frame indices
    number_of_occlusions = np.sum(frame_gaps > 1)  # If the frame gap > 1, it's an occlusion

    occlusion_lengths = frame_gaps[frame_gaps > 1] - 1  # Length of occlusions
    avg_occlusion_length = np.mean(occlusion_lengths) if len(occlusion_lengths) > 0 else 0
    max_occlusion_length = np.max(occlusion_lengths) if len(occlusion_lengths) > 0 else 0
    min_occlusion_length = np.min(occlusion_lengths) if len(occlusion_lengths) > 0 else 0

    # Compute track length
    track_length_visible = len(track_annotations)  # Number of frames where the object is visible (has a bbox)
    track_length = track_length_visible + np.sum(occlusion_lengths)  # Total frames in the video where the object is visible or occluded

    # Occlusion percentage
    presence_ratio = track_length / video_length
    occlusion_percentage = (track_length - track_length_visible) / track_length * 100

    # Motion statistics (speed & abrupt changes)
    center_x = bboxes[:, 0] + bboxes[:, 2] / 2
    center_y = bboxes[:, 1] + bboxes[:, 3] / 2
    distances = np.sqrt(np.diff(center_x) ** 2 + np.diff(center_y) ** 2) / v_width  # Normalize by video width
    avg_speed = np.mean(distances) if len(distances) > 0 else 0
    number_of_abrupt_motion_changes = np.sum(distances > (1/25)) if len(distances) > 1 else 0

    # Scale statistics
    scales = (bboxes[:, 2] * bboxes[:, 3]) #/ v_area  # Normalize by image area
    min_scale = np.min(scales)
    max_scale = np.max(scales)
    avg_scale = np.mean(scales)
    #scale_changes = np.abs(np.diff(scales))
    scale_changes = np.abs(np.diff(scales)) / scales[:-1]
    number_of_scale_changes = np.sum(scale_changes > 0.5) if len(scale_changes) > 1 else 0
    avg_scale_change = np.mean(scale_changes) if len(scale_changes) > 0 else 0

    # Shape changes statistics
    aspect_ratios = bboxes[:, 2] / bboxes[:, 3]
    aspect_ratios_changes = np.abs(np.diff(aspect_ratios))
    number_of_shape_changes = np.sum(aspect_ratios_changes > 0.2) if len(aspect_ratios_changes) > 1 else 0

    return {
        'dataset': dataset_name,
        'video_id': video_id,
        'video': video_name,
        'video_width': v_width,
        'video_height': v_height,
        'video_length': video_length,  # Total frames in video
        'track': track_id,
        'track_length': track_length,  # Frames with valid bboxes
        'track_length_visible': track_length_visible,  # Frames with valid bboxes
        'presence_ratio': presence_ratio,
        'occlusion_percentage': occlusion_percentage,
        'number_of_occlusions': number_of_occlusions,
        'avg_occlusion_length': avg_occlusion_length,
        'max_occlusion_length': max_occlusion_length,
        'min_occlusion_length': min_occlusion_length,
        'avg_speed': avg_speed,
        'number_of_abrupt_motion_changes': number_of_abrupt_motion_changes,
        'min_scale': min_scale,
        'max_scale': max_scale,
        'avg_scale': avg_scale,
        'number_of_scale_changes': number_of_scale_changes,
        'avg_scale_change': avg_scale_change,
        'number_of_shape_changes': number_of_shape_changes,
        'category': cat_name,
    }

def process_dataset(data, dataset_name):
    """Process TAO dataset JSON and compute statistics for each track"""

    video_info = {video['id']: video for video in data['videos']}
    tracks_info = {track['id']: track for track in data['tracks']}
    image_info_dict = {image['id']: image for image in data['images']}
    cat_info_dict = {cat['id']: cat for cat in data['categories']}

    print(f"Number of videos: {len(video_info)}")
    print(f"Number of tracks: {len(tracks_info)}")
    print(f"Number of images: {len(data['images'])}")
    print(f"Number of annotations: {len(data['annotations'])}")

    track_annotations_dict = {}  # Dictionary to group annotations by track
    track_frame_indices = {}  # Dictionary to store frame indices per track
    video_frames_info = {}  # Dictionary to store frames per video

    # Group annotations by track_id
    for ann in data['annotations']:
        track_id = ann['track_id']
        image_id = ann['image_id']
        image_info = image_info_dict[image_id]

        if track_id not in track_annotations_dict:
            track_annotations_dict[track_id] = []
            track_frame_indices[track_id] = []

        track_annotations_dict[track_id].append(ann)
        track_frame_indices[track_id].append(image_info['frame_index'])

    # Group images by video_id
    for img in data['images']:
        video_id = img['video_id']
        if video_id not in video_frames_info:
            video_frames_info[video_id] = []
        video_frames_info[video_id].append(img)



    data_records = []

    for track_id, track_annotations in tqdm(track_annotations_dict.items()):
        track_info = tracks_info[track_id]
        video_id = track_info["video_id"]
        video_name = video_info[video_id]["name"]
        video_length = len(video_frames_info[video_id])
        video_frame_range = video_info[video_id]["frame_range"]
        video_dims = (video_info[video_id]["width"], video_info[video_id]["height"])
        cat_name = cat_info_dict[track_info["category_id"]]["name"]

        record = compute_statistics(track_id, dataset_name, video_id, video_name, track_annotations, track_frame_indices[track_id], video_length, video_frame_range, video_dims, cat_name)
        data_records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(data_records)
    return df

def sanity_check(data, df):

    assert len(df["dataset"].unique()) == 1
    assert len(df["video_id"].unique()) == len(data["videos"])
    assert len(df["video"].unique()) == len(data["videos"])
    assert len(df["track"].unique()) == len(data["tracks"])
    assert len(df["track"].unique()) == len(df)
    assert df.notnull().all().all()
    assert not df.isnull().values.any()

    images_by_video = {}
    for img in data["images"]:
        video_id = img["video_id"]
        if video_id not in images_by_video:
            images_by_video[video_id] = []
        images_by_video[video_id].append(img)

    tracks_by_video = {}
    for track in data["tracks"]:
        video_id = track["video_id"]
        if video_id not in tracks_by_video:
            tracks_by_video[video_id] = []
        tracks_by_video[video_id].append(track)

    for v in data["videos"]:
        video_id = v["id"]
        video_name = v["name"]
        video_width = v["width"]
        video_height = v["height"]
        video_length = len(images_by_video[video_id])

        assert video_length == df[df["video_id"] == video_id]["video_length"].values[0]
        assert video_width == df[df["video_id"] == video_id]["video_width"].values[0]
        assert video_height == df[df["video_id"] == video_id]["video_height"].values[0]
        assert video_id == df[df["video"] == video_name]["video_id"].values[0]
        assert video_name == df[df["video_id"] == video_id]["video"].values[0]

        # assert all track ids for each video are present in the dataframe with the same video_id
        assert set(df[df["video_id"] == video_id]["track"]) == set([track["id"] for track in tracks_by_video[video_id]])

    assert df["max_occlusion_length"].ge(df["min_occlusion_length"]).all()
    assert df["avg_occlusion_length"].ge(df["min_occlusion_length"]).all()
    assert df["avg_occlusion_length"].le(df["max_occlusion_length"]).all()

    assert df["video_length"].ge(df["track_length"]).all()
    assert (((df["video_length"] * df["presence_ratio"]) - df["track_length"]).abs() < 0.01).all()
    assert (((df["track_length"] * df["occlusion_percentage"] / 100) - df["number_of_occlusions"]).abs() >= 0).all()

    assert df["track_length"].ge(df["number_of_occlusions"]).all()
    assert df["track_length"].ge(df["number_of_abrupt_motion_changes"]).all()
    assert df["track_length"].ge(df["number_of_scale_changes"]).all()

    assert df["max_scale"].ge(df["min_scale"]).all()
    assert ((df["avg_scale"] >= df["min_scale"]) | ((df["avg_scale"] - df["min_scale"]).abs() < 0.01)).all()
    assert ((df["avg_scale"] <= df["max_scale"]) | ((df["avg_scale"] - df["max_scale"]).abs() < 0.01)).all()



if __name__ == "__main__":

    configs = [
        {"dataset": "HTD", "json_path": "../data/hard_tracks_dataset/annotations/hard_tracks_dataset_coco.json"},
        {"dataset": "HTD_VAL", "json_path": "../data/hard_tracks_dataset/annotations/hard_tracks_dataset_coco_val.json"},
        {"dataset": "ANIMALTRACK", "json_path": "../data/animaltrack/annotations/animaltrack_coco_val.json"},
        {"dataset": "BDD", "json_path": "../data/bdd/annotations/bdd_coco_val.json"},
        {"dataset": "BFT", "json_path": "../data/bft/annotations/bft_coco_val.json"},
        {"dataset": "DANCETRACK", "json_path": "../data/dancetrack/annotations/dancetrack_coco_val.json"},
        {"dataset": "GOT", "json_path": "../data/got/annotations/got_coco_val.json"},
        {"dataset": "LASOT", "json_path": "../data/lasot/annotations/lasot_coco_val.json"},
        {"dataset": "OVTB", "json_path": "../data/ovt-b/annotations/ovtb_coco_val.json"},
        {"dataset": "SPORTSMOT", "json_path": "../data/sportsmot/annotations/sportsmot_coco_val.json"},
        {"dataset": "TAO", "json_path": "../data/tao/annotations/tao_val_lvis_v1_classes.json"},
    ]

    for config in configs:
        print()
        print(f"Processing dataset: {config['dataset']}")

        annos = load_json(config["json_path"])
        df = process_dataset(annos, config["dataset"])

        sanity_check(annos, df)

        df.to_csv(f"statistics_{config['dataset']}.csv", index=False)

