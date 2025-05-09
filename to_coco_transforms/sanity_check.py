import json
import os
import numpy as np


def load_json(json_path):
    """ Load JSON file """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def coco_sanity_check(annotations, expected_values, annotations_template):

    # Check if the annotations have all the expected keys
    if annotations_template is not None:
        for key in annotations_template.keys():
            assert key in annotations, f"Missing key {key} in annotations"

            expected_fields = annotations_template[key]

            for v in annotations[key]:
                for field in expected_fields:
                    assert field in v, f"Missing field {field} in {key}"
                    assert v[field] is not None, f"Field {field} in {key} is None"
                    assert v[field] != "", f"Field {field} in {key} is empty"

    # Check if all expected values are present
    if expected_values is not None:
        assert len(annotations["videos"]) == expected_values["num_videos"]
        assert len(annotations["images"]) == expected_values["num_images"]
        assert len(annotations["annotations"]) == expected_values["num_annotations"]
        assert len(annotations["tracks"]) == expected_values["num_tracks"]
        assert len(annotations["categories"]) == expected_values["num_categories"]



    # Prepare for sanity check
    video_by_id = {video["id"]: video for video in annotations["videos"]}
    image_by_id = {image["id"]: image for image in annotations["images"]}
    annotation_by_id = {annotation["id"]: annotation for annotation in annotations["annotations"]}
    track_by_id = {track["id"]: track for track in annotations["tracks"]}
    category_by_id = {category["id"]: category for category in annotations["categories"]}

    images_by_video_dict = {}
    for image in annotations["images"]:
        video_id = image["video_id"]
        if video_id not in images_by_video_dict:
            images_by_video_dict[video_id] = []
        images_by_video_dict[video_id].append(image)

    annotations_by_track_dict = {}
    for annotation in annotations["annotations"]:
        track_id = annotation["track_id"]
        if track_id not in annotations_by_track_dict:
            annotations_by_track_dict[track_id] = []
        annotations_by_track_dict[track_id].append(annotation)




    # Checks for videos -----------------------------------------------------
    assert len(set([video["id"] for video in annotations["videos"]])) == len(annotations["videos"]), f"Videos with duplicate IDs found"


    # Checks for images -----------------------------------------------------
    assert len(set([image["id"] for image in annotations["images"]])) == len(annotations["images"]), f"Images with duplicate IDs found"
    videos_referenced_from_images = set([image["video_id"] for image in annotations["images"]])
    assert videos_referenced_from_images == set(video_by_id.keys()), f"Some videos have not been referenced in images"

    for image in annotations["images"]:
        v = video_by_id[image["video_id"]]
        assert image["width"] == v["width"], f"Image {image['id']} has different width than video {image['video_id']}"
        assert image["height"] == v["height"], f"Image {image['id']} has different height than video {image['video_id']}"
        assert image["video"] == v["name"], f"Image {image['id']} has different video name than video {image['video_id']}"

    for v, img_list in images_by_video_dict.items():
        assert len(set([img["frame_id"] for img in img_list])) == len(img_list), f"Images with duplicate frame IDs found in video {img_list[0]['video_id']}"
        assert len(set([img["frame_index"] for img in img_list])) == len(img_list), f"Images with duplicate frame indexes found in video {img_list[0]['video_id']}"
        v_frame_range = video_by_id[v]["frame_range"]
        assert all([img["frame_index"] % v_frame_range == 0 for img in img_list]), f"Frame index of image {img_list[0]['id']} is not divisible by frame range of video {v}"


    # Checks for annotations -----------------------------------------------------
    assert len(set([annotation["id"] for annotation in annotations["annotations"]])) == len(annotations["annotations"]), f"Annotations with duplicate IDs found"
    videos_referenced_from_annotations = set([annotation["video_id"] for annotation in annotations["annotations"]])
    images_referenced_from_annotations = set([annotation["image_id"] for annotation in annotations["annotations"]])
    tracks_referenced_from_annotations = set([annotation["track_id"] for annotation in annotations["annotations"]])
    categories_referenced_from_annotations = set([annotation["category_id"] for annotation in annotations["annotations"]])
    assert videos_referenced_from_annotations == set(video_by_id.keys()), f"Some videos have not been referenced in annotations"
    assert images_referenced_from_annotations.issubset(set(image_by_id.keys())), f"Some annotations reference images that are not in the dataset"
    assert tracks_referenced_from_annotations == set(track_by_id.keys()), f"Some tracks have not been referenced in annotations"
    assert categories_referenced_from_annotations.issubset(set(category_by_id.keys())), f"Some annotations reference categories that are not in the dataset"

    if categories_referenced_from_annotations != set(category_by_id.keys()):
        print(f"NOTE: {len(set(category_by_id.keys()) - categories_referenced_from_annotations)} categories are not referenced by annotations in the dataset")

    for t, ann in annotations_by_track_dict.items():
        assert len(set([a["instance_id"] for a in ann])) == 1, f"Annotations with different instance IDs found in track {t}"
        assert len(set([a["track_id"] for a in ann])) == 1, f"Annotations with different track IDs found in track {t}"
        assert len(set([a["video_id"] for a in ann])) == 1, f"Annotations with different video IDs found in track {t}"
        assert len(set([a["category_id"] for a in ann])) == 1, f"Annotations with different category IDs found in track {t}: {set([a['category_id'] for a in ann])}"

        referenced_track = track_by_id[t]
        for a in ann:
            a["track_id"] = referenced_track["id"]
            a["video_id"] = referenced_track["video_id"]
            a["category_id"] = referenced_track["category_id"]

    image_id_instance_id_combinations = [str(a["image_id"]) + "-" + str(a["instance_id"]) for a in annotations["annotations"]]
    assert len(set(image_id_instance_id_combinations)) == len(image_id_instance_id_combinations), f"Same image_id and instance_id combinations found multiple times"



    # Checks for tracks -----------------------------------------------------
    assert len(set([track["id"] for track in annotations["tracks"]])) == len(annotations["tracks"]), f"Tracks with duplicate IDs found"
    videos_referenced_from_tracks = set([track["video_id"] for track in annotations["tracks"]])
    categories_referenced_from_tracks = set([track["category_id"] for track in annotations["tracks"]])
    assert videos_referenced_from_tracks == set(video_by_id.keys()), f"Some videos have not been referenced in tracks"
    assert categories_referenced_from_tracks.issubset(set(category_by_id.keys())), f"Some tracks reference categories that are not in the dataset"

    if categories_referenced_from_tracks != set(category_by_id.keys()):
        print(f"NOTE: {len(set(category_by_id.keys()) - categories_referenced_from_tracks)} categories are not referenced by tracks in the dataset")


    # Checks for categories -----------------------------------------------------
    assert len(set([category["id"] for category in annotations["categories"]])) == len(annotations["categories"]), f"Categories with duplicate IDs found"


    # Check frame indices
    for v, imgs in images_by_video_dict.items():
        v_frame_range = video_by_id[v]["frame_range"]
        frame_indices = [img["frame_index"] for img in imgs]

        frame_indices_diffs = np.diff(np.sort(frame_indices))
        assert len(np.unique(frame_indices_diffs)) == 1, f"Frame indices are not equally spaced in video {v}"
        assert all([frame_index % v_frame_range == 0 for frame_index in frame_indices]), f"Frame indices are not divisible by frame range in video {v}"



ANNOTATION_TEMPLATE = {
    "videos": ["id", "name", "width", "height", "frame_range", "metadata", "neg_category_ids", "not_exhaustive_category_ids"],
    "images": ["id", "video_id", "file_name", "video", "width", "height", "frame_id", "frame_index", "license", "neg_category_ids", "not_exhaustive_category_ids"],
    "annotations": ["id", "video_id", "image_id", "instance_id", "scale_category", "track_id", "bbox", "area", "iscrowd", "category_id", "segmentation"],
    "tracks": ["id", "video_id", "category_id"],
    "categories": ["id", "name"],
}


if __name__ == "__main__":



    annotations_checks = [
        {
            "dataset": "HARD TRACKS DATASET",
            "annotation_file": "/srv/beegfs02/scratch/visobt4s/data/tracking_datasets/hard_tracks_dataset/annotations/hard_tracks_dataset_coco_original.json",
            "expected_values": None,
            "annotations_template": ANNOTATION_TEMPLATE
        },
        {
            "dataset": "HARD TRACKS DATASET",
            "annotation_file": "/srv/beegfs02/scratch/visobt4s/data/tracking_datasets/hard_tracks_dataset/annotations/hard_tracks_dataset_coco_cleaned.json",
            "expected_values": None,
            "annotations_template": ANNOTATION_TEMPLATE
        },
        {
            "dataset": "HARD TRACKS DATASET",
            "annotation_file": "/srv/beegfs02/scratch/visobt4s/data/tracking_datasets/hard_tracks_dataset/annotations/hard_tracks_dataset_coco_class_agnostic.json",
            "expected_values": None,
            "annotations_template": ANNOTATION_TEMPLATE
        },
    ]


    for check in annotations_checks:

        print()
        print(f"Sanity check for {check['dataset']}")
        print("=" * 50)
        print(f"Path: {check['annotation_file']}")
        print(f"Expected values: {check['expected_values']}")

        annos = load_json(check["annotation_file"])
        coco_sanity_check(
            annos,
            expected_values=None,
            annotations_template=check["annotations_template"]
        )