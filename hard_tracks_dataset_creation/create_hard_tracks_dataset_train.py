import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil, errno


# UTILS

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
        #assert len(set([a["category_id"] for a in ann])) == 1, f"Annotations with different category IDs found in track {t}: {set([a['category_id'] for a in ann])}"

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

ANNOTATION_TEMPLATE = {
    "videos": ["id", "name", "width", "height", "frame_range", "metadata", "neg_category_ids", "not_exhaustive_category_ids"],
    "images": ["id", "video_id", "file_name", "video", "width", "height", "frame_id", "frame_index", "license", "neg_category_ids", "not_exhaustive_category_ids"],
    "annotations": ["id", "video_id", "image_id", "instance_id", "scale_category", "track_id", "bbox", "area", "iscrowd", "category_id", "segmentation"],
    "tracks": ["id", "video_id", "category_id"],
    "categories": ["id", "name"],
}

def load_json(json_path):
    """ Load JSON file """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def copy(src, dst):

    def ignore_non_images(dir, files):
        # Only keep .jpg or .png files and directories (so we can recurse)
        return [f for f in files if not (
                os.path.isdir(os.path.join(dir, f)) or f.lower().endswith(('.jpg', '.png'))
        )]

    try:
        # copy whole folder but only .jpg or .png files
        assert os.path.exists(src), f"Source path {src} does not exist"
        assert len(os.listdir(src)) > 0, f"Source path {src} is empty"
        #shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore_non_images)
    except OSError as exc: # python >2.5
        print(f"Error copying {src} to {dst}: {exc}")
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

def merge_sanity_check(annotations, annotations_list):

    expected_values = {
        "num_videos": sum([len(annos["videos"]) for annos in annotations_list]),
        "num_images": sum([len(annos["images"]) for annos in annotations_list]),
        "num_annotations": sum([len(annos["annotations"]) for annos in annotations_list]),
        "num_tracks": sum([len(annos["tracks"]) for annos in annotations_list]),
        "num_categories": sum([len(annos["categories"]) for annos in annotations_list]),
    }


    # General sanity check
    coco_sanity_check(
        annotations,
        expected_values=expected_values,
        annotations_template=ANNOTATION_TEMPLATE,
    )

def filter_coco_annotations_by_tracks(annotations, track_ids):

    # Filter annotations & tracks
    track_ids_set = set(track_ids) # speedup
    annotations["annotations"] = [a for a in annotations["annotations"] if a["track_id"] in track_ids_set]
    annotations["tracks"] = [t for t in annotations["tracks"] if t["id"] in track_ids_set]

    # Filter images & videos
    remaining_video_ids = set([a["video_id"] for a in annotations["annotations"]])
    annotations["videos"] = [v for v in annotations["videos"] if v["id"] in remaining_video_ids]
    annotations["images"] = [i for i in annotations["images"] if i["video_id"] in remaining_video_ids]

    # Filter categories
    remaining_category_ids = set([a["category_id"] for a in annotations["annotations"]])
    annotations["categories"] = [c for c in annotations["categories"] if c["id"] in remaining_category_ids]

    # Sanity checks
    assert len(annotations["tracks"]) == len(track_ids), f"Number of tracks {len(annotations['tracks'])} != {len(track_ids)}"
    assert len(annotations["videos"]) > 0, f"No videos left after filtering"
    coco_sanity_check(
        annotations,
        expected_values=None,
        annotations_template=ANNOTATION_TEMPLATE,
    )

    return annotations

def merge_annotations(annotations_list):

    global_video_id = 1
    global_track_id = 1
    global_image_id = 1
    global_annotation_id = 1
    global_category_id = 1

    # Update IDs in annotations
    for annotations in annotations_list:

        # Create local to global dictionaries
        local_to_global_track_id = {}
        for track in annotations["tracks"]:
            local_to_global_track_id[track["id"]] = global_track_id
            global_track_id += 1

        local_to_global_video_id = {}
        for video in annotations["videos"]:
            local_to_global_video_id[video["id"]] = global_video_id
            global_video_id += 1

        local_to_global_image_id = {}
        for image in annotations["images"]:
            local_to_global_image_id[image["id"]] = global_image_id
            global_image_id += 1

        local_to_global_annotation_id = {}
        for annotation in annotations["annotations"]:
            local_to_global_annotation_id[annotation["id"]] = global_annotation_id
            global_annotation_id += 1

        local_to_global_category_id = {}
        for category in annotations["categories"]:
            local_to_global_category_id[category["id"]] = global_category_id
            global_category_id += 1

        # Update annotations
        for track in annotations["tracks"]:
            track["id"] = local_to_global_track_id[track["id"]]
            track["video_id"] = local_to_global_video_id[track["video_id"]]
            track["category_id"] = local_to_global_category_id[track["category_id"]]

        for video in annotations["videos"]:
            video["id"] = local_to_global_video_id[video["id"]]

        for image in annotations["images"]:
            image["id"] = local_to_global_image_id[image["id"]]
            image["video_id"] = local_to_global_video_id[image["video_id"]]

        for annotation in annotations["annotations"]:
            annotation["id"] = local_to_global_annotation_id[annotation["id"]]
            annotation["track_id"] = local_to_global_track_id[annotation["track_id"]]
            annotation["image_id"] = local_to_global_image_id[annotation["image_id"]]
            annotation["category_id"] = local_to_global_category_id[annotation["category_id"]]
            annotation["video_id"] = local_to_global_video_id[annotation["video_id"]]

        for category in annotations["categories"]:
            category["id"] = local_to_global_category_id[category["id"]]


    # Merge annotations
    merged_annotations = {}

    merged_annotations["videos"] = []
    merged_annotations["tracks"] = []
    merged_annotations["images"] = []
    merged_annotations["annotations"] = []
    merged_annotations["categories"] = []

    for annotations in annotations_list:
        merged_annotations["videos"].extend(annotations["videos"])
        merged_annotations["tracks"].extend(annotations["tracks"])
        merged_annotations["images"].extend(annotations["images"])
        merged_annotations["annotations"].extend(annotations["annotations"])
        merged_annotations["categories"].extend(annotations["categories"])


    # sanity checks
    merge_sanity_check(merged_annotations, annotations_list)

    return merged_annotations

def show_process_init(dataset_name, hard_tracks):
    print()
    print(f"Processing {dataset_name} dataset")
    print(f"Number of hard tracks: {len(hard_tracks)}")
    print(f"Number of videos: {len(hard_tracks['video'].unique())}")
    print(f"Number of images: {hard_tracks.groupby('video')['video_length'].max().sum()}")


def process_hard_tracks_tao(hard_tracks, annotations_path, original_dataset_path, hard_tracks_dataset_path):

    show_process_init("TAO", hard_tracks)

    # Load annotations
    annotations = load_json(annotations_path)
    target_folder = "TAO"

    hard_tracks_ids = hard_tracks["track"].unique()
    hard_tracks_video_names = hard_tracks["video"].unique()

    if not os.path.exists(os.path.join(hard_tracks_dataset_path, "data", target_folder)):
        os.makedirs(os.path.join(hard_tracks_dataset_path, "data", target_folder))

    # Copy videos
    all_videos = []
    for dataset in os.listdir(os.path.join(original_dataset_path, "frames", "train")):
        for video in os.listdir(os.path.join(original_dataset_path, "frames", "train", dataset)):
            all_videos.append(f"train/{dataset}/{video}")

    video_folders_to_copy = [v for v in all_videos if v in hard_tracks_video_names]
    for video_folder in tqdm(video_folders_to_copy):
        # copy whole folder to destination
        video_path = os.path.join(original_dataset_path, "frames", "train", video_folder.split("/")[1], video_folder.split("/")[2])
        video_path_dst = os.path.join(hard_tracks_dataset_path, "data", target_folder, video_folder.split("/")[1], video_folder.split("/")[2])
        copy(video_path, video_path_dst)

    # Filter & adapt annotations
    filtered_annotations = filter_coco_annotations_by_tracks(annotations, hard_tracks_ids)

    for video in filtered_annotations["videos"]:
        parts = video["name"].split("/")
        video["name"] = f"{target_folder}/{parts[1]}/{parts[2]}"

    for image in filtered_annotations["images"]:
        v_parts = image["video"].split("/")
        image["video"] = f"{target_folder}/{v_parts[1]}/{v_parts[2]}"
        i_parts = image["file_name"].split("/")
        image["file_name"] = f"{target_folder}/{i_parts[1]}/{i_parts[2]}/{i_parts[3]}"

    return filtered_annotations

def process_hard_tracks_bdd(hard_tracks, annotations_path, original_dataset_path, hard_tracks_dataset_path):

    show_process_init("BDD", hard_tracks)

    # Load annotations
    annotations = load_json(annotations_path)
    target_folder = "BDD"

    hard_tracks_ids = hard_tracks["track"].unique()
    hard_tracks_video_names = hard_tracks["video"].unique()

    if not os.path.exists(os.path.join(hard_tracks_dataset_path, "data", target_folder)):
        os.makedirs(os.path.join(hard_tracks_dataset_path, "data", target_folder))

    # Copy videos
    video_folders_to_copy = [v for v in os.listdir(os.path.join(original_dataset_path, "bdd100k", "images", "track", "train")) if v in hard_tracks_video_names]
    for video_folder in tqdm(video_folders_to_copy):
        # copy whole folder to destination
        video_path = os.path.join(original_dataset_path, "bdd100k", "images", "track", "train", video_folder)
        video_path_dst = os.path.join(hard_tracks_dataset_path, "data", target_folder, video_folder)
        copy(video_path, video_path_dst)

    # Filter & adapt annotations
    filtered_annotations = filter_coco_annotations_by_tracks(annotations, hard_tracks_ids)

    for video in filtered_annotations["videos"]:
        video["name"] = f"{target_folder}/{video['name']}"

    for image in filtered_annotations["images"]:
        image["video"] = f"{target_folder}/{image['video']}"
        image["file_name"] = f"{target_folder}/{image['file_name']}"

    return filtered_annotations

def process_hard_tracks_got(hard_tracks, annotations_path, original_dataset_path, hard_tracks_dataset_path):

    show_process_init("GOT", hard_tracks)

    # Load annotations
    annotations = load_json(annotations_path)
    target_folder = "GOT10k"

    hard_tracks_ids = hard_tracks["track"].unique()
    hard_tracks_video_names = hard_tracks["video"].unique()

    if not os.path.exists(os.path.join(hard_tracks_dataset_path, "data", target_folder)):
        os.makedirs(os.path.join(hard_tracks_dataset_path, "data", target_folder))

    video_folders_to_copy = [v for v in os.listdir(os.path.join(original_dataset_path, "train")) if v in hard_tracks_video_names]
    for video_folder in tqdm(video_folders_to_copy):

        # copy whole folder to destination
        video_path = os.path.join(original_dataset_path, "train", video_folder)
        video_path_dst = os.path.join(hard_tracks_dataset_path, "data", target_folder, video_folder)
        copy(video_path, video_path_dst)

    # Filter & adapt annotations
    filtered_annotations = filter_coco_annotations_by_tracks(annotations, hard_tracks_ids)

    for video in filtered_annotations["videos"]:
        video["name"] = f"{target_folder}/{video['name']}"

    for image in filtered_annotations["images"]:
        image["video"] = f"{target_folder}/{image['video']}"
        i_parts = image["file_name"].split("/")
        image["file_name"] = f"{target_folder}/{i_parts[1]}/{i_parts[2]}"

    return filtered_annotations

def process_hard_tracks_lasot(hard_tracks, annotations_path, original_dataset_path, hard_tracks_dataset_path):

    show_process_init("LaSOT", hard_tracks)

    # Load annotations
    annotations = load_json(annotations_path)
    target_folder = "LaSOT"

    hard_tracks_ids = hard_tracks["track"].unique()
    hard_tracks_video_names = hard_tracks["video"].unique()

    if not os.path.exists(os.path.join(hard_tracks_dataset_path, "data", target_folder)):
        os.makedirs(os.path.join(hard_tracks_dataset_path, "data", target_folder))

    video_folders_to_copy = [v for v in os.listdir(os.path.join(original_dataset_path, "data")) if v in hard_tracks_video_names]
    for video_folder in tqdm(video_folders_to_copy):
        # copy whole folder to destination
        video_path = os.path.join(original_dataset_path, "data", video_folder, "img")
        video_path_dst = os.path.join(hard_tracks_dataset_path, "data", target_folder, video_folder)
        copy(video_path, video_path_dst)

    # Filter & adapt annotations
    filtered_annotations = filter_coco_annotations_by_tracks(annotations, hard_tracks_ids)

    for video in filtered_annotations["videos"]:
        video["name"] = f"{target_folder}/{video['name']}"

    for image in filtered_annotations["images"]:
        image["video"] = f"{target_folder}/{image['video']}"
        i_parts = image["file_name"].split("/")
        image["file_name"] = f"{target_folder}/{i_parts[1]}/{i_parts[3]}" # skip the img part (2)

    return filtered_annotations

def process_hard_tracks_ovtb(hard_tracks, annotations_path, original_dataset_path, hard_tracks_dataset_path):

    show_process_init("OVT-B", hard_tracks)

    # Load annotations
    annotations = load_json(annotations_path)
    target_folder = "OVT-B"

    hard_tracks_ids = hard_tracks["track"].unique()
    hard_tracks_video_names = hard_tracks["video"].unique()

    if not os.path.exists(os.path.join(hard_tracks_dataset_path, "data", target_folder)):
        os.makedirs(os.path.join(hard_tracks_dataset_path, "data", target_folder))


    all_videos = []
    for dataset in os.listdir(os.path.join(original_dataset_path, "frames")):
        for video in os.listdir(os.path.join(original_dataset_path, "frames", dataset)):
            all_videos.append(f"{dataset}/{video}")

    video_folders_to_copy = [v for v in all_videos if v in hard_tracks_video_names]
    for video_folder in tqdm(video_folders_to_copy):
        # copy whole folder to destination
        video_path = os.path.join(original_dataset_path, "frames", video_folder.split("/")[0], video_folder.split("/")[1])
        video_path_dst = os.path.join(hard_tracks_dataset_path, "data", target_folder, video_folder)
        copy(video_path, video_path_dst)

    # Filter & adapt annotations
    filtered_annotations = filter_coco_annotations_by_tracks(annotations, hard_tracks_ids)

    for video in filtered_annotations["videos"]:
        video["name"] = f"{target_folder}/{video['name']}"

    for image in filtered_annotations["images"]:
        image["video"] = f"{target_folder}/{image['video']}"
        image["file_name"] = f"{target_folder}/{image['file_name']}"

    return filtered_annotations

def process_hard_tracks_animaltrack(hard_tracks, annotations_path, original_dataset_path, hard_tracks_dataset_path):

    show_process_init("AnimalTrack", hard_tracks)

    # Load annotations
    annotations = load_json(annotations_path)
    target_folder = "AnimalTrack"

    hard_tracks_ids = hard_tracks["track"].unique()
    hard_tracks_video_names = hard_tracks["video"].unique()

    if not os.path.exists(os.path.join(hard_tracks_dataset_path, "data", target_folder)):
        os.makedirs(os.path.join(hard_tracks_dataset_path, "data", target_folder))

    video_folders_to_copy = [v for v in os.listdir(os.path.join(original_dataset_path, "videos_all")) if v in hard_tracks_video_names]
    for video_folder in tqdm(video_folders_to_copy):
        # copy whole folder to destination
        video_path = os.path.join(original_dataset_path, "videos_all", video_folder)
        video_path_dst = os.path.join(hard_tracks_dataset_path, "data", target_folder, video_folder)
        copy(video_path, video_path_dst)

    # Filter & adapt annotations
    filtered_annotations = filter_coco_annotations_by_tracks(annotations, hard_tracks_ids)

    for video in filtered_annotations["videos"]:
        video["name"] = f"{target_folder}/{video['name']}"

    for image in filtered_annotations["images"]:
        image["video"] = f"{target_folder}/{image['video']}"
        i_parts = image["file_name"].split("/")
        image["file_name"] = f"{target_folder}/{i_parts[1]}/{i_parts[2]}"

    return filtered_annotations

def process_hard_tracks_sportsmot(hard_tracks, annotations_path, original_dataset_path, hard_tracks_dataset_path):

    show_process_init("SportsMOT", hard_tracks)

    # Load annotations
    annotations = load_json(annotations_path)
    target_folder = "SportsMOT"

    hard_tracks_ids = hard_tracks["track"].unique()
    hard_tracks_video_names = hard_tracks["video"].unique()

    if not os.path.exists(os.path.join(hard_tracks_dataset_path, "data", target_folder)):
        os.makedirs(os.path.join(hard_tracks_dataset_path, "data", target_folder))

    video_folders_to_copy = [v for v in os.listdir(os.path.join(original_dataset_path, "dataset", "train")) if v in hard_tracks_video_names]
    for video_folder in tqdm(video_folders_to_copy):
        # copy whole folder to destination
        video_path = os.path.join(original_dataset_path, "dataset", "train", video_folder, "img1")
        video_path_dst = os.path.join(hard_tracks_dataset_path, "data", target_folder, video_folder)
        copy(video_path, video_path_dst)

    # Filter & adapt annotations
    filtered_annotations = filter_coco_annotations_by_tracks(annotations, hard_tracks_ids)

    for video in filtered_annotations["videos"]:
        video["name"] = f"{target_folder}/{video['name']}"

    for image in filtered_annotations["images"]:
        image["video"] = f"{target_folder}/{image['video']}"
        i_parts = image["file_name"].split("/")
        image["file_name"] = f"{target_folder}/{i_parts[1]}/{i_parts[3]}"

    return filtered_annotations

def process_hard_tracks_dancetrack(hard_tracks, annotations_path, original_dataset_path, hard_tracks_dataset_path):

    show_process_init("DanceTrack", hard_tracks)

    # Load annotations
    annotations = load_json(annotations_path)
    target_folder = "DanceTrack"

    hard_tracks_ids = hard_tracks["track"].unique()
    hard_tracks_video_names = hard_tracks["video"].unique()

    if not os.path.exists(os.path.join(hard_tracks_dataset_path, "data", target_folder)):
        os.makedirs(os.path.join(hard_tracks_dataset_path, "data", target_folder))

    video_folders_to_copy = [v for v in os.listdir(os.path.join(original_dataset_path, "train")) if v in hard_tracks_video_names]
    for video_folder in tqdm(video_folders_to_copy):
        # copy whole folder to destination
        video_path = os.path.join(original_dataset_path, "train", video_folder, "img1")
        video_path_dst = os.path.join(hard_tracks_dataset_path, "data", target_folder, video_folder)
        copy(video_path, video_path_dst)

    # Filter & adapt annotations
    filtered_annotations = filter_coco_annotations_by_tracks(annotations, hard_tracks_ids)

    for video in filtered_annotations["videos"]:
        video["name"] = f"{target_folder}/{video['name']}"

    for image in filtered_annotations["images"]:
        image["video"] = f"{target_folder}/{image['video']}"
        i_parts = image["file_name"].split("/")
        image["file_name"] = f"{target_folder}/{i_parts[1]}/{i_parts[3]}"

    return filtered_annotations

def process_hard_tracks_bft(hard_tracks, annotations_path, original_dataset_path, hard_tracks_dataset_path):

    show_process_init("BFT", hard_tracks)

    # Load annotations
    annotations = load_json(annotations_path)
    target_folder = "BFT"

    hard_tracks_ids = hard_tracks["track"].unique()
    hard_tracks_video_names = hard_tracks["video"].unique()

    if not os.path.exists(os.path.join(hard_tracks_dataset_path, "data", target_folder)):
        os.makedirs(os.path.join(hard_tracks_dataset_path, "data", target_folder))

    video_folders_to_copy = [v for v in os.listdir(os.path.join(original_dataset_path, "frames", "train")) if v in hard_tracks_video_names]
    for video_folder in tqdm(video_folders_to_copy):
        # copy whole folder to destination
        video_path = os.path.join(original_dataset_path, "frames", "train", video_folder)
        video_path_dst = os.path.join(hard_tracks_dataset_path, "data", target_folder, video_folder)
        copy(video_path, video_path_dst)

    # Filter & adapt annotations
    filtered_annotations = filter_coco_annotations_by_tracks(annotations, hard_tracks_ids)

    for video in filtered_annotations["videos"]:
        video["name"] = f"{target_folder}/{video['name']}"

    for image in filtered_annotations["images"]:
        image["video"] = f"{target_folder}/{image['video']}"
        i_parts = image["file_name"].split("/")
        image["file_name"] = f"{target_folder}/{i_parts[1]}/{i_parts[2]}"

    return filtered_annotations




def main():

    # Config
    statistics_path = "."
    dataset_path = "/srv/beegfs02/scratch/visobt4s/data/tracking_datasets/"
    folder_path = "hard_tracks_dataset2"

    # Create necessary folder structure
    if os.path.exists(os.path.join(dataset_path, folder_path)):
        shutil.rmtree(os.path.join(dataset_path, folder_path))

    os.makedirs(os.path.join(dataset_path, folder_path))

    if not os.path.exists(os.path.join(dataset_path, folder_path, "annotations")):
        os.makedirs(os.path.join(dataset_path, folder_path, "annotations"))

    if not os.path.exists(os.path.join(dataset_path, folder_path, "data")):
        os.makedirs(os.path.join(dataset_path, folder_path, "data"))

    # Load dataset statistics
    #statistics_tao = pd.read_csv(os.path.join(statistics_path, "statistics_TAO_TRAIN.csv"))
    statistics_bdd = pd.read_csv(os.path.join(statistics_path, "statistics_BDD_TRAIN.csv"))
    statistics_got = pd.read_csv(os.path.join(statistics_path, "statistics_GOT_TRAIN.csv"))
    statistics_lasot = pd.read_csv(os.path.join(statistics_path, "statistics_LASOT_TRAIN.csv"))
    #statistics_ovtb = pd.read_csv(os.path.join(statistics_path, "statistics_OVTB_TRAIN.csv"))
    statistics_animaltrack = pd.read_csv(os.path.join(statistics_path, "statistics_ANIMALTRACK_TRAIN.csv"))
    statistics_sportsmot = pd.read_csv(os.path.join(statistics_path, "statistics_SPORTSMOT_TRAIN.csv"))
    statistics_dancetrack = pd.read_csv(os.path.join(statistics_path, "statistics_DANCETRACK_TRAIN.csv"))
    statistics_bft = pd.read_csv(os.path.join(statistics_path, "statistics_BFT_TRAIN.csv"))

    # Filter hard tracks
    #hard_tracks_tao = statistics_tao[(statistics_tao["number_of_occlusions"] >= 3) & (statistics_tao["track_length"] >= 40)]
    hard_tracks_bdd = statistics_bdd[(statistics_bdd["number_of_occlusions"] >= 5) & (statistics_bdd["track_length"] >= 100)]
    hard_tracks_got = statistics_got[(statistics_got["number_of_occlusions"] >= 1)]
    hard_tracks_lasot = statistics_lasot[(statistics_lasot["number_of_occlusions"] >= 20)]
    #hard_tracks_ovtb = statistics_ovtb[(statistics_ovtb["number_of_occlusions"] >= 2) & (statistics_ovtb["track_length"] >= 100)]
    hard_tracks_animaltrack = statistics_animaltrack[(statistics_animaltrack["number_of_occlusions"] >= 3) & (statistics_animaltrack["track_length"] >= 200)]
    hard_tracks_sportsmot = statistics_sportsmot[(statistics_sportsmot["number_of_occlusions"] >= 2) & (statistics_sportsmot["track_length"] >= 300)]
    hard_tracks_dancetrack = statistics_dancetrack[(statistics_dancetrack["number_of_occlusions"] >= 10) & (statistics_dancetrack["track_length"] >= 400)]
    hard_tracks_bft = statistics_bft[(statistics_bft["number_of_occlusions"] >= 2) & (statistics_bft["track_length"] >= 100)]

    # Process hard tracks
    #annotations_tao = process_hard_tracks_tao(hard_tracks_tao, "../data/tao/annotations/tao_val_lvis_v1_classes.json", "../data/tao/", os.path.join(dataset_path, folder_path))
    annotations_bdd = process_hard_tracks_bdd(hard_tracks_bdd, "../data/bdd/annotations/bdd_coco_train.json", "../data/bdd/", os.path.join(dataset_path, folder_path))
    annotations_got = process_hard_tracks_got(hard_tracks_got, "../data/got10k/annotations/got_coco_train.json", "../data/got10k/", os.path.join(dataset_path, folder_path))
    annotations_lasot = process_hard_tracks_lasot(hard_tracks_lasot, "../data/lasot/annotations/lasot_coco_train.json", "../data/lasot/", os.path.join(dataset_path, folder_path))
    #annotations_ovtb = process_hard_tracks_ovtb(hard_tracks_ovtb, "../data/ovtb/ovtb_coco_val.json", "../data/ovtb/", os.path.join(dataset_path, folder_path))
    annotations_animaltrack = process_hard_tracks_animaltrack(hard_tracks_animaltrack, "../data/animaltrack/annotations/animaltrack_coco_train.json", "../data/animaltrack/", os.path.join(dataset_path, folder_path))
    annotations_sportsmot = process_hard_tracks_sportsmot(hard_tracks_sportsmot, "../data/metadata/sportsmot_coco_train.json", "../data/sportsmot/", os.path.join(dataset_path, folder_path))
    annotations_dancetrack = process_hard_tracks_dancetrack(hard_tracks_dancetrack, "../data/dancetrack/annotations/dancetrack_coco_train.json", "../data/dancetrack/", os.path.join(dataset_path, folder_path))
    annotations_bft = process_hard_tracks_bft(hard_tracks_bft, "../data/bft/annotations/bft_coco_train.json", "../data/bft/", os.path.join(dataset_path, folder_path))




    # Merge annotations & save
    annotations_list = [
        #annotations_tao,
        annotations_bdd,
        annotations_got,
        annotations_lasot,
        #annotations_ovtb,
        annotations_animaltrack,
        annotations_sportsmot,
        annotations_dancetrack,
        annotations_bft,
    ]

    merged_annotations = merge_annotations(annotations_list)

    with open(os.path.join(dataset_path, folder_path, "annotations", "hard_tracks_dataset_coco.json"), "w") as json_file:
        json.dump(merged_annotations, json_file, indent=4)

    print()
    print("Summary of hard tracks dataset:")

    merged_df = pd.concat([
        #hard_tracks_tao,
        hard_tracks_bdd,
        hard_tracks_got,
        hard_tracks_lasot,
        #hard_tracks_ovtb,
        hard_tracks_animaltrack,
        hard_tracks_sportsmot,
        hard_tracks_dancetrack,
        hard_tracks_bft,
    ])

    print(f"Number of hard tracks: {len(merged_df)} ({len(merged_annotations['tracks'])})")
    print(f"Number of videos: {len(merged_df['video'].unique())} ({len(merged_annotations['videos'])})")
    print(f"Number of images: {len(merged_annotations['images'])}")
    print(f"Number of categories: {len(merged_annotations['categories'])}")
    print(f"Minimum track length: {merged_df['track_length'].min()}")
    print(f"Maximum track length: {merged_df['track_length'].max()}")
    print(f"Mean track length: {merged_df['track_length'].mean()}")
    print(f"Minimum number of occlusions: {merged_df['number_of_occlusions'].min()}")
    print(f"Maximum number of occlusions: {merged_df['number_of_occlusions'].max()}")
    print(f"Mean number of occlusions: {merged_df['number_of_occlusions'].mean()}")





if __name__ == "__main__":
    main()




# For testing -----------------------------------

#statistics_tao[(statistics_tao["track"] == 15553) | (statistics_tao["track"] == 10282)]
#statistics_bdd[statistics_bdd["track"] == 225]
#statistics_got[statistics_got["track"] == 1]
#statistics_lasot[statistics_lasot["track"] == 1]
#statistics_ovtb[(statistics_ovtb["track"] == 0) | (statistics_ovtb["track"] == 899)]
#statistics_animaltrack[statistics_animaltrack["track"] == 1]
#statistics_sportsmot[statistics_sportsmot["track"] == 306]
#statistics_dancetrack[statistics_dancetrack["track"] == 20]
#statistics_bft[statistics_bft["track"] == 6]
