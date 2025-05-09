import os
import json
import cv2
import multiprocessing
from tqdm import tqdm
import numpy as np
import re
import traceback

# Define COCO format structure
coco_data = {
    "tracks": [],
    "videos": [],
    "images": [],
    "annotations": [],
    "categories": []
}


def load_txt_file(filepath, dtype=float):
    """ Load a text file and return a NumPy array. """
    if not os.path.exists(filepath):
        return None
    return np.loadtxt(filepath, dtype=dtype, delimiter=",")


def process_video(data_dir, image_root_path, video_folder, shared_data):
    """Processes a single video folder and returns image & annotation data."""

    try:
        video_path = os.path.join(data_dir, video_folder)
        gt_file = os.path.join(data_dir, "..", "gt_all", f"{video_folder}_gt.txt")
        image_folder_path = video_path

        if not os.path.exists(gt_file) or not os.path.exists(video_path):
            print(f"Skipping video {video_folder} (missing groundtruth.txt or image folder)")
            return [], [], [], []

        gt = load_txt_file(gt_file, dtype=float)

        images = []
        annotations = []

        # Add video
        video_id = shared_data["video_id"]
        shared_data["video_id"] += 1

        first_img = cv2.imread(os.path.join(image_folder_path, os.listdir(os.path.join(image_folder_path))[0]))
        img_width = first_img.shape[1]
        img_height = first_img.shape[0]

        videos = [{
            "id": video_id,
            "name": video_folder,
            "width": img_width,
            "height": img_height,
            "frame_range": 1,
            "metadata": {
                "dataset": "AnimalTrack",
                "user_id": "user",
                "username": "user",
            },
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        }]

        # Add categories if not already present
        unique_classes = np.unique(gt[:, 7])
        for class_id in unique_classes:
            if class_id not in [cat['id'] for cat in coco_data["categories"]]:
                class_name = video_folder.split("_")[0]
                coco_data["categories"].append({"id": int(class_id), "name": class_name})

        # Add tracks
        tracks = []
        unique_track_ids = np.unique(gt[:, 1])
        local_to_global_track_id = {}
        for idx, track_id in enumerate(unique_track_ids):
            global_track_id = shared_data["track_id"]
            shared_data["track_id"] += 1

            # find first entry with the track id in the gt list and take its class
            cat_id = gt[gt[:, 1] == track_id][0, 7]  # Class ID

            tracks.append({
                "id": global_track_id,
                "category_id": int(cat_id),
                "video_id": video_id
            })

            # Map local track ID to global track ID
            local_to_global_track_id[track_id] = global_track_id

        # Add images
        image_files = [f for f in os.listdir(image_folder_path)]
        for idx, image_name in enumerate(sorted(image_files)):
            img_id = shared_data["image_id"]
            shared_data["image_id"] += 1

            # Add image metadata
            images.append({
                "id": img_id,
                "video": video_folder,
                "file_name": os.path.join(image_root_path, video_folder, image_name),
                "width": img_width,
                "height": img_height,
                "frame_index": idx,
                "license": 0,
                "video_id": video_id,
                "frame_id": idx + 1,  # +1 to match the mot format
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
                #"text": text
            })

        # Add annotations
        for idx, gt_row in enumerate(gt):
            track_id = gt_row[1]  # Local track ID
            global_track_id = local_to_global_track_id[track_id]
            image = [img for img in images if img["frame_id"] == gt_row[0] and img["video"] == video_folder]
            assert len(image) == 1
            image_id = image[0]["id"]

            class_id = gt_row[7]  # Class ID

            x, y, w, h = gt_row[2:6]  # x, y, w, h

            # Lock and update annotation ID
            ann_id = shared_data["annotation_id"]
            shared_data["annotation_id"] += 1

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "video_id": video_id,
                "instance_id": int(track_id),  # Unique within a video
                "scale_category": "moving-object",
                "track_id": global_track_id,
                "segmentation": [],
                "category_id": int(class_id),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })

        return images, annotations, videos, tracks

    except Exception as e:
        print(f"Error processing video {video_folder}: {e}, Stack trace: {traceback.format_exc()}")
        return [], [], [], []


if __name__ == "__main__":

    data_dir = "../data/animaltrack/videos_all"
    annotations_dir = "../data/animaltrack/annotations"
    image_root_path = "videos_all"

    video_folders = os.listdir(data_dir)
    video_folders = list(filter(lambda x: os.path.isdir(os.path.join(data_dir, x)), video_folders))  # Filter out non-folders


    #with open(f"../data/animaltrack/gt_test.txt", "r") as json_file:
    #    test_files = json_file.readlines()
    #video_folders = list(map(lambda x: '_'.join(x.split("_")[:2]), test_files))

    with open(f"../data/animaltrack/gt_train.txt", "r") as json_file:
        train_files = json_file.readlines()
    video_folders = list(map(lambda x: '_'.join(x.split("_")[:2]), train_files))


    print(f"Processing {len(video_folders)} videos")

    shared_data = {
        "image_id": 1,
        "annotation_id": 1,
        "video_id": 1,
        "track_id": 1,
        "category_id": 1
    }

    for vf in tqdm(video_folders):
        results = process_video(data_dir, image_root_path, vf, shared_data)
        coco_data["images"].extend(results[0])
        coco_data["annotations"].extend(results[1])
        coco_data["videos"].extend(results[2])
        coco_data["tracks"].extend(results[3])

    # Save to JSON
    with open(f"{annotations_dir}/animaltrack_coco_train.json", "w") as json_file:
        json.dump(coco_data, json_file, indent=4)

    print("COCO dataset saved as coco_dataset.json")