import os
import json
import cv2
import multiprocessing
from tqdm import tqdm
import numpy as np
import re

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
        bbox_file = os.path.join(video_path, "groundtruth.txt")
        visibility_file = os.path.join(video_path, "cover.label")
        absence_file = os.path.join(video_path, "absence.label")
        cut_by_image_file = os.path.join(video_path, "cut_by_image.label")
        meta_info_file = os.path.join(video_path, "meta_info.ini")

        if not os.path.exists(bbox_file) or not os.path.exists(video_path):
            print(f"Skipping video {video_folder} (missing groundtruth.txt or image folder)")
            return [], [], [], []

        bboxes = load_txt_file(bbox_file, dtype=float)
        visibility = load_txt_file(visibility_file, dtype=int)
        absence = load_txt_file(absence_file, dtype=int)
        cut_by_image = load_txt_file(cut_by_image_file, dtype=int)

        with open(meta_info_file, "r", encoding="utf-8") as file:
            file_content = file.read()

        # Extracting object_class value and adding it to categories if it is not already there
        object_class = re.search(r"object_class:\s*(.+)", file_content).group(1)
        if object_class not in [cat['name'] for cat in coco_data["categories"]]:
            cat_id = shared_data["category_id"]
            shared_data["category_id"] += 1
            coco_data["categories"].append({"id": cat_id, "name": object_class})
        else:
            matches = [cat['id'] for cat in coco_data["categories"] if cat["name"] == object_class]
            assert len(matches) == 1
            cat_id = matches[0]

        images = []
        annotations = []

        video_id = shared_data["video_id"]
        shared_data["video_id"] += 1
        track_id = shared_data["track_id"]
        shared_data["track_id"] += 1

        first_img = cv2.imread(os.path.join(video_path, os.listdir(video_path)[0]))

        videos = [{
            "id": video_id,
            "name": video_folder,
            "width": first_img.shape[1],
            "height": first_img.shape[0],
            "frame_range": 1,
            "metadata": {
                "dataset": "GOT10K",
                "user_id": "user",
                "username": "user",
            },
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        }]

        tracks = [{
            "id": track_id,
            "category_id": cat_id,
            "video_id": video_id
        }]


        image_files = [f for f in os.listdir(video_path) if f.lower().endswith(('.jpg', '.png'))]
        for idx, image_name in enumerate(sorted(image_files)):

            img_id = shared_data["image_id"]
            shared_data["image_id"] += 1

            # Add image metadata
            images.append({
                "id": img_id,
                "video": video_folder,
                "file_name": os.path.join(image_root_path, video_folder, image_name),
                "width": first_img.shape[1],
                "height": first_img.shape[0],
                "frame_index": idx,
                "license": 0,
                "video_id": video_id,
                "frame_id": idx, # only unique with a video
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
                #"text": text
            })

            # Add annotation (bbox format: [x, y, width, height]) if it is not occluded
            if absence[idx] == 0 and visibility[idx] > 3:
                x, y, w, h = bboxes[idx]

                # Lock and update annotation ID
                ann_id = shared_data["annotation_id"]
                shared_data["annotation_id"] += 1

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "video_id": video_id,
                    "instance_id": 1, # Just a single object
                    "scale_category": "moving-object",
                    "track_id": track_id,
                    "segmentation": [],
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
            #else:
                #if video_folder == "GOT-10k_Val_000021":
                    #print(f"skipping image {idx + 1} because of occlusion")

        return images, annotations, videos, tracks

    except Exception as e:
        print(f"Error processing video {video_folder}: {e}")
        return [], [], [], []


if __name__ == "__main__":

    data_dir = "../data/got10k/train"
    annotations_dir = "../data/got10k/annotations"
    image_root_path = "train"

    video_folders = os.listdir(data_dir)
    video_folders = list(filter(lambda x: os.path.isdir(os.path.join(data_dir, x)), video_folders)) # Filter out non-folders

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
    with open(f"{annotations_dir}/got_coco_train.json", "w") as json_file:
        json.dump(coco_data, json_file, indent=4)

    print("COCO dataset saved as coco_dataset.json")
