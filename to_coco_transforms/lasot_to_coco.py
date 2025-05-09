import os
import json
import cv2
import multiprocessing

from PIL import Image
from tqdm import tqdm
import numpy as np

# Define COCO format structure
coco_data = {
    "tracks": [],
    "videos": [],
    "images": [],
    "annotations": [],
    "categories": []
}

def process_video(data_dir, image_root_path, video_folder, shared_data):
    """Processes a single video folder and returns image & annotation data."""

    #print(f"Processing video {video_folder}")

    try:
        video_path = os.path.join(data_dir, video_folder, "img")
        gt_path = os.path.join(data_dir, video_folder, "groundtruth.txt")
        text_path = os.path.join(data_dir, video_folder, "nlp.txt")
        occlusion_path = os.path.join(data_dir, video_folder, "full_occlusion.txt")
        out_of_view_path = os.path.join(data_dir, video_folder, "out_of_view.txt")

        if not os.path.exists(gt_path) or not os.path.exists(video_path):
            print(f"Skipping video {video_folder} (missing groundtruth.txt or image folder)")
            return [], [], [], []

        with open(gt_path, "r") as f:
            bboxes = [list(map(int, line.strip().split(','))) for line in f.readlines()]

        if not os.path.exists(text_path):
            text = ""
        else:
            with open(text_path, "r") as f:
                text = f.read()

        occlusion = np.loadtxt(occlusion_path, delimiter=',', dtype=int)

        if not os.path.exists(out_of_view_path):
            out_of_view = np.zeros(len(occlusion))
        else:
            out_of_view = np.loadtxt(out_of_view_path, delimiter=',', dtype=int)


        # Extracting object_class value and adding it to categories if it is not already there
        object_class = video_folder.split("-")[0]
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


        #with shared_data["lock"]:
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
                "dataset": "LaSOT",
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


        image_paths = sorted([i for i in os.listdir(video_path) if i.endswith('.jpg') or i.endswith('.png')])
        for idx, image_name in enumerate(image_paths):

            img_id = shared_data["image_id"]
            shared_data["image_id"] += 1

            # Add image metadata
            images.append({
                "id": img_id,
                "video": video_folder,
                "file_name": os.path.join(image_root_path, video_folder, "img", image_name),
                "width": first_img.shape[1],
                "height": first_img.shape[0],
                "frame_index": idx,
                "license": 0,
                "video_id": video_id,
                "frame_id": idx, # only unique with a video
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
                "text": text
            })

            # Add annotation (bbox format: [x, y, width, height]) if it is not occluded
            if occlusion[idx] == 0 and out_of_view[idx] == 0:
                x, y, w, h = bboxes[idx]


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
                #print(f"skipping image {idx} because of occlusion")

        return images, annotations, videos, tracks

    except Exception as e:
        raise e
        print(f"Error processing video {video_folder}: {e}")
        return [], [], [], []


if __name__ == "__main__":

    data_dir = "../data/lasot/data"
    annotations_dir = "../data/lasot/annotations"
    image_root_path = "airplane"
    dataset_root_dir = "../data/lasot"

    video_folders = os.listdir(data_dir)

    #with open(f"{dataset_root_dir}/testing_set.txt", "r") as f:
    #    video_folders = f.read().splitlines()

    with open(f"{dataset_root_dir}/training_set.txt", "r") as f:
        video_folders = f.read().splitlines()

    print(f"Processing {len(video_folders)} videos")

    # Use Manager to share counters across processes
    #with multiprocessing.Manager() as manager:
    shared_data = {
        "image_id": 1,
        "annotation_id": 1,
        "video_id": 1,
        "track_id": 1,
        "category_id": 1,
    }

    for vf in tqdm(video_folders):
        results = process_video(data_dir, image_root_path, vf, shared_data)
        coco_data["images"].extend(results[0])
        coco_data["annotations"].extend(results[1])
        coco_data["videos"].extend(results[2])
        coco_data["tracks"].extend(results[3])

    # Save to JSON
    with open(f"{annotations_dir}/lasot_coco_train.json", "w") as json_file:
        json.dump(coco_data, json_file, indent=4)

    print("COCO dataset saved as coco_dataset.json")
