import os
import json
import cv2
import multiprocessing
from tqdm import tqdm
import numpy as np
import re
import traceback


def load_json(json_path):
    """ Load JSON file """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data



if __name__ == "__main__":

    annos = load_json("../data/bdd/annotations/box_track_20/box_track_train_cocofmt.json")
    coco_data = {
        "tracks": [],
        "videos": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    images_by_video_dict = {}
    for image in annos["images"]:
        video_id = image["video_id"]
        if video_id not in images_by_video_dict:
            images_by_video_dict[video_id] = []
        images_by_video_dict[video_id].append(image)

    annotation_by_image_dict = {}
    for annotation in annos["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in annotation_by_image_dict:
            annotation_by_image_dict[image_id] = []
        annotation_by_image_dict[image_id].append(annotation)

    global_track_id = 1 # counter

    # adapt annotations
    for existing_video in annos["videos"]:
        video_id, video_name = existing_video["id"], existing_video["name"]

        # Get all images and annotations for this video
        images = images_by_video_dict[video_id]

        # Save new video
        coco_data["videos"].append({
            "id": video_id,
            "name": video_name,
            "width": images[0]["width"],
            "height": images[0]["height"],
            "frame_range": 1,
            "metadata": {
                "dataset": "BDD",
                "user_id": "user",
                "username": "user",
            },
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        })

        # dictionary for local to global track id
        local_to_global_track_id = {}

        # Save new images
        for image in images:

            coco_data["images"].append({
                "id": image["id"],
                "video": video_name,
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
                "frame_index": image["frame_id"],
                "license": 0,
                "video_id": image["video_id"],
                "frame_id": image["frame_id"],
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            })

            # get all annotation (an image can also have no annotation --> occlusion)
            annotations = annotation_by_image_dict[image["id"]] if image["id"] in annotation_by_image_dict else []

            for annotation in annotations:

                # the dict "local_to_global_track_id" is recreated for each video and in each video the instance id is unique
                key = annotation["instance_id"]
                if key not in local_to_global_track_id:
                    coco_data["tracks"].append({
                        "id": global_track_id,
                        "category_id": annotation["category_id"],
                        "video_id": video_id
                    })
                    local_to_global_track_id[key] = global_track_id
                    global_track_id += 1

                coco_data["annotations"].append({
                    "id": annotation["id"],
                    "image_id": annotation["image_id"],
                    "video_id": video_id,
                    "instance_id": annotation["instance_id"], # Unique within a video
                    "scale_category": "moving-object",
                    "track_id": local_to_global_track_id[key],
                    "segmentation": [],
                    "category_id": annotation["category_id"],
                    "bbox": annotation["bbox"],
                    "area": annotation["area"],
                    "iscrowd": annotation["iscrowd"]
                })

    coco_data["categories"] = annos["categories"] # Stays the same

    # save
    with open("../data/bdd/annotations/bdd_coco_train.json", 'w') as f:
        json.dump(coco_data, f)