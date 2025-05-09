import json


def load_json(json_path):
    """ Load JSON file """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":

    annos = load_json("../data/ovtb/ovtb_ann.json")
    coco_data = annos.copy()

    # Curate videos
    for v in coco_data["videos"]:
        v["frame_range"] = 1
        v["metadata"] = {
            "dataset": "OVT-B",
            "user_id": "user",
            "username": "user",
        }

    # Curate images
    video_id_video_dict = {}
    for video in coco_data["videos"]:
        video_id_video_dict[video["id"]] = video

    for img in coco_data["images"]:
        img["license"] = 0
        img["video"] = video_id_video_dict[img["video_id"]]["name"]
        img["neg_category_ids"] = []
        img["not_exhaustive_category_ids"] = []

    # Curate annotations
    ## Create dict from video id to track
    tracks_by_video_dict = {}
    for track in annos["tracks"]:
        video_id = track["video_id"]
        if video_id not in tracks_by_video_dict:
            tracks_by_video_dict[video_id] = []
        tracks_by_video_dict[video_id].append(track)


    # Create dict from track id to instance id
    track_id_to_instance_id_dict = {}
    for video in annos["videos"]:
        video_id = video["id"]

        # get all tracks for this video
        tracks_in_video = tracks_by_video_dict[video_id]

        instance_counter = 1
        for track in tracks_in_video:
            track_id = track["id"]
            if track_id in track_id_to_instance_id_dict:
                raise ValueError(f"Track ID {track_id} already exists in track_id_to_instance_id_dict") # thrown when track id is not unique across all videos

            track_id_to_instance_id_dict[track_id] = instance_counter
            instance_counter += 1

    for anno in coco_data["annotations"]:
        anno["instance_id"] = track_id_to_instance_id_dict[anno["track_id"]]
        anno["iscrowd"] = 0
        anno["scale_category"] = "moving-object"
        anno["segmentation"] = []

    # Save to JSON
    with open(f"../data/ovtb/ovtb_coco_val.json", "w") as json_file:
        json.dump(coco_data, json_file, indent=4)
