import json
import numpy as np
import os


def load_json(json_path):
    """ Load JSON file """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def print_categories(annos):

    print()
    print(f"Number of categories: {len(annos['categories'])}")
    print(f"Number of unique categories: {len(np.unique([cat['name'] for cat in annos['categories']]))}")
    print("=" * 50)

    all_names = set()
    duplicated_names = []
    for cat in annos["categories"]:
        if cat["name"] not in all_names:
            all_names.add(cat["name"])
        else:
            duplicated_names.append(cat["name"])

    partly_duplicated_names = []
    for cat1 in annos["categories"]:
        for cat2 in annos["categories"]:
            if cat1["name"] != cat2["name"] and cat1["name"] in cat2["name"]:
                partly_duplicated_names.append(cat1["name"])
                break

    for cat in annos["categories"]:
        print(
            cat["id"],
            "\t", "X" if cat["name"] in duplicated_names else " ",
            "\t", "O" if cat["name"] in partly_duplicated_names else " ",
            "\t", cat["name"]
        )

def clean_categories(annos, cats_to_merge=None):
    cat_id_mapping = {}

    cats_by_name = {}
    for cat in annos["categories"]:
        if cat["name"] not in cats_by_name:
            cats_by_name[cat["name"]] = []
        cats_by_name[cat["name"]].append(cat)

    # remove duplicated names (join their ids)
    for cat in annos["categories"]:
        cat_id_mapping[cat["id"]] = cats_by_name[cat["name"]][0]["id"]

    if cats_to_merge is not None:
        for cats in cats_to_merge:
            cat_ids = [cat["id"] for cat in annos["categories"] if cat["name"] in cats]
            first_cat_id = cat_ids[0]
            for cat_id in cat_ids[1:]:
                cat_id_mapping[cat_id] = first_cat_id


    # Update the category ids in the annotations
    for annotation in annos["annotations"]:
        annotation["category_id"] = cat_id_mapping[annotation["category_id"]]

    # Update the category ids in the tracks
    for track in annos["tracks"]:
        track["category_id"] = cat_id_mapping[track["category_id"]]

    # Remove categories
    annos["categories"] = [cat for cat in annos["categories"] if cat["id"] in cat_id_mapping.values()]
    



    # Re enumerate the category ids ------------------------
    cat_id_mapping = {}
    for i, cat in enumerate(annos["categories"]):
        cat_id_mapping[cat["id"]] = i + 1

    # Update the category ids in the annotations
    for annotation in annos["annotations"]:
        annotation["category_id"] = cat_id_mapping[annotation["category_id"]]

    # Update the category ids in the tracks
    for track in annos["tracks"]:
        track["category_id"] = cat_id_mapping[track["category_id"]]

    # Update the category ids in the categories
    for cat in annos["categories"]:
        cat["id"] = cat_id_mapping[cat["id"]]


    # simple test
    # find annotation in the video which cntains chicken in the name
    videos_by_id = {video["id"]: video for video in annos["videos"]}
    for annotation in annos["annotations"]:
        if "chicken" in videos_by_id[annotation["video_id"]]["name"]:
            print(annotation["id"], annotation["category_id"], annotation["video_id"])
            break

    # print resulting categories
    print()
    print(f"Number of categories after cleaning: {len(annos['categories'])}")
    print("=" * 50)
    for cat in annos["categories"]:
        print(cat["id"], cat["name"])

    # Print labels as tuple
    print()
    tuple_string = "("
    for cat in annos["categories"]:
        tuple_string += f"'{cat['name']}', "
    tuple_string = tuple_string[:-2] + ")"
    print(f"Labels as tuple: {tuple_string}")

    return annos

def create_class_agnostic_annotations(annos):
    """ Create class-agnostic annotations """
    # set all category ids to 1
    for annotation in annos["annotations"]:
        annotation["category_id"] = 1

    # set all category ids to 1
    annos["categories"] = [{
        "id": 1,
        "name": "object"
    }]

    # set all category ids to 1
    for track in annos["tracks"]:
        track["category_id"] = 1

    return annos


if __name__ == "__main__":

    # Load the annotations
    annotations_dir = "/srv/beegfs02/scratch/visobt4s/data/tracking_datasets/hard_tracks_dataset2/annotations"
    annos = load_json(os.path.join(annotations_dir, "hard_tracks_dataset_coco.json"))
    print_categories(annos)

    # remove all fields but name and id from categories
    annos["categories"] = [
        {
            "id": cat["id"],
            "name": cat["name"]
        }
        for cat in annos["categories"]
    ]

    print()
    print(f"Stats before cleaning:")
    print(f"Number of categories: {len(annos['categories'])}")
    print(f"Number of videos: {len(annos['videos'])}")
    print(f"Number of images: {len(annos['images'])}")
    print(f"Number of annotations: {len(annos['annotations'])}")
    print(f"Number of tracks: {len(annos['tracks'])}")



    # Clean the categories
    cats_to_merge = [
        ["car", "car_(automobile)", "car_(vehicle)"],
        ["bear", "bear cub"],
        ["person", "human", "pedestrian"]
    ]

    annos = clean_categories(annos, cats_to_merge)

    print()
    print(f"Stats after cleaning:")
    print(f"Number of categories: {len(annos['categories'])}")
    print(f"Number of videos: {len(annos['videos'])}")
    print(f"Number of images: {len(annos['images'])}")
    print(f"Number of annotations: {len(annos['annotations'])}")
    print(f"Number of tracks: {len(annos['tracks'])}")

    # Save the cleaned annotations
    with open(os.path.join(annotations_dir, "hard_tracks_dataset_coco_cleaned.json"), 'w') as f:
        json.dump(annos, f)

    # Create class-agnostic annotations
    annos = create_class_agnostic_annotations(annos)

    print()
    print(f"Stats after class agnostic transformation:")
    print(f"Number of categories: {len(annos['categories'])}")
    print(f"Number of videos: {len(annos['videos'])}")
    print(f"Number of images: {len(annos['images'])}")
    print(f"Number of annotations: {len(annos['annotations'])}")
    print(f"Number of tracks: {len(annos['tracks'])}")

    # Save the class-agnostic annotations
    with open(os.path.join(annotations_dir, "hard_tracks_dataset_coco_class_agnostic.json"), 'w') as f:
        json.dump(annos, f)



