# Annotation format for HTD dataset


The annotations folder is structured as follows:

```
├── annotations
    ├── classes.txt
    ├── hard_tracks_dataset_coco_test.json
    ├── hard_tracks_dataset_coco_val.json
    ├── hard_tracks_dataset_coco.json
    ├── hard_tracks_dataset_coco_class_agnostic.json
```

Details about the annotations:
- `classes.txt`: Contains the list of classes in the dataset. Useful for Open-Vocabulary tracking.
- `hard_tracks_dataset_coco_test.json`: Contains the annotations for the test set.
- `hard_tracks_dataset_coco_val.json`: Contains the annotations for the validation set.
- `hard_tracks_dataset_coco.json`: Contains the annotations for the entire dataset.
- `hard_tracks_dataset_coco_class_agnostic.json`: Contains the annotations for the entire dataset in a class-agnostic format. This means that there is only one category namely "object" and all the objects in the dataset are assigned to this category.


The HTD dataset is annotated in COCO format. The annotations are stored in JSON files, which contain information about the images, annotations, categories, and other metadata.
The format of the annotations is as follows:


````python
{
    "images" : [image],
    "videos": [video],
    "tracks": [track],
    "annotations" : [annotation],
    "categories": [category]
}

image: {
    "id": int,
    "video_id": int,
    "file_name": str,
    "width": int,
    "height": int,
    "frame_index": int,
    "frame_id": int
}
        
video: {
    "id": int,
    "name": str,
    "width": int,
    "height": int,
    "neg_category_ids": [],
    "not_exhaustive_category_ids": []
}
        
track: {
    "id": int,
    "category_id": int,
    "video_id": int
}
        
category: {
    "id": int,
    "name": str,
    "synset": "unknown",
    "frequency": "r" or "b",
}
        
annotation: {
    "id": int,
    "image_id": int,
    "video_id": int,
    "track_id": int,
    "bbox": [x,y,width,height],
    "area": float,
    "category_id": int
}
````