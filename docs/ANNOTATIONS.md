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
    "images": [image],
    "videos": [video],
    "tracks": [track],
    "annotations": [annotation],
    "categories": [category]
}

image: {
    "id": int,                            # Unique ID of the image
    "video_id": int,                      # Reference to the parent video
    "file_name": str,                     # Path to the image file
    "width": int,                         # Image width in pixels
    "height": int,                        # Image height in pixels
    "frame_index": int,                   # Index of the frame within the video (starting from 0)
    "frame_id": int                       # Redundant or external frame ID (optional alignment)
    "video": str,                         # Name of the video 
    "neg_category_ids": [int],            # List of category IDs explicitly not present (optional)
    "not_exhaustive_category_ids": [int]  # Categories not exhaustively labeled in this image (optional)
        
video: {
    "id": int,                            # Unique video ID
    "name": str,                          # Human-readable or path-based name
    "width": int,                         # Frame width
    "height": int,                        # Frame height
    "neg_category_ids": [int],            # List of category IDs explicitly not present (optional)
    "not_exhaustive_category_ids": [int]  # Categories not exhaustively labeled in this video (optional)
    "frame_range": int,                   # Number of frames between annotated frames
    "metadata": dict,                     # Metadata for the video    
}
        
track: {
    "id": int,             # Unique track ID
    "category_id": int,    # Object category
    "video_id": int        # Associated video
}
        
category: {
    "id": int,            # Unique category ID
    "name": str,          # Human-readable name of the category
}
        
annotation: {
    "id": int,                    # Unique annotation ID
    "image_id": int,              # Image/frame ID
    "video_id": int,              # Video ID
    "track_id": int,              # Associated track ID
    "bbox": [x, y, w, h],         # Bounding box in absolute pixel coordinates
    "area": float,                # Area of the bounding box
    "category_id": int            # Category of the object
    "iscrowd": int,               # Crowd flag (from COCO)
    "segmentation": [],           # Polygon-based segmentation (if available)
    "instance_id": int,           # Instance index with a video
    "scale_category": str         # Scale type (e.g., 'moving-object')
}
````
