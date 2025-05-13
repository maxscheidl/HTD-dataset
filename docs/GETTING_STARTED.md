
# Testing on HTD

This document describes how to run MASA+ on HTD. This guide generalizes to other trackers as well.


## 1. Download and setup HTD

You can download HTD using one of the following links:
- [HuggingFace](https://huggingface.co/datasets/mscheidl/htd)


Please download the dataset and unzip it under the `data` folder. It is recommended to create a folder named `htd` under the `data` folder. 
Then move the `metadata` folder from the `htd` folder to the `data` folder. The `data` folder structure should look like this:
``` 
├── data
    ├── htd
        ├── data
            ├── AnimalTrack
            ├── BDD
            ├── ...
        ├── annotations
            ├── classes.txt
            ├── hard_tracks_dataset_coco_test.json
            ├── hard_tracks_dataset_coco_val.json
            ├── ...
    ├── metadata
            ├── lvis_v1_clip_a+cname.npy
            ├── lvis_v1_train_cat_info.json
```

## 2. Download masa weights

You can download the pretrained weights for masa from [HF 🤗](https://huggingface.co/dereksiyuanli/masa/resolve/main/detic_masa.pth) and place it in the `saved_models/masa_models` folder. 


## 3. Setup Detic

You can download the pretrained weights for Detic from [here](https://download.openmmlab.com/mmdetection/v3.0/detic/detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis/detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis-ec91245d.pth) and place it under the `saved_models/pretrained_weights` folder.

Your final folder structure should look like this:

```
├── masa
├── projects
├── tools
├── configs
├── results
├── data
    ├── htd
        ├── data
            ├── AnimalTrack
            ├── BDD
            ├── ...
        ├── annotations
            ├── classes.txt
            ├── hard_tracks_dataset_coco_test.json
            ├── hard_tracks_dataset_coco_val.json
            ├── ...
    ├── metadata
            ├── classes.txt
            ├── hard_tracks_dataset_coco_test.json
|── saved_models 
    ├── pretrain_weights
        ├── detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis-ec91245d.pth
    ├── masa_models
        ├── detic_masa.pth
├── ... # Other folders
```

## 4. Run MASA+ 

This codebase is inherited from [mmdetection](https://github.com/open-mmlab/mmdetection).
You can refer to the [offical instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md).
You can also refer to the short instructions below.
We provide config files in [configs](../configs).

To run MASA+ on HTD validation set, you can use the following command:

```angular2html
tools/dist_test.sh configs/masa-detic/hard_tracks_dataset/masa_detic_hard_tracks_dataset_val_double_avg.py saved_models/masa_models/detic_masa.pth 4
```




### Test other models with COCO-format

Note that, in this repo, the evaluation metrics are computed with COCO-format.

- single GPU
- single node multiple GPU

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--cfg-options]

# multi-gpu testing
tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--cfg-options]
```

Optional arguments:
- `--cfg-options`: If specified, some setting in the used config will be overridden.

For other models, you can replace the `${CONFIG_FILE}` and `${CHECKPOINT_FILE}` with the corresponding paths.






