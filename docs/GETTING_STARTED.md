
# Testing on HTD

This document describes how to run Double-MASA on HTD. This guide generalizes to other trackers as well.


## 1. Download HTD [TODO]

You can download HTD using one of the following links:
- [Google Drive](https://drive.google.com/file/d/1q0g2vXk4x5j6b3r7m8z9c4f8e4f8e4f/view?usp=sharing)
- [OneDrive](https://1drv.ms/u/s!Aq4g2vXk4x5j6b3r7m8z9c4f8e4f?e=0f8e4f)

Please download the dataset and unzip it under the `data` folder. It is recommended to create a folder named `htd` under the `data` folder. The folder structure should look like this:

```
├── masa
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

|── saved_models # saved_models are the folder to save downloaded pretrained models and also the models you trained.
    ├── pretrain_weights
    ├── masa_models
```

## 2. Download masa weights [TODO]

You can download the pretrained weights for masa from [here](https://huggingface.co/dereksiyuanli/masa/resolve/main/masa_weights.zip) and unzip it under the `saved_models` folder. 


## 3. Setup Detic [TODO]

You can download the pretrained weights for Detic from [here](https://huggingface.co/dereksiyuanli/detic/resolve/main/detic_weights.zip) and unzip it under the `pretrained_weights` folder.

Then also create a folder named `metadata` under the `data` folder. Then download the CLIP embeddings and the class infos from here [here](https://huggingface.co/dereksiyuanli/detic/resolve/main/metadata.zip) and unzip it under the `metadata` folder.


## 4. Run Double-MASA [TODO]

This codebase is inherited from [mmdetection](https://github.com/open-mmlab/mmdetection).
You can refer to the [offical instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md).
You can also refer to the short instructions below.
We provide config files in [configs](../configs).

To run Double-MASA on HTD validation set, you can use the following command:

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






