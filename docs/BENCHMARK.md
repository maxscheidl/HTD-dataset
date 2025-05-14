
# Benchmark on HTD

This document describes how to run MASA+ on HTD. This guide generalizes to other trackers as well.

## 1. Install dependencies

Please refer to the [INSTALL.md](docs/INSTALL.md) for installation instructions.

## 2. Download and setup HTD

We provide the full dataset with annotations and metadata on HuggingFace:

- [HTD Dataset 🤗](https://huggingface.co/datasets/mscheidl/htd)

To download the dataset you can use the HuggingFace CLI. 
First install the HuggingFace CLI according to the official [HuggingFace documentation](https://huggingface.co/docs/huggingface_hub/main/guides/cli)
and login with your HuggingFace account. Then, you can download the dataset using the following command:

```bash
huggingface-cli download mscheidl/htd --repo-type dataset --local-dir htd
```

The video folders are provided as zip files. Before usage please unzip the files. You can use the following command to unzip all files in the `data` folder.
Please note that the unzipping process can take a while (especially for _TAO.zip_)

```bash
cd htd
for z in data/*.zip; do (unzip -o -q "$z" -d data && echo "Unzipped: $z") & done; wait; echo "✅ Done"
mkdir -p data/zips        # create a folder for the zip files
mv data/*.zip data/zips/  # move the zip files to the zips folder
```

After downloading the dataset create a folder called `data` in the root directory of the project and place (or symlink) the dataset there.
Then move the `metadata` folder from the `htd` folder to the `data` folder. 

```bash
mv metadata ../
```


The `data` folder structure should look like this:

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

## 3. Download masa weights

You can download the pretrained weights for masa from [HF 🤗](https://huggingface.co/dereksiyuanli/masa/resolve/main/detic_masa.pth) and place it in the `saved_models/masa_models` folder. 


## 4. Setup Detic

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
            ├──  lvis_v1_clip_a+cname.npy
            ├──  lvis_v1_train_cat_info.json
|── saved_models 
    ├── pretrain_weights
        ├── detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis-ec91245d.pth
    ├── masa_models
        ├── detic_masa.pth
├── ... # Other folders
```

## 5. Run MASA+ 

This codebase is inherited from [mmdetection](https://github.com/open-mmlab/mmdetection).
You can refer to the [offical instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md).
You can also refer to the short instructions below.
We provide config files in [configs](../configs).

To run MASA+ on HTD validation set, you can use the following command for single GPU testing:

```shell
python tools/test.py configs/masa-detic/hard_tracks_dataset/masa_detic_hard_tracks_dataset_val_double_avg.py saved_models/masa_models/detic_masa.pth
````

For multi-GPU testing, you can use the following command (e.g. 4 GPUs):

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






