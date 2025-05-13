# HardTracksDataset: A Benchmark for Robust Object Tracking under Heavy Occlusion and Challenging Conditions

[Computer Vision Lab, ETH Zurich](https://vision.ee.ethz.ch/)


<p align="center">
    <img src="./docs/imgs/main.png" alt="Image" width="100%"/>
</p>


## Introduction 
We introduce the HardTracksDataset (HTD), a novel multi-object tracking (MOT) benchmark specifically designed to address two critical limitations prevalent in existing tracking datasets. First, most current MOT benchmarks narrowly focus on restricted scenarios, such as pedestrian movements, dance sequences, or autonomous driving environments, thus lacking the object diversity and scenario complexity representative of real-world conditions. Second, datasets featuring broader vocabularies, such as, OVT-B and TAO, typically do not sufficiently emphasize challenging scenarios involving long-term occlusions, abrupt appearance changes, and significant position variations. As a consequence, the majority of tracking instances evaluated are relatively easy, obscuring trackers’ limitations on truly challenging cases. HTD addresses these gaps by curating a challenging subset of scenarios from existing datasets, explicitly combining large vocabulary diversity with severe visual challenges. By emphasizing difficult tracking scenarios, particularly long-term occlusions and substantial appearance shifts, HTD provides a focused benchmark aimed at fostering the development of more robust and reliable tracking algorithms for complex real-world situations.

## Results of state of the art trackers on HTD
<table>
  <caption>TETA evaluation of state-of-the-art trackers on the HTD validation and test sets, grouped by tracking approach.</caption>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">Validation</th>
      <th colspan="4">Test</th>
    </tr>
    <tr>
      <th>TETA</th>
      <th>LocA</th>
      <th>AssocA</th>
      <th>ClsA</th>
      <th>TETA</th>
      <th>LocA</th>
      <th>AssocA</th>
      <th>ClsA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="9"><em>Motion-based</em></td>
    </tr>
    <tr>
      <td>ByteTrack</td>
      <td>34.877</td>
      <td>54.624</td>
      <td>19.085</td>
      <td>30.922</td>
      <td>37.875</td>
      <td>56.135</td>
      <td>19.464</td>
      <td>38.025</td>
    </tr>
    <tr>
      <td>DeepSORT</td>
      <td>33.782</td>
      <td>57.350</td>
      <td>15.009</td>
      <td>28.987</td>
      <td>37.099</td>
      <td>58.766</td>
      <td>15.729</td>
      <td>36.803</td>
    </tr>
    <tr>
      <td>OCSORT</td>
      <td>33.012</td>
      <td>57.599</td>
      <td>12.558</td>
      <td>28.880</td>
      <td>35.164</td>
      <td>59.117</td>
      <td>11.549</td>
      <td>34.825</td>
    </tr>
    <tr>
      <td colspan="9"><em>Appearance-based</em></td>
    </tr>
    <tr>
      <td>MASA</td>
      <td>42.246</td>
      <td>60.260</td>
      <td>34.241</td>
      <td>32.237</td>
      <td>43.656</td>
      <td>60.125</td>
      <td>31.454</td>
      <td><strong>39.390</strong></td>
    </tr>
    <tr>
      <td>OV-Track</td>
      <td>29.179</td>
      <td>47.393</td>
      <td>25.758</td>
      <td>14.385</td>
      <td>33.586</td>
      <td>51.310</td>
      <td>26.507</td>
      <td>22.941</td>
    </tr>
    <tr>
      <td colspan="9"><em>Transformer-based</em></td>
    </tr>
    <tr>
      <td>OVTR</td>
      <td>26.585</td>
      <td>44.031</td>
      <td>23.724</td>
      <td>14.138</td>
      <td>29.771</td>
      <td>46.338</td>
      <td>24.974</td>
      <td>21.643</td>
    </tr>
    <tr>
      <td colspan="9"></td>
    </tr>
    <tr>
      <td><strong>MASA+</strong></td>
      <td><strong>42.716</strong></td>
      <td><strong>60.364</strong></td>
      <td><strong>35.252</strong></td>
      <td><strong>32.532</strong></td>
      <td><strong>44.063</strong></td>
      <td><strong>60.319</strong></td>
      <td><strong>32.735</strong></td>
      <td>39.135</td>
    </tr>
  </tbody>
</table>


## Download Instructions

We provide the full dataset with annotations and metadata on HuggingFace:

- [HTD Dataset 🤗](https://huggingface.co/datasets/mscheidl/htd)

To download the dataset you can use the HuggingFace CLI. 
First install the HuggingFace CLI according to the official [HuggingFace documentation](https://huggingface.co/docs/huggingface_hub/main/guides/cli)
and login with your HuggingFace account. Then, you can download the dataset using the following command:

```bash
huggingface-cli download mscheidl/htd --repo-type dataset
```

The dataset is organized in the following structure:

``` 
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

The `data` folder contains the videos, the `annotations` folder contains the annotations in COCO (TAO) format, and the `metadata` folder contains the metadata files for running MASA+. 
If you use HTD independently, you can ignore the `metadata` folder. More information about the annotation format please refer to [ANNOTATIONS.md](docs/ANNOTATIONS.md).


## Installation

If you want to run code from this repository, please first install the dependencies. 
Please refer to [INSTALL.md](docs/INSTALL.md)


## Run Masa+ on HTD

To run MASA+ or other models on HTD please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md)

### Acknowledgments

Our code is built on [mmdetection](https://github.com/open-mmlab/mmdetection), [MASA](https://github.com/siyuanliii/masa), [TETA](https://github.com/SysCV/tet). If you find our work useful, consider checking out their work.
