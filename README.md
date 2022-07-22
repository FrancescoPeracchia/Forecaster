# Forecaster
## Introduction


<details>
  <summary>Dataset</summary>


  ##### Folder structure

```
F2F
├── configs
├── data
|   ├── KITTI   
|   └── RAW
├── dataset
├── inference
├── models
├── runs
└── tools

```
  
  ##### Download Raw Kitti
    
    In order to download the entire RAW DATA of KITTI is possible use a script download the zip and create a predefined data structure from each video-clip.

```bash
python tools/download_kitti.py path-to-kitti.text download-folder
```

```
F2F
├── configs
├── data
|   ├── KITTI   
|   └── RAW
|        ├── processed
|        └── zip
|
├── dataset
├── inference
├── models
├── runs
└── tools
```


  ##### Extract images
  
    e.g: "zip" should be populated with the zip folders, keep a new folder "processed" for clips wanted to be processed


    percentage : splitting rate TRAIN VALIDATION TEST



  


```bash
python tools/convert_kitti_raw.py download-folder dataset-folder --resize False --percentage 0.7 0.2 0.1
```


```
F2F
├── configs
├── data
|   ├── KITTI 
|   |   ├── output
|   |   ├── test
|   |   ├── training
|   |   ├── validation
|   |   ├── test.json
|   |   ├── train.json
|   |   └── validation.json
|   | 
|   └── RAW
|        └── processed
|        └── zip
├── dataset
├── inference
├── models
├── runs
└── tools
```



</details>

<details>
<summary>Training & Testing</summary>

  ##### train structure

``` bash 
python train.py /home/fperacch/Forecaster/configs/3DConv_00_20.py /home/fperacch/Forecaster/saved 
```
  
  ##### Download Raw Kitti
    
    In order to download the entire RAW DATA of KITTI is possible use a script download the zip and create a predefined data structure from each video-clip.

```bash
python tools/download_kitti.py path-to-kitti.text download-folder
```

```
F2F
├── configs
├── data
|   ├── KITTI   
|   └── RAW
|        └── processed
├── dataset
├── inference
├── models
├── runs
└── tools
```


  ##### Extract images
  

```bash
python tools/convert_kitti_raw.py download-folder dataset-folder --resize False --percentage 0.7 0.2 0.1
```

```
F2F
├── configs
├── data
|   ├── KITTI 
|   |   ├── output
|   |   ├── test
|   |   ├── training
|   |   ├── validation
|   |   ├── test.json
|   |   ├── train.json
|   |   └── validation.json
|   | 
|   └── RAW
|        └── processed
├── dataset
├── inference
├── model
├── PS
├── runs
└── tools
```

</details>



conda init bash
source ~/.bashrc
tmux kill-session -t ostechnix(numeber)