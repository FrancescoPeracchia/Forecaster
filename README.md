# Forecaster
=======
## Introduction


<details>
  <summary>DATASET</summary>


  ##### Folder structure

```
F2F
├── configs
├── data
|   ├── KITTI   
|   └── RAW
├── dataset
├── inference
├── model
├── PS
├── runs
└── tools
├── PS
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
├── model
├── PS
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








=======
  
  ## Heading
  1. Download Raw Kitti 
  2. Extract images
     * With some
     * Sub bullets
</details>




<details>
  <summary>MODEL</summary>
  
  ## Heading
  1. Download Raw Kitti 
  2. Extract images
     * With some
     * Sub bullets
  
  
  ## Heading
   1. Download Raw Kitti 
  2. Extract images
     * With some
     * Sub bullets
</details>
