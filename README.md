# PoseTrack21
PoseTrack21 is a [...]

## How to get the dataset?
In order to obtain the entire dataset, please run the following command:
```
python3 download_dataset.py --save_path /target/root/path/of/the/dataset
```

## Structure of the dataset 
The dataset is organized as follows: 

    .
    ├── images                              # contains all images  
        ├── train
        ├── val
    ├── posetrack_data                      # contains annotations for pose reid tracking
        ├── train
            ├── 000001_bonn_train.json
            ├── ...
        ├── val
            ├── ...
    ├── posetrack_mot                       # contains annotations for multi-object tracking 
        ├── mot
            ├── train
                ├── 000001_bonn_train
                    ├── image_info.json
                    ├── gt
                        ├── gt.txt          # ground truth annotations in mot format
                        ├── gt_kpts.txt     # ground truth poses for each frame
                ├── ...
            ├── val
    ├── posetrack_person_search             # person search annotations
        ├── query.json
        ├── train.json
        ├── val.json

A detailed description of the respective dataset formats can be found [here](docs/dataset_structure.md)

## Usage 
tbd
### Cite us here 
