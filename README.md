# PoseTrack21
Current research evaluates person search, multi-object tracking and multi-person pose estimation as separate tasks and on different datasets although these tasks are very akin to each other and comprise similar sub-tasks, e.g. person detection or appearance-based association of detected persons. Consequently, approaches on these respective tasks are eligible to complement each other. Therefore, we introduce PoseTrack21, a large-scale dataset for person search, multi-object tracking and multi-person pose tracking in real-world scenarios with a high diversity of poses. The dataset provides rich annotations like human pose annotations including annotations of joint occlusions, bounding box annotations even for small persons, and person-ids within and across video sequences. The dataset allows to evaluate multi-object tracking and multi-person pose tracking jointly with person re-identification or exploit structural knowledge of human poses to improve person search and tracking, particularly in the context of severe occlusions. With PoseTrack21, we want to encourage researchers to work on joint approaches that perform reasonably well on all three tasks.        

## How to get the dataset?
In order to obtain the entire dataset, please contact **posetrack21[at]googlegroups[dot]com**.

Afterwards, please run the following command with you access token:
```
python3 download_dataset.py --save_path /target/root/path/of/the/dataset --token [your token]
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

A detailed description of the respective dataset formats can be found [here](doc/dataset_structure.md).

## Usage 
tbd

## Citation 
```
@inproceedings{doering22,
  title={PoseTrack21: A Dataset for Person Search, Multi-Object Tracking and Multi-Person Pose Tracking},
  author={Andreas Doering and Di Chen and Shanshan Zhang and Bernt Schiele and Juergen Gall},
  booktitle={CVPR},
  year={2022}
}
```
