# PoseTrack21
Current research evaluates person search, multi-object tracking and multi-person pose estimation as distinct tasks
and on distinct datasets. Though, these tasks are very akin to each other and comprise similar building blocks, 
i.e. person detection or appearance-based association of detected objects. 
Consequently, approaches on these respective tasks are eligible to complement each other.
Real-world scenarios such as surveillance or sports-analysis yield challenging scenarios with a lot of occlusions
in human crowds or by obstacles. Especially bounding-box based approaches such as MOT or person search, which
rely on bounding box information, are prone to errors in heavily occluded scenes. Human pose annotations, one the
other hand, provide structural knowledge, which is very helpful for an disentanglement of person-related features
and background or occlusion noise.
Therefore, we introduce PoseTrack21, a large-scale dataset for person search, multi-object tracking and multi-
person pose (re-id) tracking in real-world scenarios. With PoseTrack21, we want to 1) provide possibilities to analyse
weaknesses of current state-of-the-art approaches on different tasks and 2) want to encourage future researchers
to work on joint approaches, that perform reasonably well
on all three tasks.

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
