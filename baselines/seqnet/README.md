## SeqNet Baseline 

We adopted the code of SeqNet from the [official repository](https://github.com/serend1p1ty/SeqNet).

### Inference
To run SeqNet, please follow these steps:
1) extract the model checkpoint to `./exps/exp_posetrack/epoch_5.pth`. You can find a zip-file in the same folder. Due to file size limits on github, you need to merge the zip files beforehand following:
 ```
cat epoch_5.pth.z01 epoch_5.pth.z02 epoch_5.pth.z03 epoch_5.pth.zip > merged.zip
```
 
3) extract the required annotation files to `./data/PoseTrackPSearch/posetrack_data/person_search_annotations` 
4) extract the images to  `./data/PoseTrackPSearch/posetrack_data/images`

Afterwards, run `bash scripts/eval_person_search.sh` to perform inference. 

If you plan to use the SeqNet-baseline in your work, please don't forget to cite
```
@inproceedings{li2021sequential,
  title={Sequential End-to-end Network for Efficient Person Search},
  author={Li, Zhengjia and Miao, Duoqian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={3},
  pages={2011--2019},
  year={2021}
}
```
