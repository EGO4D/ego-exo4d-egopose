# Ego-Exo4D Hand Ego Pose Baseline Model 
Implementation of hand-ego-pose-potter, a 3D hand pose estimation baseline model based on [POTTER](https://github.com/zczcwh/POTTER/tree/main) in [Ego-Exo4D](https://github.com/facebookresearch/Ego4d) hand ego pose benchmark.


## Data preparation
Follow instructions [here](https://github.com/EGO4D/ego-exo4d-egopose/tree/main/handpose/data_preparation) to get:
- ground truth annotation files in `$gt_output_dir/annotation/manual` or `$gt_output_dir/annotation/auto` if using automatic annotations,
referred as `gt_anno_dir` below
- corresponding undistorted Aria images in `$gt_output_dir/image/undistorted`, 
referred as `aria_img_dir` below

## Setup

- Follow the instructions below to set up the environment for model training and inference.
```
conda create -n potter_hand_pose python=3.9.16 -y
conda activate potter_hand_pose
pip install -r requirement.txt
```
- Install [pytorch](https://pytorch.org/get-started/previous-versions/). The model is tested with `pytorch==2.1.0` and `torchvision==0.16.0`. 


## Training

Download POTTER_cls model weights from [here](https://github.com/zczcwh/POTTER/tree/main/image_classification#2-poolattnformer-models-in-paper-we-denote-as-potter_cls). Put all model weight at `${ROOT}/output/ckpt` directory.  
Specifically, download model `poolattnformer_s12` and rename it to `cls_s12.pth`

Run command below to perform training on manual data with pretrained POTTER_cls weight:
```
python3 train.py \
    --gt_anno_dir <gt_anno_dir> \
    --aria_img_dir <aria_img_dir>
```
If choose to finetuning on manual data with pretrained weight on automatic data, set `pretrained_ckpt` to be the path of pretrained hand-ego-pose-potter model weight.

## Inference

Download pretrained ([EvalAI baseline](https://eval.ai/web/challenges/challenge-page/2249/overview)) model weight of hand-ego-pose-potter from [here](https://drive.google.com/drive/folders/1WSvV7wvmYBvFhB5KwK6PRXwV5dpHd9Hf?usp=sharing).

Run command below to perform inference of pretrained model on test set, and save the inference output as a single JSON file. It will be stored at `output/inference_output` by default. 
```
python3 inference.py \
    --pretrained_ckpt <pretrained_ckpt> \
    --gt_anno_dir <gt_anno_dir> \
    --aria_img_dir <aria_img_dir>
```

The output format is: 
```
{
    "<take_uid>": {
        "<frame_number>": {
                "left_hand_3d": [],
                "right_hand_3d": []     
        }
        ...
    }
    ...
}
```

## Note
For the 21 keypoints annotation in each hand, its index and label are listed as below:
```
{0: Wrist,
 1: Thumb_1, 2: Thumb_2, 3: Thumb_3, 4: Thumb_4,
 5: Index_1, 6: Index_2, 7: Index_3, 8: Index_4,
 9: Middle_1, 10: Middle_2, 11: Middle_3, 12: Middle_4,
 13: Ring_1, 14: Ring_2, 15: Ring_3, 16: Ring_4,
 17: Pinky_1, 18: Pinky_2, 19: Pinky_3, 20: Pinky_4}
```
The 21 keypoints for right hand are visualized below, with left hand has symmetric keypoints position. 

<img src="assets/hand_index.png" width ="350" height="400">