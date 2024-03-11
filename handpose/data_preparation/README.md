# Hand Ego Pose Data Preparation
Data preparation for hand ego-pose benchmark task in [Ego-Exo4D](https://github.com/facebookresearch/Ego4d/tree/main), [paper](https://arxiv.org/abs/2311.18259).

## Getting Started
Follow instruction below to first install necessary packages, and then follow step 0 to step 3 to generate the hand ego pose data, including undisorted Aria images and corresponding 2D & 3D hand pose annotaton JSON files.  
A general comment: list the path used (ego4d-out-dir, gt-output-dir) here and explain a bit. 

### Set up
Run command below to install necessary packages.
```
pip install -r requirement.txt
```

### Step 0: Download Ego-Exo4D data
To prepare the data needed for hand ego pose estimation, you need to first download the data via [Ego-Exo4D Downloader](https://docs.ego-exo4d-data.org/download/) including `annotations`, `metadata`, frame aligned videos and VRS files. 

First, run command below to download ego pose related `annotations` and `metadata`:

```
egoexo -o <ego4d-out-dir> --parts annotations metadata --benchmarks egopose
```

Then, run command below to download frame aligned videos for only annotated takes:

```
comment: cd handpose/data_preparation/
python3 scripts/download.py \
    --output_dir <ego4d-out-dir> \
    --parts takes
```

Finally, for VRS files (which is needed to generate aria calibration JSON file), you can either choose to follow option 1: download `take_vrs` or option 2: skip `take_vrs` download and get pre-generated aria calibration JSON file.  
comment: for 333 takes with the annotations, only 184 take videos are downloaded?

*NOTE: If you choose to followw option 2, please check if the the calibration file exists for every take as this list might not be up-to-date.*

#### Option 1: Download `take_vrs`
Run command below to download `take_vrs` for all annotated takes:
```
python3 scripts/download.py \
    --output_dir <ego4d-out-dir> \
    --parts take_vrs
```

#### Option 2: Download Aria calibration JSON file
Download pre-generated Aria calibration file from [here](https://drive.google.com/drive/folders/1qPAzPbRdx65-UhIWGoFUGPrgM4JLpwUO?usp=sharing) and put it under `<gt-output-dir>/aria_calib_json/`, where `<gt-output-dir>` is hand ego pose data preparation output directory.  
comment: make it a zip


### Step 1: Generate Aria calibration JSON file
If you follow option 1 in downloading VRS file, run command below to generate Aria calibration JSON files which will be used later for Aria image undistortion. If you follow option 2, you may skip this step.
```
python3 main.py \
    --steps aria_calib
    --ego4d_data_dir <ego4d-out-dir> \
    --gt_output_dir <gt-output-dir>
```
comment: TODO: double check for the data folder of hand annotations  
comment: from downloading the metadata, it is actually splits.json, not egoexo_split_latest_train_val_test.csv?  
comment: we need to hide the test_private after the data is publicly released. 

### Step 2: Generate ground-truth annotation JSON file 
Run `main.py` with `steps=gt_anno` to generate ground truth annotation file. Default is to create ground truth annotation JSON file for manually annotated data in all splits (`train/val/test`).

```
python3 main.py \
    --steps gt_anno
    --ego4d_data_dir <ego4d-out-dir> \
    --gt_output_dir <gt-output-dir>
```

Four annotation JSON files will be generated:
- **ego_pose_gt_anno_train_public.json**: includes 3D hand joints coordinates, hand bbox and valid hand joints flag in all annotation available `train` takes. Available to public. 
- **ego_pose_gt_anno_val_public.json**: includes 3D hand joints coordinates, hand bbox and valid hand joints flag in all annotation available `val` takes. Available to public. 
- **ego_pose_gt_anno_test_public.json**: includes hand bbox in all annotation available `test` takes. Available to public. 
- **ego_pose_gt_anno_test_private.json**: includes 3D hand joints coordinates, hand bbox and valid hand joints flag in all annotation available `test` takes. Not available to public, only for server evaluation.

In this step, we filtered valid 3D annotations by biomechanical constraints, and save it in the generated files. We encourage but do not request the users to use it as a data cleanup process.  

The structure of `ego_pose_gt_anno_train_public.json`, `ego_pose_gt_anno_val_public.json` and `ego_pose_gt_anno_test_private.json` is:  
```
{
    <"take_uid"> :{
        <"frame_number">: {
            "right_hand_2d": a (21,2) array for 2D annotations for each joint, joint ordering in note. 
                            Empty if the hand is invisible, and NaN if the joint is invisible. 
            "right_hand_3d": a (21,3) array for 3D annotations for each joint, joint ordering in note.  
                            Empty if the hand is invisible, and NaN if the joint is invisible. 
            "right_hand_bbox": a (4,) array for hand bounding box with format xyxy. 
                            Empty if the hand is invisible. 
            "right_hand_valid": a (21,) array to show if each joint is valid.
                            Empty if the hand is invisible.
            the same entries for left hand ...
            "metadata": some metadata for the take and the frame.
        }
    }
}
```
The structure of `ego_pose_gt_anno_test_public.json` is:  
```
{
    <"take_uid">: {
        <"frame_number">: {
            "right_hand_bbox": a (4,) array for right hand bounding box with format xyxy. 
                            Empty if the hand is invisible and not evaluated.  
            "left_hand_bbox": a (4,) array for left hand bounding box with format xyxy. 
                            Empty if the hand is invisible and not evaluated.  
            "metadata": some metadata for the take and the frame.
        }
    }
}
```

A sample of four ground truth annotation files can be found from [here](https://drive.google.com/drive/folders/17TYpJl523r8nzjRB3cBzxbhr2BhM7R8U?usp=sharing).

### Step 3: Extract & undistort Aria images
Run `main.py` with `steps=raw_image undistorted_image` to first extract Aria raw images to `<gt-output-dir>`, then perform undistortion to get undistorted Aria images. Default is to extract and undistort all manually annotated frames used in all splits (`train/val/test`).
```
python3 main.py \
    --steps raw_image undistorted_image \
    --ego4d_data_dir <ego4d-out-dir> \
    --gt_output_dir <gt-output-dir>
```
comment: some take videos are not present duw to previous inconsistency  
comment: need to double-check with the undistortion orientation  
comment: need to explain the orientation somewhere  
comment: need a visualization code here  
comment: need a script for to rotate the image and the annotations (2D+3D) to the regular orientation

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
