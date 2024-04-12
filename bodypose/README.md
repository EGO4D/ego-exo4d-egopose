**[NEW!] 2024 Ego-Exo4D Challenges now open**
- [Body Pose](https://eval.ai/web/challenges/challenge-page/2245/overview) (deadline May 30 2024)


# Ego-Exo4D Body Pose Baseline Model

Implementation of a 3D body pose estimation baseline model in [Ego-Exo4D](https://ego-exo4d-data.org).

## Dependencies and Installation
Follow the instructions below to set up the environment for model training, testing, and inference.

    
    conda env create -f environment.yaml
    

## Data Preparation 

Follow the instructions to install the [Ego-Exo4D CLI Downloader](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/egoexo/download/README.md)

Download ego pose related `annotations` and `metadata` for both manual and automatic annotations:
```
egoexo -o $egoexo_output_dir --parts annotations metadata ego_pose_pseudo_gt --benchmarks body_pose
```
## Train 
- **Training command**:

    ```bash
    python train.py
    ```
Modify the corresponding training option file at `options/train_egoexo.json`.

## Test 
- **Testing command**:

    ```bash
    python test.py
    ```
Modify the corresponding training option file at `options/test_egoexo.json`. This code is designed for testing on the publicly available annotations from the train or val set. For the test set follow the inference instructions. 

## Inference 
- **Inference command**:

    ```bash
    cd data
    unzip dummy_test.zip
    cd ..
    python inference.py 
    ```
Modify the corresponding training option file at `options/inference_egoexo.json`. The inference will copy the format of the `dummy_test.json` file and will be stored in `results/EgoExo4D/inference/test_pred.json` by default. 
Download pretrained [EvalAI baseline](https://eval.ai/web/challenges/challenge-page/2245/overview) model weights of this baseline from [here](https://drive.google.com/file/d/1XpY7aa7I7XFNDM6tJPcyS17xPsDlW0g7/view?usp=sharing).

## Frame of reference for 3D predictions

There are three options for the frame of reference used in this baseline approach, you can select which frame of reference to use in the options files for training and testing. The default one is using the positions in the reference frame of the camera at each timeframe `coord: null`. The second option is using the positions in the reference frame of the first frame of the camera (this option will show relative movement) `coord: aria`. The third option is using the positions directly in the global reference frame  (this option will also show relative movement)`coord: global`. 

Note that the EvalAI evaluation submission requires the predicted 3D positions for each joint to be in the global reference framework. This baseline outputs by default the positions in the reference frame of the camera at each timeframe, for converting the predicted position into the global reference frame run 
```
global_coordinates.py --root $path_to_egoexo_annotations --pred $path_to_json_file_to_convert
```
 and you will obtain a `test_pred_global.json` file that can be submitted to the EvalAI server. 

## Acknowledgments
Cite [Ego-Exo4D](https://arxiv.org/abs/2311.18259) if you are using this code
```
@article{grauman2023ego,
  title={Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives},
  author={Grauman, Kristen and Westbury, Andrew and Torresani, Lorenzo and Kitani, Kris and Malik, Jitendra and Afouras, Triantafyllos and Ashutosh, Kumar and Baiyya, Vijay and Bansal, Siddhant and Boote, Bikram and others},
  journal={arXiv preprint arXiv:2311.18259},
  year={2023}
}
```
This code base is built on [BoDiffusion](https://bcv-uniandes.github.io/bodiffusion-wp/) and [AvatarPoser](https://github.com/eth-siplab/AvatarPoser). Please also consider citing these works if you use this codebase. 

```
@article{castillo2023bodiffusion,
  author    = {Castillo, Angela and Escobar, Maria and Jeanneret, Guillaume and Pumarola, Albert and Arbeláez, Pablo and Thabet, Ali and Sanakoyeu, Artsiom},
  title     = {BoDiffusion: Diffusing Sparse Observations for Full-Body Human Motion Synthesis},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2023},
}
```
```
@inproceedings{jiang2022avatarposer,
  title={AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing},
  author={Jiang, Jiaxi and Streli, Paul and Qiu, Huajian and Fender, Andreas and Laich, Larissa and Snape, Patrick and Holz, Christian},
  booktitle={Proceedings of European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
```
