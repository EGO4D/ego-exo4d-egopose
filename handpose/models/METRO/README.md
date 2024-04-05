Inference by 

# Setup the environment

### System
- Ubuntu 20.04
- CUDA 10.1
- Python 3.7
- Pytorch 1.4
- torchvision

### Setup METRO with Conda

```bash
git clone --recursive https://github.com/microsoft/MeshTransformer.git
cd MeshTransformer

conda create --name metro-hand python=3.7
conda activate metro-hand

# Install Pytorch
# conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
pip install torch==1.4.0 torchvision==0.5.0

# Install METRO
python setup.py build develop

# Install requirements
pip install -r requirements.txt

# Install manopth
pip install ./manopth/.

# Install pytorch3d-0.6.2 and libzhifan (for visualisation purpose)
# Note make sure gcc and g++ version are v9, do export CC=gcc-9 and export CXX=g++-9 if necessary
git clone --depth 1 --branch v0.6.2 git@github.com:facebookresearch/pytorch3d.git thirdparty/pytorch3d
pip install ./thirdparty/pytorch3d
pip install ./thirdparty/libzhifan
```


### Downloading the weights (SMPL, MANO, METRO)

- Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [SMPLify](http://smplify.is.tue.mpg.de/), and place it at `${REPO_DIR}/metro/modeling/data`.
- Download `MANO_RIGHT.pkl` from [MANO](https://mano.is.tue.mpg.de/), and place it at `${REPO_DIR}/metro/modeling/data`.
- Download METRO hand network weights into `$REPO_DIR/models/metro_release/metro_hand_state_dict.bin`
- Model config into `$REPO_DIR/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml`

# Run the method
Here we provide the script to run METRO on EgoExo4D benchmark. 
Note that the model is not trained on the dataset. The scripts provided are for inference only. 
## Data preparation
Follow instructions [here](https://github.com/EGO4D/ego-exo4d-egopose/tree/main/handpose/data_preparation) to get:
- ground truth annotation files in `$gt_output_dir/annotation/manual` or `$gt_output_dir/annotation/auto` if using automatic annotations,
referred as `gt_anno_dir` below
- corresponding undistorted Aria images in `$gt_output_dir/image/undistorted`, 
referred as `aria_img_dir` below

## Crop and save output images

METRO takes as input images crops centered at the hand, hence we cache the images.
```bash
python hand_crop.py --gt_output_dir $gt_output_dir --storage_dir $output_folder
```

## Run MeshTransformer

Finally
```
python ./metro/tools/end2end_inference_handmesh_egopose.py --resume_checkpoint ./models/metro_release/metro_hand_state_dict.bin --gt_output_dir $gt_output_dir --storage_dir $output_folder
```

This saves the result into `$output_folder/results_gt_bbox/{rends, pred_3ds, pred_2ds}`, 
and corresponding result json file in `$output_folder/results_gt_bbox/metro_inference.json`