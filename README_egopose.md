# Setup the environment

### System (Zhifan)
- Ubuntu 20.04
- CUDA 10.1
- Python 3.7
- Pytorch 1.4
- torchvision

### Setup METRO with Conda (Zhifan modified)

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

### Install 100DOH detector

```bash
git clone git@github.com:ddshan/hand_object_detector.git ./thirdparty/hand_object_detector
cd thirdparty/hand_object_detector/lib
pip install -r thirdparty/hand_object_detector/requirements.txt
cd ../../..  # Go back to MeshTransformer
cp run_ho_detector.py thirdparty/hand_object_detector/
```


### Downloading the weights (SMPL, MANO, METRO, 100DOH)

- Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [SMPLify](http://smplify.is.tue.mpg.de/), and place it at `${REPO_DIR}/metro/modeling/data`.
- Download `MANO_RIGHT.pkl` from [MANO](https://mano.is.tue.mpg.de/), and place it at `${REPO_DIR}/metro/modeling/data`.
- Download METRO hand network weights into `$REPO_DIR/models/metro_release/metro_hand_state_dict.bin`
- Model config into `$REPO_DIR/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml`
- 100DOH Faster RCNN weights into `./thirdparty/hand_object_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth`


# Run the method

Assuming videos uid is `98f58f0f-53d6-4e41-bf41-d8d74ccbc37c`, 
frames have been extracted to `./egopose_storage/frames/98f58f0f-53d6-4e41-bf41-d8d74ccbc37c/`

## Run & Save 100DOH hand detector bounding boxes
```bash
# In Bash
export METRODIR=$PWD
export FRAMESDIR=$(realpath egopose_storage/frames/98f58f0f-53d6-4e41-bf41-d8d74ccbc37c)
export DET_RESULT=$(realpath egopose_storage)/dets/98f58f0f-53d6-4e41-bf41-d8d74ccbc37c.csv
cd thirdparty/hand_object_detector

export PYTHONPATH=$PYTHONPATH:$PWD
python run_ho_detector.py \
  --cuda --checkepoch=8 \
  --checkpoint=132028 \
  --image_dir=$FRAMESDIR \
  --result_path=$DET_RESULT
cd $METRODIR  # GO back
```
## Crop and save output images

METRO takes as input images crops centered at the hand, hence we cache the images.
```bash
python hand_crop.py  # Assuming we run 98f58f0f-53d6-4e41-bf41-d8d74ccbc37c
```

## Run MeshTransformer

Finally
```
python ./metro/tools/end2end_inference_handmesh_egopose.py --resume_checkpoint ./models/metro_release/metro_hand_state_dict.bin 
```

This saves the result into `./egopose_storage/results/{rends, pred_3ds, pred_2ds}`