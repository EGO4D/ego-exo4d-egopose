# Ego-Exo4D Hand Ego Pose Benchmark Evaluation

Follow instruction below to perform hand ego pose model performance evaluation.

## Requirement

To perform model evaluation, you need to have 1. ground truth annotation JSON file 2. user inference output. The user inference output need to be saved as a single JSON file with specific format:
```
{
    "<take_uid>": {
        "<frame_number>": {
                "left_hand_3d": [],
                "right_hand_3d": []
        }
    }
}
```
You can find one sample inference output JSON file from [here](https://drive.google.com/drive/folders/1jmN427e2f1vsOLcTUkiAKVlqJUx5acfS?usp=sharing).


## Evaluation

Evaluate the model performance based on prediction inference output JSON file (which is at `<pred_path>`) and ground truth JSON file (which is at `<gt_path>`). Remember to set `offset` if the user inference output is offset by hand wrist. 
```
python3 evaluate.py \
    --pred_path <pred_path> \
    --gt_path <gt_path> \
    --offset 
```