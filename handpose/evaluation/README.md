# Ego-Exo4D Hand Ego Pose Benchmark Evaluation

Here is an example for ego hand pose model evaluation.  
For challenge participants, please submit model outputs to [EvalAI challenge](https://eval.ai/web/challenges/challenge-page/2249/overview)

## File requirements
### Ground truth annotation JSON file 
Ground truth annotation for test split is not released to public. The file structure the same as files generated in 
[data preparation](https://github.com/EGO4D/ego-exo4d-egopose/tree/main/handpose/data_preparation) step 2. 
### Model inference results 
The inference output needs to be saved as a single JSON file with the following format:
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

## Evaluation

Evaluate the model performance based on prediction inference output JSON file (which is at `<pred_path>`) and ground truth JSON file (which is at `<gt_path>`). Remember to set `offset` if the user inference output is offset by hand wrist. 
```
python3 evaluate.py \
    --pred_path <pred_path> \
    --gt_path <gt_path> 
```