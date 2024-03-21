# Ego-Exo4D Ego Body Pose Benchmark Evaluation

Here is an example for ego hand pose model evaluation.  
For challenge participates, please submit model outputs to [EvalAI challenge](https://eval.ai/web/challenges/challenge-page/2249/overview)

## File requirement
### Ground truth annotation JSON file 
Ground truth annotation for test split is not released to public. 
### Model inference results 
The inference output needs to be saved as a single JSON file with the following format:
```
{
    "<take_uid>": {
        "take_name": <take_name>,
        "body": {
            "<frame_number>": [],
            ...
        }
    }
}
```

## Evaluation

Evaluate the model performance based on prediction inference output JSON file (which is at `<pred_path>`) and ground truth JSON file (which is at `<gt_path>`).
```
python3 evaluate.py \
    --gt_path <gt_path> \
    --pred_path <pred_path>
```

[//]: # (For the example inputs above, you will see something like:)

[//]: # ()
[//]: # (```)

[//]: # ([basketball] mpjpe: 36.52, mpjve: 0.83)

[//]: # ([bike] mpjpe: 44.22, mpjve: 0.42)

[//]: # ([cooking] mpjpe: 27.61, mpjve: 0.42)

[//]: # ([dance] mpjpe: 36.31, mpjve: 0.86)

[//]: # ([soccer] mpjpe: 34.61, mpjve: 0.73)

[//]: # ([health] mpjpe: 32.67, mpjve: 0.23)

[//]: # (bouldering doesn't have any samples yet!)

[//]: # ([music] mpjpe: 35.47, mpjve: 0.37)

[//]: # ([overall] mpjpe: 35.02,  mpjve: 0.57)

[//]: # (total_mpjpe_count: 226938)

[//]: # (total_mpjve_count: 215020)

[//]: # (```)