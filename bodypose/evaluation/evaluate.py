import argparse
import json

import numpy as np


def add_arguments(parser):
    parser.add_argument("--gt_path", help="Path of ground truth annotation JSON file")
    parser.add_argument(
        "--pred_path", help="Path of user inference prediction JSON file"
    )
    parser.add_argument(
        "--min_clip_len",
        type=int,
        default=120,
        help="Minimum clip length to be evaluated in GT. The default 120 means 4 seconds clip with 30 FPS",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        "Performs ego-body-pose evaluation and report MPJPE & MPJVE as metric."
    )
    add_arguments(parser)
    args = parser.parse_args()
    print(args)

    return args


def compute_mpjpe(gt, pred):
    assert gt.shape == pred.shape, f"gt.shape = {gt.shape} != {pred.shape} = pred.shape"
    total_error = 0
    total_count = 0
    for j in range(gt.shape[0]):
        if not np.isnan(gt[j][0]):
            total_error += np.linalg.norm(gt[j] - pred[j])
            total_count += 1
    return total_error, total_count


def compute_mpjve(gt, pred, t, prev_gt, prev_pred, prev_t):
    total_error = 0
    total_count = 0

    if prev_gt is not None and prev_pred is not None and prev_t is not None:
        assert t > prev_t
        if t - prev_t < 10:
            # only calculate the velocity when the neighboring frames are <10 frames away
            fps_multiplier = 30.0 / (t - prev_t)
            for j in range(gt.shape[0]):
                if not np.isnan(gt[j][0]) and not np.isnan(prev_gt[j][0]):
                    total_error += (
                        np.linalg.norm((gt[j] - prev_gt[j]) - (pred[j] - prev_pred[j]))
                        * fps_multiplier
                    )
                    total_count += 1
    return total_error, total_count


def infer_category_from_take_name(take_name, category_map):
    category_for_take = None
    for category in category_map:
        for sub_category in category_map[category]:
            if sub_category in take_name.lower():
                category_for_take = category
                break
        if category_for_take is not None:
            break
    assert category_for_take is not None, take_name
    return category_for_take


def filter_short_clips(frame_list, min_clip_len=120):
    prev_t = -1
    final_list = []

    for t in frame_list:
        if prev_t == -1:
            prev_t = t
            start_t = t
            buffer = []
        elif t - prev_t > 3:
            if prev_t - start_t + 3 >= min_clip_len:
                final_list += buffer

            start_t = t
            prev_t = t
            buffer = []

        buffer.append(t)
        prev_t = t

    # Remaining buffer
    if prev_t != -1 and prev_t - start_t + 3 >= min_clip_len:
        final_list += buffer

    return final_list


def main(args):
    pred_all = json.load(open(args.pred_path))
    gt_all = json.load(open(args.gt_path))

    category_map = {
        "basketball": ["basketball"],
        "bike": ["bike"],
        "cooking": ["cooking"],
        "dance": ["dance"],
        "soccer": ["soccer"],
        "health": ["cpr", "covid"],
        "bouldering": ["bouldering"],
        "music": ["piano", "violin", "guitar", "music"],
    }

    mpjpe = {}
    mpjpe_count = {}
    mpjve = {}
    mpjve_count = {}
    for category in category_map:
        mpjpe[category] = 0
        mpjpe_count[category] = 0
        mpjve[category] = 0
        mpjve_count[category] = 0

    for take_uid in gt_all:
        take_name = gt_all[take_uid]["take_name"]
        gt_take = gt_all[take_uid]["body"]
        pred_take = pred_all[take_uid]["body"]

        category_for_take = infer_category_from_take_name(take_name, category_map)

        frame_list = sorted([int(frame_str) for frame_str in gt_take.keys()])
        frame_list = filter_short_clips(frame_list, min_clip_len=args.min_clip_len)

        previous_pred = None
        previous_gt = None
        previous_t = None

        for t in frame_list:
            idx = str(t)

            gt = np.array(gt_take[idx])

            try:
                pred = np.array(pred_take[idx])
            except KeyError:
                raise KeyError(
                    f"Cannot find result for frame {idx} for {take_uid} / {take_name} in {args.pred_path}!"
                )

            pjpe_error, pjpe_count = compute_mpjpe(gt, pred)

            mpjpe[category_for_take] += pjpe_error
            mpjpe_count[category_for_take] += pjpe_count

            pjve_error, pjve_count = compute_mpjve(
                gt, pred, t, previous_gt, previous_pred, previous_t
            )
            mpjve[category_for_take] += pjve_error
            mpjve_count[category_for_take] += pjve_count

            previous_gt = gt.copy()
            previous_pred = pred.copy()
            previous_t = t

    for category in category_map:
        if mpjpe_count[category] == 0:
            print(f"{category} doesn't have any samples yet!")
            continue
        print(
            " ".join(
                [
                    f"[{category}] mpjpe: {mpjpe[category]/mpjpe_count[category]*100:.2f},",
                    f"mpjve: {mpjve[category]/mpjve_count[category]:.2f}",
                ]
            )
        )

    total_mpjpe = 0
    total_mpjpe_count = 0
    total_mpjve = 0
    total_mpjve_count = 0

    for category in category_map:
        total_mpjpe += mpjpe[category]
        total_mpjpe_count += mpjpe_count[category]
        total_mpjve += mpjve[category]
        total_mpjve_count += mpjve_count[category]

    print(
        " ".join(
            [
                f"[overall] mpjpe: {total_mpjpe/total_mpjpe_count*100:.2f},",
                f" mpjve: {total_mpjve/total_mpjve_count:.2f}",
            ]
        )
    )

    print(
        f"total_mpjpe_count: {total_mpjpe_count}\ntotal_mpjve_count: {total_mpjve_count}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
