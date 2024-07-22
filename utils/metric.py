"""IoU"""

import numpy as np
from dataset.scannet.scannet_constants import SCANNET20_CLASS_LABELS, COCOMAP_CLASS_LABELS

def confusion_matrix(pred_ids, gt_ids, num_classes):
    """calculate the confusion matrix."""

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)

    # the sum of each row (axis=1) is predicted truth, the sum of each column (axis=0) is ground truth
    confusion = (
        np.bincount(pred_ids * (num_classes + 1) + gt_ids, minlength=(num_classes + 1) ** 2)
        .reshape((num_classes + 1, num_classes + 1))
        .astype(np.ulonglong)
    )
    return confusion[:, 1:] # do not calculate unlabeled points (the first column)

def get_iou(label_id, confusion):
    """calculate IoU."""

    # true positives
    tp = np.longlong(confusion[label_id + 1, label_id])
    # false positives
    fp = np.longlong(confusion[label_id + 1, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = tp + fp + fn
    if denom == 0:
        return float("nan")
    return float(tp) / denom, tp, denom


def evaluate_confusion(confusion, stdout=False, dataset="scannet20"):
    if stdout:
        print("evaluating", confusion.sum(), "points...")

    if "scannet20" in dataset:
        CLASS_LABELS = SCANNET20_CLASS_LABELS
    elif "cocomap" in dataset:
        CLASS_LABELS = COCOMAP_CLASS_LABELS
    else:
        raise NotImplementedError
    N_CLASSES = len(CLASS_LABELS)
    print("num_classes:", N_CLASSES)

    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if confusion.sum(axis=0)[i] == 0:  # at least 1 point needs to be in the evaluation for this class
            continue

        class_ious[label_name] = get_iou(i, confusion)
        class_accs[label_name] = class_ious[label_name][1] / confusion.sum(axis=0)[i]
        count += 1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]

    mean_iou /= count
    mean_acc /= count
    if stdout:
        print("classes          IoU")
        print("----------------------------")
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                print(
                    "{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})".format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2],
                    )
                )
            except:
                print(label_name + " error!")
                continue
        print("Mean IoU", mean_iou)
        print("Mean Acc", mean_acc)

    with open("eval_result.log", "a") as fp:
        fp.write("classes,IoU\n")
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                fp.write(
                    "{0:<14s}: {1:>5.3f}  ({2:>6d}/{3:<6d})\n".format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2],
                    )
                )
            except:
                fp.write(label_name + ",error\n")
        fp.write("mean IoU,{}\n".format(mean_iou))
        fp.write("mean Acc,{}\n\n".format(mean_acc))
    return mean_iou
