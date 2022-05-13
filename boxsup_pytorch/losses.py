"""Module for loss functions."""

from typing import Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


def overlapping_loss(
    box: npt.NDArray[np.float64], candidates: npt.NDArray[np.float64]
) -> Union[npt.NDArray[np.float64], np.float64]:
    """Calculate how well candidates matches the bounding box.

    Args:
        box (np.array): boundingbox
        candidates (np.array): candidate or array of candidates

    Returns:
        np.float64: the calculated loss
    """
    print(1 - inter_o_union(box, candidates))
    if len(candidates.shape) == 3:
        N = candidates.shape[0]
        return (
            1 / N * np.sum((1 - inter_o_union(box, candidates)) * compare_labels(box, candidates))
        )
    else:
        return (1 - inter_o_union(box, candidates)) * compare_labels(box, candidates)


def regression_loss(
    est_mask: npt.NDArray[np.float64], lab_mask: npt.NDArray[np.float64]
) -> Union[npt.NDArray[np.float64], np.float64]:
    """Calculate logistic regression.

    Args:
        est_mask (npt.NDArray[np.float64]): Estimated mask
        lab_mask (npt.NDArray[np.float64]): Label mask

    Returns:
        Union[npt.NDArray[np.float64], np.float64]: logistic regression loss
    """
    if len(lab_mask.shape) == 3:  # More than 1 candidate
        num_labels = lab_mask.shape[0]
        est_mask = np.concatenate([est_mask] * num_labels)
    else:
        lab_mask = lab_mask.reshape(1, *lab_mask.shape)
    loss = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
    input = torch.from_numpy(est_mask.astype(np.float64))
    target = torch.from_numpy(lab_mask.astype(np.int64))
    return torch.mean(loss(input, target), dim=(1, 2)).numpy()


def weighted_loss(
    o_loss: Union[np.float64, npt.NDArray[np.float64]],
    r_loss: Union[np.float64, npt.NDArray[np.float64]],
    weight: np.float64 = np.float64(3.0),
) -> Union[np.float64, npt.NDArray[np.float64]]:
    """Calculate the weighted loss.

    Args:
        o_loss (Union): single overlapping or array loss
        r_loss (Union): single regression or array loss
        weight (np.float64): weighting factor

    Returns:
        np.float64: weighted loss
    """
    return o_loss + weight * r_loss


def compare_labels(
    box: npt.NDArray[np.float64], candidates: npt.NDArray[np.float64]
) -> Union[bool, npt.NDArray[np.bool_]]:
    """Check if label of box is equal to labels of candidates.

    Args:
        box (np.array): BoundingBox
        candidates (np.array): Candidate or array of candidates

    Returns:
        bool: True if labels are equal
    """
    axis = None
    if len(candidates.shape) == 3:  # More than 1 candidate
        axis = (1, 2)
    return np.max(box) == np.max(candidates, axis=axis)


def inter_o_union(
    pred: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Calculate Intersection over Union.

    Args:
        pred (np.array): The prediction(s) px array
        target (np.array): The target(s) px array

    Returns:
        float: the divison of sum of intersection px by union px
    """
    axis = None
    if len(pred.shape) == 3 or len(target.shape) == 3:  # More than 1 pred
        axis = (1, 2)
    # Get highest Class idx
    high_class = np.max((np.max(pred), np.max(target)))

    inter = 0
    union = 0
    for class_value in range(1, high_class + 1):
        mask_pred = np.ma.masked_where(pred == class_value, pred).mask
        mask_target = np.ma.masked_where(target == class_value, target).mask
        if not mask_pred.shape == pred.shape:
            mask_pred = np.zeros_like(pred, dtype=bool)
        if not mask_target.shape == target.shape:
            mask_target = np.zeros_like(target, dtype=bool)

        inter += np.sum(np.logical_and(mask_pred, mask_target), axis=axis, dtype=np.float64)
        union += np.sum(np.logical_or(mask_pred, mask_target), axis=axis, dtype=np.float64)

    return np.divide(inter, union, out=np.zeros_like(union), where=union != 0)
