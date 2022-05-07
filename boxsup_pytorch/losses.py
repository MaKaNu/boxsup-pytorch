"""Module for loss functions."""

from typing import Union

import numpy as np
import numpy.typing as npt


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
