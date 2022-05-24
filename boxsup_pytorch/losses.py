"""Module for loss functions."""

from typing import Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


def overlapping_loss(
    box: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """Calculate how well candidates matches the bounding box.

    Args:
        box (torch.Tensor): boundingbox
        candidates (torch.Tensor): candidate or array of candidates

    Returns:
        torch.Tensor: the calculated loss
    """
    if len(candidates.shape) == 3:
        N = candidates.shape[0]
        return (
            1 / N * torch.sum(
                (1 - inter_o_union(box, candidates)) * compare_labels(box, candidates)
            )
        )
    else:
        return (1 - inter_o_union(box, candidates)) * compare_labels(box, candidates)


def regression_loss(
    est_mask: torch.Tensor, lab_mask: torch.Tensor
) -> torch.Tensor:
    """Calculate logistic regression.

    Args:
        est_mask (torch.Tensor): Estimated mask
        lab_mask (torch.Tensor): Label mask

    Returns:
        torch.Tensor: logistic regression loss
    """
    if len(lab_mask.shape) == 3:  # More than 1 candidate
        num_labels = lab_mask.shape[0]
        est_mask = est_mask.repeat(num_labels, 1, 1, 1)
    else:
        lab_mask = lab_mask.reshape(1, *lab_mask.shape)
    loss = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
    return torch.mean(loss(est_mask, lab_mask), dim=(1, 2))


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
    box: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """Check if label of box is equal to labels of candidates.

    Args:
        box (torch.Tensor): BoundingBox
        candidates (torch.Tensor): Candidate or array of candidates

    Returns:
        bool: True if labels are equal
    """
    dim = (0, 1)
    if len(candidates.shape) == 3:  # More than 1 candidate
        dim = (1, 2)
    return torch.max(box) == torch.amax(candidates, dim=dim)


def inter_o_union(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Calculate Intersection over Union.

    Args:
        pred (torch.Tensor): The prediction(s) px array
        target (torch.Tensor): The target(s) px array

    Returns:
        torch.Tensor: the divison of sum of intersection px by union px
    """
    dim = (0, 1)
    if len(pred.shape) == 3 or len(target.shape) == 3:  # More than 1 pred
        dim = (1, 2)
    # Get highest Class idx
    high_class = max(pred.max(), target.max())

    inter = 0
    union = 0
    for class_value in range(1, high_class + 1):
        mask_pred = pred == class_value
        mask_target = target == class_value
        if not mask_pred.shape == pred.shape:
            mask_pred = torch.zeros(pred.shape, dtype=bool)
        if not mask_target.shape == target.shape:
            mask_target = torch.zeros(target.shape, dtype=bool)

        inter += torch.sum(mask_pred.logical_and(mask_target), dim=dim, dtype=torch.float64)
        union += torch.sum(mask_pred.logical_or(mask_target), dim=dim, dtype=torch.float64)
    zero_mask = union != 0
    i_o_u = torch.zeros(inter.shape, dtype=torch.float64)
    i_o_u[zero_mask] = inter[zero_mask].div(union[zero_mask])
    return i_o_u
