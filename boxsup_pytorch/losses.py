"""Module for loss functions."""

from typing import List, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from boxsup_pytorch.pipeline.core import create_mask_from_index


def overlapping_loss(
    bbox: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """Calculate how well candidates matches the bounding box.

    Args:
        box (torch.Tensor): boundingbox
        candidates (torch.Tensor): candidate or array of candidates

    Returns:
        torch.Tensor: the calculated loss
    """
    N = len(candidates)
    loss = 0
    bbox_mask_class = torch.Tensor([bbox[0]])
    bbox_mask_indices = bbox[1]
    for candidate in candidates:
        candidate_mask_class = torch.Tensor([candidate[0]])
        candidate_mask_indices = candidate[1]
        loss += (1 - inter_o_union(bbox_mask_indices, candidate_mask_indices)) * \
            compare_labels(bbox_mask_class, candidate_mask_class)
    return 1/N * loss


def regression_loss(
    prediction: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """Calculate logistic regression.

    Args:
        est_mask (torch.Tensor): Estimated mask
        lab_mask (torch.Tensor): Label mask

    Returns:
        List: logistic regression loss
    """
    resolution = prediction.shape[2:]
    loss = nn.CrossEntropyLoss(ignore_index=0)
    result = []
    for candidate in candidates:
        created_mask = create_mask_from_index(
            mask_indices=candidate[1],
            mask_class=candidate[0],
            resolution=resolution
        )
        result.append(loss(prediction, created_mask.reshape(1, *created_mask.shape)))
    return result


def weighted_loss(
    o_loss: torch.Tensor,
    r_loss: List[torch.Tensor],
    weight: torch.Tensor = torch.Tensor([3.0]),
) -> Union[np.float64, npt.NDArray[np.float64]]:
    """Calculate the weighted loss.

    Args:
        o_loss (Union): single overlapping or array loss
        r_loss (Union): single regression or array loss
        weight (np.float64): weighting factor

    Returns:
        np.float64: weighted loss
    """

    return torch.Tensor([reg * weight + o_loss for reg in r_loss])


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
    return box == candidates


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
    if len(pred) > len(target):
        inter = sum(torch.isin(pred, target, assume_unique=True))
    else:
        inter = sum(torch.isin(target, pred, assume_unique=True))

    union = len(pred) + len(target) - inter
    return inter/union if union > 0 else 0
