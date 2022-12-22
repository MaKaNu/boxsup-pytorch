"""Module for loss functions."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
import torch.nn as nn

from boxsup_pytorch.utils.check import check_shape_len_msg


__all__ = ['MixSoftmaxCrossEntropyLoss', 'Losses', 'get_segmentation_loss']


class Losses():
    """Loss class which provides all loss methods of this project."""

    config: Optional[Dict[str, Any]]

    def __init__(self, config=None) -> None:
        """Construct of Losses class.

        Args:
            config (Dict[str, Any]): config dictionary provided via toml
        """
        self.classes = 6 if config is None else config["num_classes"]

    def overlapping_loss(
        self, box: Tensor, candidates: Tensor
    ) -> Tensor:
        """Calculate how well candidates matches the bounding box.

        Args:
            box (torch.Tensor): boundingbox [1 x w x h]
            candidates (torch.Tensor): candidate or array of candidates [N x w x h]

        Returns:
            torch.Tensor: the calculated loss
        """
        assert len(box.shape) == 2, check_shape_len_msg(2)
        assert len(candidates.shape) in [2, 3], check_shape_len_msg((2, 3))

        if len(candidates.shape) == 3:
            N = candidates.shape[0]
            classes = candidates.max(1).values.max(1).values
            binary_candidates = candidates.div(classes[..., None, None])
        else:
            N = 1
            classes = candidates.max()
            binary_candidates = candidates.div(classes)
        return (
            1 / N * torch.sum(
                (1 - self._inter_over_union(binary_candidates, box)) *
                self._compare_labels(candidates, box)
            )
        )

    def regression_loss(
        self, est_mask: Tensor, lab_mask: Tensor
    ) -> Tensor:
        """Calculate logistic regression.

        Args:
            est_mask (torch.Tensor): Estimated mask
            lab_mask (torch.Tensor): Label mask

        Returns:
            torch.Tensor: logistic regression loss
        """
        if len(lab_mask.shape) == 3:  # More than 1 candidate
            num_labels = lab_mask.shape[0]
            est_mask = est_mask.expand(num_labels, *list(est_mask.shape[1:]))
        else:
            lab_mask = lab_mask.reshape(1, *lab_mask.shape)
        loss = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
        return torch.mean(loss(est_mask, lab_mask), dim=(1, 2))

    def weighted_loss(
        self, o_loss: Tensor, r_loss: Tensor, weight: float = 3.0
    ) -> Tensor:
        """Calculate the weighted loss.

        Args:
            o_loss (Union): single overlapping or array loss
            r_loss (Union): single regression or array loss
            weight (np.float64): weighting factor

        Returns:
            np.float64: weighted loss
        """
        return o_loss + weight * r_loss

    def _compare_labels(
        self, candidates: Tensor, box: Tensor
    ) -> Tensor:
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

    def _inter_over_union(
        self, pred: Tensor, target: Tensor
    ) -> Tensor:
        """Calculate Intersection over Union.

        Args:
            pred (torch.Tensor): The prediction(s) px array Shape: [N ...]
            target (torch.Tensor): The target px array Shape: [1 ...]

        Returns:
            torch.Tensor: the divison of sum of intersection px by union px
        """
        dim = (1, 2) if len(pred.shape) == 3 else (0, 1)

        expanded_target = target.expand_as(pred)
        class_idx = target.max()
        inter_mask = expanded_target.mul(pred)
        inter = inter_mask.sum(dim=dim) / class_idx
        traget_area = expanded_target.sum(dim) / class_idx
        pred_area = pred.sum(dim)
        union = traget_area + pred_area - inter
        return inter / union
