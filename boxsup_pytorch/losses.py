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
