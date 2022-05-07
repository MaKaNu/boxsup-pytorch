r"""Test Module for losses.

test scenario for overlapping_loss:
    The loss formula:
    $\Epsilon_o = {1\over N} \sum (1 - IoU(B,S)\delta (l_B, l_S)$

    we test with following inputs:

    Grid size is always 2x2

    test 1:
    - bounding_box: count 1, label 1
    - candidates: count 3, label 1, no overlapping

    expected result = 1

    test 2:
    - bounding_box: count 1, label 1
    - candidates: count 1, label 1, no overlapping
    - candidates: count 1, label 1, overlapping
    - candidates: count 1, label 1, full overlapping

    expected result = 1/2

    test 3:
    - bounding_box: count 1, label 1
    - candidates: count 3, label 1, full overlapping

    expected result = 0

    test 4:
    - bounding_box: count 1, label 1
    - candidates: count 3, label 2, full overlapping

    expected result = 0

    test 5:
    - bounding_box: count 1, label 1
    - candidates: count 1, label 1, no overlapping

    expected result = 1
"""

import numpy as np
import pytest

# Fixture Setup


@pytest.fixture()
def bounding_box() -> np.array:
    """Pytest fixture of 2x2 BoundingBox."""
    return np.array([[0, 1], [0, 0]])


@pytest.fixture()
def bounding_box3() -> np.array:
    """Pytest fixture of 2x2 BoundingBox."""
    return np.array([[1, 1, 1], [1, 0, 2], [2, 2, 2]])


@pytest.fixture()
def multi_cand() -> np.array:
    """Pytest fixture of 2x2 BoundingBox."""
    return np.array([[1, 1, 1], [2, 0, 1], [2, 2, 2]])


@pytest.fixture()
def overlap_cand() -> np.array:
    """Pytest fixture of 2x2 overlapping candidate."""
    return np.array([[0, 1], [0, 1]])


@pytest.fixture()
def not_overlap_cand() -> np.array:
    """Pytest fixture of 2x2 not overlapping candidate."""
    return np.array([[0, 0], [1, 0]])


@pytest.fixture()
def full_overlap_cand() -> np.array:
    """Pytest fixture of 2x2 full overlapping candidate."""
    return np.array([[0, 1], [0, 0]])


@pytest.fixture()
def not_overlap_cands(not_overlap_cand: np.array) -> np.array:
    """Pytest fixture of 3 2x2 not overlapping candidates."""
    return np.array((not_overlap_cand, not_overlap_cand, not_overlap_cand))


@pytest.fixture()
def mixed_overlap_cands(
    not_overlap_cand: np.array, overlap_cand: np.array, full_overlap_cand: np.array
) -> np.array:
    """Pytest fixture of 3 2x2 mixed overlapping candidates."""
    return np.array((not_overlap_cand, overlap_cand, full_overlap_cand))


@pytest.fixture()
def full_overlap_cands(full_overlap_cand: np.array) -> np.array:
    """Pytest fixture of 3 2x2 not overlapping candidates."""
    return np.array((full_overlap_cand, full_overlap_cand, full_overlap_cand))


@pytest.fixture()
def diff_label_cands(full_overlap_cand: np.array) -> np.array:
    """Pytest fixture of 3 2x2 not overlapping candidates."""
    return np.array((full_overlap_cand * 2, full_overlap_cand * 2, full_overlap_cand * 2))
