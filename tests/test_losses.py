r"""Test Module for losses.

Each test is documented with a few explanations about the test.
This is a class based test scenario with Global test fixtures.

TestClasses:
    - TestCompareLabel
    - TestInterOUnion
    - TestOverlapping
    - TestRegression
"""

from cmath import isclose

import numpy as np
import pytest
import torch

from boxsup_pytorch.losses import (
    compare_labels,
    inter_o_union,
    overlapping_loss,
    regression_loss,
    weighted_loss,
)


# Fixture Setup
@pytest.fixture()
def bounding_box() -> torch.Tensor:
    """Pytest fixture of 2x2 BoundingBox."""
    return torch.tensor([[0, 1], [0, 0]])


@pytest.fixture()
def bounding_box3() -> torch.Tensor:
    """Pytest fixture of 3x3 BoundingBox."""
    return torch.tensor([[1, 1, 1], [1, 0, 2], [2, 2, 2]])


@pytest.fixture()
def prediction_2_cls() -> torch.Tensor:
    """Pytest fixture of 2x2 Prediction based on mask."""
    prediction_mask = torch.tensor([[0, 1], [0, 0]])
    bsize = 1
    num_cls = 2
    dim = prediction_mask.shape
    for cls_idx in prediction_mask.flatten():
        if "pred" not in vars():
            pred = np.eye(num_cls, 1, k=-cls_idx).flatten()
            continue
        pred = np.vstack((pred, np.eye(num_cls, 1, k=-cls_idx).flatten()))

    return torch.tensor(pred.transpose().reshape(bsize, num_cls, *dim))


@pytest.fixture()
def prediction_3_cls() -> np.array:
    """Pytest fixture of 2x2 Prediction based on mask."""
    prediction_mask = np.array([[0, 1], [0, 0]])
    bsize = 1
    num_cls = 3
    dim = prediction_mask.shape
    for cls_idx in prediction_mask.flatten():
        if "pred" not in vars():
            pred = np.eye(num_cls, 1, k=-cls_idx).flatten()
            continue
        pred = np.vstack((pred, np.eye(num_cls, 1, k=-cls_idx).flatten()))
    return torch.tensor(pred.transpose().reshape(bsize, num_cls, *dim))


@pytest.fixture()
def multi_cand() -> np.array:
    """Pytest fixture of 3x3 multi class."""
    return torch.tensor([[1, 1, 1], [2, 0, 1], [2, 2, 2]])


@pytest.fixture()
def overlap_cand() -> np.array:
    """Pytest fixture of 2x2 overlapping candidate."""
    return torch.tensor([[0, 1], [0, 1]])


@pytest.fixture()
def not_overlap_cand() -> np.array:
    """Pytest fixture of 2x2 not overlapping candidate."""
    return torch.tensor([[0, 0], [1, 0]])


@pytest.fixture()
def full_overlap_cand():
    """Pytest fixture of 2x2 full overlapping candidate."""
    return torch.tensor([[0, 1], [0, 0]])


@pytest.fixture()
def not_overlap_cands(not_overlap_cand):
    """Pytest fixture of 3 2x2 not overlapping candidates."""
    return torch.stack((not_overlap_cand, not_overlap_cand, not_overlap_cand))


@pytest.fixture()
def mixed_overlap_cands(
    not_overlap_cand, overlap_cand, full_overlap_cand
):
    """Pytest fixture of 3 2x2 mixed overlapping candidates."""
    return torch.stack((not_overlap_cand, overlap_cand, full_overlap_cand))


@pytest.fixture()
def full_overlap_cands(full_overlap_cand: np.array) -> np.array:
    """Pytest fixture of 3 2x2 not overlapping candidates."""
    return torch.stack((full_overlap_cand, full_overlap_cand, full_overlap_cand))


@pytest.fixture()
def diff_label_cands(full_overlap_cand):
    """Pytest fixture of 3 2x2 not overlapping candidates."""
    return torch.stack((full_overlap_cand * 2, full_overlap_cand * 2, full_overlap_cand * 2))


class TestCompareLabel:
    r"""Test scenario for compare labels.

    The formula:

    $\delta (l_B, l_S) =
    \begin{cases}
    1 & \text{if } l_B = l_S \\
    0 & \text{otherwise.}
    \end{cases}$

    Test using following inputs:

    Grid size is always 2x2
    """

    def test_compare_labels_1(self, bounding_box: np.array, overlap_cand: np.array):
        """Test 1: same class labels.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1

        expected result = True
        """
        assert compare_labels(bounding_box, overlap_cand)

    def test_compare_labels_2(self, bounding_box: np.array, overlap_cand: np.array):
        """Test 2: different class labels.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 2

        expected result = False
        """
        assert not compare_labels(bounding_box, overlap_cand * 2)

    def test_compare_labels_3(self, bounding_box, overlap_cand, ):
        """Test 3: multi candidates with different class labels.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1
        - candidates: count 1, label 2

        expected result = Array[True, False]
        """
        result = compare_labels(bounding_box, torch.stack((overlap_cand, overlap_cand * 2)))
        assert (result == torch.tensor((True, False))).all()


class TestInterOUnion:
    r"""Test scenario for intersection over union.

    The formula:

    $IoU (B, S) =
    {sum_of_intersecting_pixel\over sum_of_union_pixel}
    $

    $sum_of_intersecting_pixel =
    sum(B(l_B)\wedge S(l_S))
    $

    $sum_of_union_pixel =
    sum(B(l_B)\vee S(l_S))
    $

    Tests using following inputs:

    Grid size is 2x2 and 3x3
    """

    def test_inter_o_union_1(self, bounding_box, overlap_cand, ):
        """Test 1: even overlapping.

        - bounding_box: count 1, label 1, grid 2x2
        - candidates: count 1, label 1, grid 2x2, overlapping

        expected result = 0.5
        """
        assert inter_o_union(bounding_box, overlap_cand) == 0.5

    def test_inter_o_union_2(self, bounding_box, full_overlap_cands, ):
        """Test 2: uneven overlapping.

        - bounding_box: count 1, label 1, grid 2x2
        - candidates: count 1, label 1, grid 3x2x2, full overlapping

        expected result = Array[1,1,1]
        """
        result = inter_o_union(bounding_box, full_overlap_cands)
        assert (result == torch.tensor((1, 1, 1))).all()

    def test_inter_o_union_3(self, bounding_box3, multi_cand, ):
        """Test 3: multi class.

        - bounding_box: count 1, label 1, grid 3x3
        - candidates: count 1, label 1, grid 3x3, full overlapping

        expected result = 3/5
        """
        assert inter_o_union(bounding_box3, multi_cand) == 3 / 5


class TestOverlapping:
    r"""Test scenario for overlapping_loss.

    The loss formula:

    $\Epsilon_o = {1\over N} \sum (1 - IoU(B,S)\delta (l_B, l_S)$

    Tests using following inputs:

    Grid size is always 2x2
    """

    def test_overlapping_1(self, bounding_box, not_overlap_cands, ):
        """Test 1: no overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 3, label 1, no overlapping

        expected result = 1
        """
        assert overlapping_loss(bounding_box, not_overlap_cands) == 1

    def test_overlapping_2(self, bounding_box, mixed_overlap_cands, ):
        """Test 2: mixed overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1, no overlapping
        - candidates: count 1, label 1, overlapping
        - candidates: count 1, label 1, full overlapping

        expected result = 1/2
        """
        assert overlapping_loss(bounding_box, mixed_overlap_cands) == 1 / 2

    def test_overlapping_3(self, bounding_box, full_overlap_cands, ):
        """Test 3: full overlapping.array

        - bounding_box: count 1, label 1
        - candidates: count 3, label 1, full overlapping

        expected result = 0
        """
        assert overlapping_loss(bounding_box, full_overlap_cands) == 0

    def test_overlapping_4(self, bounding_box, diff_label_cands, ):
        """Test 4: diff overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 3, label 2, full overlapping

        expected result = 0
        """
        assert overlapping_loss(bounding_box, diff_label_cands) == 0

    def test_overlapping_5(self, bounding_box, not_overlap_cand, ):
        """Test 5: no overlapping single.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1, no overlapping

        expected result = 1
        """
        assert overlapping_loss(bounding_box, not_overlap_cand) == 1


class TestRegression:
    r"""Test scenario for regression_loss.

    The loss formula:

    $\Epsilon(\phi) = \sum_p e(X_\phi(p), l_S(p))$
    """

    def test_regression_1(self, prediction_2_cls, not_overlap_cands):
        """Test 1: no overlapping.

        - prediction: count 1, label 1
        - candidates: count 3, label 1, no overlapping

        expected result = array[0.3283, 0.3283, 0.3283]
        """
        # Ignoring Background with idx=0
        # since only one field on the candidates is cls idx 1
        # which is not 1 on the prediction
        # the target_0 is the negative logsoftmax of 0
        target_0 = np.mean((-np.log(np.exp(0) / (np.exp(1) + np.exp(0))), 0, 0, 0))
        result = regression_loss(prediction_2_cls, not_overlap_cands)
        target_array = np.array((target_0, target_0, target_0))
        for values in zip(result, target_array):
            assert isclose(*values, abs_tol=1e-15)

    def test_regression_2(self, prediction_2_cls: np.array, full_overlap_cands: np.array):
        """Test 2: full overlapping.

        - prediction: count 1, label 1
        - candidates: count 3, label 1, full overlapping

        expected result = array[0.0783, 0.0783, 0.0783]
        """
        # Ignoring Background with idx=0
        # since only one field on the candidates is cls idx 1
        # which is 1 on the prediction
        # the target_1 is the negative logsoftmax of 1
        target_1 = np.mean((-np.log(np.exp(1) / (np.exp(1) + np.exp(0))), 0, 0, 0))
        result = regression_loss(prediction_2_cls, full_overlap_cands)
        target_array = np.array((target_1, target_1, target_1))
        for values in zip(result, target_array):
            assert isclose(*values, abs_tol=1e-15)

    def test_regression_3(self, prediction_2_cls: np.array, mixed_overlap_cands: np.array):
        """Test 3: mixed overlapping.

        - prediction: count 1, label 1
        - candidates: count 1, label 1, no overlapping
        - candidates: count 1, label 1, overlapping
        - candidates: count 1, label 1, full overlapping

        expected result = array[0.3283, 0.4066, 0.0783]
        """
        # Ignoring Background with idx=0
        # since upto two fields on the candidates are cls idx 1
        # where one is 1 and the other is 0 on the prediction
        # the target_0 is the negative logsoftmax of 0
        # the target_1 is the negative logsoftmax of 1
        target_0 = np.mean((-np.log(np.exp(0) / (np.exp(1) + np.exp(0))), 0, 0, 0))
        target_1 = np.mean((-np.log(np.exp(1) / (np.exp(1) + np.exp(0))), 0, 0, 0))
        result = regression_loss(prediction_2_cls, mixed_overlap_cands)
        target_array = np.array((target_0, target_0 + target_1, target_1))
        for values in zip(result, target_array):
            assert isclose(*values, abs_tol=1e-15)

    def test_regression_4(self, prediction_3_cls: np.array, full_overlap_cand: np.array):
        """Test 4: diff overlapping.

        - prediction: count 1, label 1
        - candidates: count 1, label 2, full overlapping

        expected result = array[0.3283, 0.4066, 0.0783]
        """
        # Ignoring Background with idx=0
        # since only one field on the candidates is cls idx 2
        # but the corresponding prediction fireld is 1
        # the target_0 is the negative logsoftmax of 0
        target_0 = np.mean(
            (
                -np.log(np.exp(0) / np.sum(np.exp(np.array((0, 1, 0))))),  # Now three Classes
                0,  # Background
                0,  # Backgorund
                0,  # Background
            )
        )
        result = regression_loss(prediction_3_cls, full_overlap_cand * 2)
        assert isclose(result, target_0, abs_tol=1e-15)


class TestWeightedLoss:
    r"""Test scenario for weighted_loss.

    The loss formula:

    $\Epsilon = \Epsilon_o + \lamda\Epsilon_r$
    """

    def test_weighted_1(self):
        """Test 1: standard weight.

        - epsilon_o: 3
        - epsilon_r: 4

        expected result = 15
        """
        assert weighted_loss(3, 4) == 15

    def test_weighted_2(self):
        """Test 2: custom weight.

        - epsilon_o: 3
        - epsilon_r: 4
        - weight: 2

        expected result = 11
        """
        assert weighted_loss(3, 4, 2) == 11
