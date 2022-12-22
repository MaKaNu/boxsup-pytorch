r"""Test Module for losses.

Each test is documented with a few explanations about the test.
This is a class based test scenario with Global test fixtures.

TestClasses:
    - TestCompareLabel
    - TestInterOUnion
    - TestOverlapping
    - TestRegression
"""

# from cmath import isclose
from typing import Dict

import numpy as np
import pytest
import torch
from torch import Tensor

from boxsup_pytorch.utils.losses import Losses


# Fixture Setup
@pytest.fixture()
def filled_config() -> Dict:
    """Pytest fixture of filled config."""
    return {"num_classes": 7}


@pytest.fixture()
def bounding_box() -> Tensor:
    """Pytest fixture of 2x2 BoundingBox."""
    return torch.tensor([[0, 1], [0, 0]])


@pytest.fixture()
def bounding_box3() -> Tensor:
    """Pytest fixture of 3x3 BoundingBox."""
    return torch.tensor([[2, 2, 2], [2, 0, 0], [2, 2, 2]])


@pytest.fixture()
def prediction_2_cls() -> Tensor:
    """Pytest fixture of 2x3 Prediction based on mask."""
    prediction_mask = np.array([[0, 1, 0], [1, 0, 1]])
    bsize = 1
    num_cls = 2
    dim = prediction_mask.shape
    pred = np.eye(num_cls, 1, k=0).flatten()
    for cls_idx in prediction_mask.flatten():
        pred = np.vstack((pred, np.eye(num_cls, 1, k=-cls_idx).flatten()))
    pred = pred[1:, :]
    return torch.tensor(pred.transpose().reshape(bsize, num_cls, *dim))


@pytest.fixture()
def label_2_cls_single() -> Tensor:
    """Pytest fixture of 2x3 Label."""
    return torch.ones(2, 3, dtype=torch.long)


@pytest.fixture()
def label_2_cls_quadruple() -> Tensor:
    """Pytest fixture of 4x2x3 Label."""
    return torch.tensor([
        [[1, 1, 1],
         [0, 0, 0]],
        [[0, 0, 0],
         [1, 1, 1]],
        [[1, 0, 1],
         [1, 0, 1]],
        [[0, 1, 0],
         [0, 1, 0]],
    ])


@pytest.fixture()
def prediction_3_cls() -> Tensor:
    """Pytest fixture of 2x2 Prediction based on mask."""
    prediction_mask = np.array([[0, 1, 0], [0, 2, 0]])
    bsize = 1
    num_cls = 3
    dim = prediction_mask.shape
    pred = np.eye(num_cls, 1, k=0).flatten()
    for cls_idx in prediction_mask.flatten():
        pred = np.vstack((pred, np.eye(num_cls, 1, k=-cls_idx).flatten()))
    pred = pred[1:, :]
    return torch.tensor(pred.transpose().reshape(bsize, num_cls, *dim))


@pytest.fixture()
def label_3_cls_single() -> Tensor:
    """Pytest fixture of 2x3 Label."""
    label = torch.ones(2, 3, dtype=torch.long)
    label[1, :] = label[1, :] * 2
    return label


@pytest.fixture()
def overlap_cand3() -> Tensor:
    """Pytest fixture of 3x3 multi class."""
    return torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


@pytest.fixture()
def overlap_cand() -> Tensor:
    """Pytest fixture of 2x2 overlapping candidate."""
    return torch.tensor([[0, 1], [0, 1]])


@pytest.fixture()
def not_overlap_cand() -> Tensor:
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
def full_overlap_cands(full_overlap_cand: Tensor) -> Tensor:
    """Pytest fixture of 3 2x2 not overlapping candidates."""
    return torch.stack((full_overlap_cand, full_overlap_cand, full_overlap_cand))


@pytest.fixture()
def diff_label_cands(full_overlap_cand):
    """Pytest fixture of 3 2x2 not overlapping candidates."""
    return torch.stack((full_overlap_cand * 2, full_overlap_cand * 2, full_overlap_cand * 2))


@pytest.fixture()
def full_overlap_diff_cands(full_overlap_cand):
    """Pytest fixture of 3 2x2 not overlapping candidates."""
    return torch.stack((full_overlap_cand * 2, full_overlap_cand))


@pytest.fixture()
def too_many_cands() -> Tensor:
    """Pytest fixture of 4x2x3x3 Zeros Tensor."""
    return torch.zeros((4, 2, 3, 3))


@pytest.fixture()
def too_few_cands() -> Tensor:
    """Pytest fixture of 4x2x3x3 Zeros Tensor."""
    return torch.zeros((3))


@pytest.fixture()
def too_many_box() -> Tensor:
    """Pytest fixture of 4x2x3x3 Zeros Tensor."""
    return torch.zeros((4, 2, 3, 3))


@pytest.fixture()
def too_few_box() -> Tensor:
    """Pytest fixture of 4x2x3x3 Zeros Tensor."""
    return torch.zeros((3))


class TestLosses:
    r"""Test Losses Instances."""

    def test_with_empty_config(self):
        """Test the Losses instance with empty config.

        expected result = self.classes == 6
        """
        instance = Losses()

        assert instance.classes == 6

    def test_with_filled_config(self, filled_config: Dict):
        """Test the Losses instance with filled config.

        expected result = self.classes == 7
        """
        instance = Losses(filled_config)

        assert instance.classes == 7


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

    def test_compare_labels_1(self, bounding_box: Tensor, overlap_cand: Tensor):
        """Test 1: same class labels.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1

        expected result = True
        """
        instance = Losses()
        assert instance._compare_labels(overlap_cand, bounding_box)

    def test_compare_labels_2(self, bounding_box: Tensor, overlap_cand: Tensor):
        """Test 2: different class labels.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 2

        expected result = False
        """
        instance = Losses()
        assert not instance._compare_labels(overlap_cand * 2, bounding_box)

    def test_compare_labels_3(self, bounding_box: Tensor, overlap_cand: Tensor):
        """Test 3: multi candidates with different class labels.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1
        - candidates: count 1, label 2

        expected result = Array[True, False]
        """
        instance = Losses()
        result = instance._compare_labels(torch.stack((
            overlap_cand, overlap_cand * 2)), bounding_box)
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

    def test_inter_o_union_1(self, bounding_box: Tensor, overlap_cand: Tensor):
        """Test 1: even overlapping.

        - bounding_box: count 1, label 1, grid 2x2
        - candidates: count 1, label 1, grid 2x2, overlapping

        expected result = 0.5
        """
        instance = Losses()
        assert instance._inter_over_union(overlap_cand, bounding_box) == torch.tensor(0.5)

    def test_inter_o_union_2(
            self, bounding_box: Tensor, full_overlap_cands: Tensor):
        """Test 2: uneven overlapping.

        - bounding_box: count 1, label 1, grid 2x2
        - candidates: count 1, label 1, grid 3x2x2, full overlapping

        expected result = Array[1,1,1]
        """
        instance = Losses()
        result = instance._inter_over_union(full_overlap_cands, bounding_box)
        assert (result == torch.tensor((1, 1, 1))).all()

    def test_inter_o_union_3(self, bounding_box3: Tensor, overlap_cand3: Tensor):
        """Test 3: multi class.

        - bounding_box: count 1, label 2, grid 3x3
        - candidates: count 1, label 1, grid 3x3, full overlapping

        expected result = 7/8
        """
        instance = Losses()
        result = instance._inter_over_union(overlap_cand3, bounding_box3)
        assert result == torch.tensor(7 / 8)


class TestOverlapping:
    r"""Test scenario for overlapping_loss.

    The loss formula:

    $\Epsilon_o = {1\over N} \sum (1 - IoU(B,S)\delta (l_B, l_S)$

    Tests using following inputs:

    Grid size is always 2x2
    """

    def test_overlapping_assert1(self, bounding_box, too_many_cands):
        """Test Assert 1.

        expect to crash with AssertionError if cands Tensor shape is not 2 or 3
        """
        instance = Losses()

        with pytest.raises(AssertionError) as errmsg:
            instance.overlapping_loss(bounding_box, too_many_cands)

        assert 'expected shape length: (2, 3)' in str(errmsg.value)

    def test_overlapping_assert2(self, bounding_box, too_few_cands):
        """Test Assert 1.

        expect to crash with AssertionError if cands Tensor shape is not 2 or 3
        """
        instance = Losses()

        with pytest.raises(AssertionError) as errmsg:
            instance.overlapping_loss(bounding_box, too_few_cands)

        assert 'expected shape length: (2, 3)' in str(errmsg.value)

    def test_overlapping_assert3(self, too_many_box, not_overlap_cands):
        """Test Assert 1.

        expect to crash with AssertionError if cands Tensor shape is not 2 or 3
        """
        instance = Losses()

        with pytest.raises(AssertionError) as errmsg:
            instance.overlapping_loss(not_overlap_cands, too_many_box)

        assert 'expected shape length: 2' in str(errmsg.value)

    def test_overlapping_assert4(self, too_few_box, not_overlap_cands):
        """Test Assert 1.

        expect to crash with AssertionError if cands Tensor shape is not 2 or 3
        """
        instance = Losses()

        with pytest.raises(AssertionError) as errmsg:
            instance.overlapping_loss(too_few_box, not_overlap_cands)

        assert 'expected shape length: 2' in str(errmsg.value)

    def test_overlapping_1(self, bounding_box, not_overlap_cands):
        """Test 1: no overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 3, label 1, no overlapping

        expected result = 1
        """
        instance = Losses()
        assert instance.overlapping_loss(bounding_box, not_overlap_cands) == 1

    def test_overlapping_2(self, bounding_box, mixed_overlap_cands):
        """Test 2: mixed overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1, no overlapping
        - candidates: count 1, label 1, overlapping
        - candidates: count 1, label 1, full overlapping

        expected result = 1/2
        """
        instance = Losses()
        assert instance.overlapping_loss(bounding_box, mixed_overlap_cands) == 1 / 2

    def test_overlapping_3(self, bounding_box, full_overlap_cands):
        """Test 3: full overlapping.array.

        - bounding_box: count 1, label 1
        - candidates: count 3, label 1, full overlapping

        expected result = 0
        """
        instance = Losses()
        assert instance.overlapping_loss(bounding_box, full_overlap_cands) == 0

    def test_overlapping_4(self, bounding_box, diff_label_cands):
        """Test 4: diff overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 3, label 2, full overlapping

        expected result = 0
        """
        instance = Losses()
        assert instance.overlapping_loss(bounding_box, diff_label_cands) == 0

    def test_overlapping_5(self, bounding_box, not_overlap_cand):
        """Test 5: no overlapping single.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1, no overlapping

        expected result = 1
        """
        instance = Losses()
        assert instance.overlapping_loss(bounding_box, not_overlap_cand) == 1

    def test_overlapping_6(self, bounding_box, full_overlap_diff_cands):
        """Test 6: full overlap diff classes.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1, no overlapping

        expected result = 1
        """
        instance = Losses()
        assert instance.overlapping_loss(bounding_box, full_overlap_diff_cands) == 0


class TestRegression:
    r"""Test scenario for regression_loss.

    The loss formula:

    $\Epsilon(\phi) = \sum_p e(X_\phi(p), l_S(p))$
    """

    def test_regression_1(self, prediction_2_cls, label_2_cls_single):
        """Test 1: Single label half overlapping.

        - prediction: count 1, label 1
        - candidates: count 1, label 1, half overlapping

        expected result = 0.8133

        target explanation:

        - Ignoring Background with idx=0
        - faked loftsoftmax set to one class = 1
        - the target_0 is the negative logsoftmax of 0 for 2 classes
        - the target_1 is the negative logsoftmax of 1 for 2 classes
        """
        target_0 = -np.log(np.exp(1) / (np.exp(1) + np.exp(0)))
        target_1 = -np.log(np.exp(0) / (np.exp(1) + np.exp(0)))
        target = (3*target_0 + 3*target_1)/6
        instance = Losses()
        result = instance.regression_loss(prediction_2_cls, label_2_cls_single)
        assert result == pytest.approx(target)

    def test_regression_2(self, prediction_2_cls, label_2_cls_quadruple):
        """Test 2: quadruple label different overlapping.

        - prediction: count 1, label 1
        - candidates: count 3, label 1, different overlapping

        expected result = [0.4900, 0.3233, 0.5422, 0.2711]

        target explanation:

        - Ignoring Background with idx=0
        - faked loftsoftmax set to one class = 1
        - the target_0 is the negative logsoftmax of 0 for 2 classes
        - the target_1 is the negative logsoftmax of 1 for 2 classes
        """
        target_0 = -np.log(np.exp(1) / (np.exp(1) + np.exp(0)))
        target_1 = -np.log(np.exp(0) / (np.exp(1) + np.exp(0)))
        target = [(1*target_0 + 2*target_1 + 3*0) / 6,
                  (2*target_0 + 1*target_1 + 3*0) / 6,
                  (2*target_0 + 2*target_1 + 2*0) / 6,
                  (1*target_0 + 1*target_1 + 4*0) / 6
                  ]
        instance = Losses()
        result = instance.regression_loss(prediction_2_cls, label_2_cls_quadruple)
        for values in zip(result, target):
            assert values[0] == pytest.approx(values[1])

    def test_regression_4(self, prediction_3_cls, label_3_cls_single):
        """Test 3: Test 3 classes.

        - prediction: count 1, label 1
        - candidates: count 1, label 2

        expected result = array[0.3283, 0.4066, 0.0783]

        target explanation:

        - Ignoring Background with idx=0
        - faked loftsoftmax set to one class = 1
        - the target_0 is the negative logsoftmax of 0 for 3 classes
        - the target_1 is the negative logsoftmax of 1 for 3 classes
        """
        target_1 = -np.log(np.exp(1) / (np.exp(1) + 2 * np.exp(0)))
        target_0 = -np.log(np.exp(0) / (np.exp(1) + 2 * np.exp(0)))
        target = (2 * target_1 + 4 * target_0) / 6
        instance = Losses()
        result = instance.regression_loss(prediction_3_cls, label_3_cls_single)
        assert result == pytest.approx(target)


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
        instance = Losses()
        assert instance.weighted_loss(Tensor([3]), Tensor([4])) == 15

    def test_weighted_2(self):
        """Test 2: custom weight.

        - epsilon_o: 3
        - epsilon_r: 4
        - weight: 2

        expected result = 11
        """
        instance = Losses()
        assert instance.weighted_loss(Tensor([3]), Tensor([4]), 2) == 11
