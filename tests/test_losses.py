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
import torch

from boxsup_pytorch.losses import (
    compare_labels,
    inter_o_union,
    overlapping_loss,
    regression_loss,
    weighted_loss,
)


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

    def test_compare_labels_same_classes(self):
        """Test 1: same class labels.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1

        expected result = True
        """
        bbox = torch.Tensor([1])
        candidate = torch.Tensor([1])
        assert compare_labels(bbox, candidate)

    def test_compare_labels_different_classes(self):
        """Test 2: different class labels.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 2

        expected result = False
        """
        bbox = torch.Tensor([1])
        candidate = torch.Tensor([2])
        assert not compare_labels(bbox, candidate)

    def test_compare_labels_multi_different_classes(self):
        """Test 3: multi candidates with different class labels.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1
        - candidates: count 1, label 2

        expected result = Array[True, False]
        """
        bbox = torch.Tensor([1])
        candidate = torch.Tensor([1, 2])
        result = compare_labels(bbox, candidate)
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
    """

    def test_inter_o_union_overlap(self):
        """Test 1: even overlapping.

        - bounding_box: count 1, label 1, grid 2x2
        - candidates: count 1, label 1, grid 2x2, overlapping

        expected result = 0.5
        """
        bbox = torch.Tensor([1])
        candidate = torch.Tensor([1, 3])
        assert inter_o_union(bbox, candidate) == 0.5

    def test_inter_o_union_no_overlap(self):
        """Test 1: even overlapping.

        - bounding_box: count 1, label 1, grid 2x2
        - candidates: count 1, label 1, grid 2x2, no overlap

        expected result = 0
        """
        bbox = torch.Tensor([1])
        candidate = torch.Tensor([0, 2])
        assert inter_o_union(bbox, candidate) == 0

    def test_inter_o_union_empty_mask(self):
        """Test 1: even overlapping.

        - bounding_box: count 1, label 1, grid 2x2
        - candidates: count 1, label 1, grid 2x2, overlapping

        expected result = 0.5
        """
        bbox = torch.Tensor([])
        candidate = torch.Tensor([])
        assert inter_o_union(bbox, candidate) == 0


class TestOverlapping:
    r"""Test scenario for overlapping_loss.

    The loss formula:

    $\Epsilon_o = {1\over N} \sum (1 - IoU(B,S)\delta (l_B, l_S)$

    Tests using following inputs:

    Grid size is always 2x2
    """

    def test_overlapping_no_overlap(self):
        """Test 1: no overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 3, label 1, no overlapping

        expected result =bounding_box, not_overlap_cands 1
        """
        bbox = (1, torch.Tensor([1]))
        candidates = [
            (1, torch.Tensor([2])),
            (1, torch.Tensor([2])),
            (1, torch.Tensor([2]))
        ]
        assert overlapping_loss(bbox, candidates) == 1

    def test_overlapping_mixed_overlap(self):
        """Test 2: mixed overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1, no overlapping
        - candidates: count 1, label 1, overlapping
        - candidates: count 1, label 1, full overlapping

        expected result = 1/2
        """
        bbox = (1, torch.Tensor([1]))
        candidates = [
            (1, torch.Tensor([2])),
            (1, torch.Tensor([1, 3])),
            (1, torch.Tensor([1]))
        ]
        assert overlapping_loss(bbox, candidates) == 1 / 2

    def test_overlapping_full_overlap(self):
        """Test 3: full overlapping.array.

        - bounding_box: count 1, label 1
        - candidates: count 3, label 1, full overlapping

        expected result = 0
        """
        bbox = (1, torch.Tensor([1]))
        candidates = [
            (1, torch.Tensor([1])),
            (1, torch.Tensor([1])),
            (1, torch.Tensor([1]))
        ]
        assert overlapping_loss(bbox, candidates) == 0

    def test_overlapping_different_class(self):
        """Test 4: diff overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 3, label 2, full overlapping

        expected result = 0
        """
        bbox = (1, torch.Tensor([1]))
        candidates = [
            (2, torch.Tensor([1])),
            (2, torch.Tensor([1])),
            (2, torch.Tensor([1]))
        ]
        assert overlapping_loss(bbox, candidates) == 0

    def test_overlapping_no_overlap_single(self):
        """Test 5: no overlapping single.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1, no overlapping

        expected result = 1
        """
        bbox = (1, torch.Tensor([1]))
        candidates = [
            (1, torch.Tensor([2])),
        ]
        assert overlapping_loss(bbox, candidates) == 1


class TestRegression:
    r"""Test scenario for regression_loss.

    The loss formula:

    $\Epsilon(\phi) = \sum_p e(X_\phi(p), l_S(p))$
    """

    def test_regression_no_overlap(self):
        """Test 1: no overlapping.

        - prediction: count 1, label 1
        - candidates: count 3, label 1, no overlapping

        expected result = array[0.3283, 0.3283, 0.3283]
        """
        # Ignoring Background with idx=0
        # since only one field on the candidates is cls idx 1
        # which is not 1 on the prediction
        # the target_0 is the negative logsoftmax of 0
        pred = torch.Tensor(
            [[
                [
                    [1, 0],
                    [1, 1]
                ],
                [
                    [0, 1],
                    [0, 0],
                ],
            ]]
        )
        candidates = [
            (1, torch.Tensor([2]).to(torch.long)),
            (1, torch.Tensor([2]).to(torch.long)),
            (1, torch.Tensor([2]).to(torch.long))
        ]
        result = regression_loss(pred, candidates)
        target_0 = -np.log(np.exp(0) / (np.exp(1) + np.exp(0)))
        target_array = (target_0, target_0, target_0)
        for values in zip(result, target_array):
            assert isclose(*values, abs_tol=1e-7)

    def test_regression_full_overlap(self):
        """Test 2: full overlapping.

        - prediction: count 1, label 1
        - candidates: count 3, label 1, full overlapping

        expected result = array[0.0783, 0.0783, 0.0783]
        """
        # Ignoring Background with idx=0
        # since only one field on the candidates is cls idx 1
        # which is 1 on the prediction
        # the target_1 is the negative logsoftmax of 1
        pred = torch.Tensor(
            [[
                [
                    [1, 0],
                    [1, 1]
                ],
                [
                    [0, 1],
                    [0, 0],
                ],
            ]]
        )
        candidates = [
            (1, torch.Tensor([1]).to(torch.long)),
            (1, torch.Tensor([1]).to(torch.long)),
            (1, torch.Tensor([1]).to(torch.long))
        ]
        result = regression_loss(pred, candidates)
        target_1 = -np.log(np.exp(1) / (np.exp(1) + np.exp(0)))
        target_array = (target_1, target_1, target_1)
        for values in zip(result, target_array):
            assert isclose(*values, abs_tol=1e-7)

    def test_regression_mixed(self):
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
        pred = torch.Tensor(
            [[
                [
                    [1, 0],
                    [1, 1]
                ],
                [
                    [0, 1],
                    [0, 0],
                ],
            ]]
        )
        candidates = [
            (1, torch.Tensor([2]).to(torch.long)),
            (1, torch.Tensor([1, 3]).to(torch.long)),
            (1, torch.Tensor([1]).to(torch.long))
        ]
        result = regression_loss(pred, candidates)
        target_0 = -np.log(np.exp(0) / (np.exp(1) + np.exp(0)))
        target_1 = np.mean((
            -np.log(np.exp(1) / (np.exp(1) + np.exp(0))),
            -np.log(np.exp(0) / (np.exp(1) + np.exp(0)))
        ))
        target_2 = -np.log(np.exp(1) / (np.exp(1) + np.exp(0)))
        target_array = (target_0, target_1, target_2)
        for values in zip(result, target_array):
            assert isclose(*values, abs_tol=1e-7)

    def test_regression_diff_overlap(self):
        """Test 4: diff overlapping.

        - prediction: count 1, label 1
        - candidates: count 1, label 2, full overlapping

        expected result = array[0.3283, 0.4066, 0.0783]
        """
        # Ignoring Background with idx=0
        # since only one field on the candidates is cls idx 2
        # but the corresponding prediction fireld is 1
        # the target_0 is the negative logsoftmax of 0
        pred = torch.Tensor(
            [[
                [
                    [1, 0],
                    [1, 1]
                ],
                [
                    [0, 1],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 0]
                ]
            ]]
        )
        candidates = [
            (2, torch.Tensor([1]).to(torch.long)),
        ]
        result = regression_loss(pred, candidates)
        target_0 = -np.log(np.exp(0) / np.sum(np.exp(np.array((0, 1, 0)))))

        assert isclose(result[0], target_0, abs_tol=1e-7)


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
        assert weighted_loss(3, [4]) == 15

    def test_weighted_2(self):
        """Test 2: custom weight.

        - epsilon_o: 3
        - epsilon_r: 4
        - weight: 2

        expected result = 11
        """
        assert weighted_loss(3, [4], 2) == 11
