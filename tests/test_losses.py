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
