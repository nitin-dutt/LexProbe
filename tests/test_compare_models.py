import numpy as np

from compare_models import average_precision


def test_ap_perfect_and_worst():
    y = np.array([1, 1, 0, 0])
    assert average_precision(y, np.array([4.0, 3.0, 2.0, 1.0])) == 1.0
    assert average_precision(y, np.array([1.0, 2.0, 3.0, 4.0])) < 0.5


def test_ties_are_broken_against_the_model():
    y = np.array([1, 0, 0, 0])
    assert average_precision(y, np.zeros(4)) == 0.25  # all tied: the positive is placed last, not first


def test_no_positives_gives_nan():
    assert np.isnan(average_precision(np.array([0, 0]), np.array([1.0, 2.0])))
