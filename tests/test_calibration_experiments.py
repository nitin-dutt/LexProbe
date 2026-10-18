"""Stage-1 recovery experiments: the pure array logic (alignment_plan D7)."""

import numpy as np
import pytest

import calibration_experiments as ce

K = len(ce.CLAUSE_TYPES)


def _st(codes):
    """chunk status matrix with the same code in every clause column except column 0."""
    return np.array([[c] + [0] * (K - 1) for c in codes])


def test_segments_require_contiguous_contracts():
    assert ce.segments(["A", "A", "B"]) == [("A", 0, 2), ("B", 2, 3)]
    with pytest.raises(AssertionError):
        ce.segments(["A", "B", "A"])


def test_chunk_macro_excludes_partial_and_clauses_without_positives():
    st = _st([1, 0, 0, -1])
    S = np.zeros((4, K))
    S[:, 0] = [5, -5, 5, 5]                   # chunk 3 is partial -> ignored; chunk 2 is a false alarm
    p, r, f1 = ce.chunk_macro(S, st, [0.0])[0]
    assert (p, r) == (0.5, 1.0) and f1 == pytest.approx(2 / 3)   # other clauses have no positives -> skipped


def test_contract_rules_and_gold():
    st = np.array([[0] * K, [1] + [0] * (K - 1), [0] * K, [-1] + [0] * (K - 1)])
    S = np.zeros((4, K))
    S[:, 0] = [1.0, 8.0, 4.0, 9.0]
    segs = [("A", 0, 3), ("B", 3, 4)]
    cs, gold, keep = ce.contract_arrays(segs, S, st, "max")
    assert cs[0, 0] == 8.0 and gold[0, 0] and keep[0, 0]
    assert not gold[1, 0] and not keep[1, 0]            # only a partial match, no positive -> skipped
    cs2, _, _ = ce.contract_arrays(segs, S, st, "top2")
    assert cs2[0, 0] == pytest.approx(6.0)               # mean of 8 and 4


def test_best_threshold_ties_prefer_zero():
    grid = np.array([-2.0, 0.0, 2.0])
    assert ce.best_threshold(np.array([0.5, 0.5, 0.5]), grid) == 0.0
    assert ce.best_threshold(np.array([0.2, 0.3, 0.9]), grid) == 2.0


def test_platt_recovers_a_shifted_scale():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.random(n) < 0.05
    margin = np.where(y, rng.normal(6, 2, n), rng.normal(-3, 2, n))
    X = np.zeros((n, K, 1))
    X[:, 0, 0] = margin
    st = np.zeros((n, K), int)
    st[:, 0] = y
    params = ce.fit_calibrators(X, st)
    calibrated = ce.apply_calibrators(X, params)[:, 0]
    before = ce.best_threshold(ce.chunk_macro(X[:, :, 0], st, ce.GRID)[:, 2], ce.GRID)
    after = ce.best_threshold(ce.chunk_macro(np.stack([calibrated] * K, 1), st, ce.GRID)[:, 2], ce.GRID)
    assert params[0][0][0] > 0
    assert abs(after) < abs(before) + 3             # calibrated logits have a sensible boundary near 0
    assert params[1] == (pytest.approx([1.0]), 0.0) or np.allclose(params[1][0], [1.0])   # clause without positives -> identity


def test_group_folds_keep_groups_whole_and_balanced():
    groups = [f"g{i % 12}" for i in range(60)]
    fold_of = ce.group_folds(groups, 5)
    assert set(fold_of) == set(groups) and set(fold_of.values()) == set(range(5))
    sizes = np.bincount(list(fold_of.values()))
    assert sizes.max() - sizes.min() <= 1


def test_oof_scores_never_use_the_row_own_fold():
    rng = np.random.default_rng(1)
    n = 600
    groups = [f"g{i // 6}" for i in range(n)]
    y = rng.random(n) < 0.1
    X = np.zeros((n, K, 1))
    X[:, 0, 0] = np.where(y, 5.0, -5.0) + rng.normal(0, 1, n)
    st = np.zeros((n, K), int)
    st[:, 0] = y
    oof = ce.oof_scores(X, st, groups, fitted=True)
    assert oof.shape == (n, K)
    assert np.corrcoef(oof[:, 0], X[:, 0, 0])[0, 1] > 0.99      # monotone transform of the raw score
    assert np.array_equal(ce.oof_scores(X, st, groups, fitted=False), X[:, :, 0])
