"""Evaluation metric helpers (alignment_plan D2, D3)."""

import pytest

from eval_metrics import clean_quote, grounding, mcnemar, paired_bootstrap, per_clause_metrics, token_f1


def _rec(i, clause, label, pred, title=None):
    return {"query_id": f"q{i}", "contract_title": title or f"C{i % 10}", "clause_type": clause,
            "label": label, "pred": pred}


def test_per_clause_macro_skips_clauses_without_positives():
    recs = [_rec(0, "A", 1, 1), _rec(1, "A", 0, 0), _rec(2, "B", 0, 1)]
    m = per_clause_metrics(recs)
    assert "B" not in m and m["macro"]["f1"] == pytest.approx(1.0)


def test_grounding_exact_and_fuzzy():
    src = "This Agreement shall be governed by   the laws of Delaware."
    assert grounding("this agreement shall be governed by the laws of Delaware", src)["exact"]
    assert not grounding("The parties agree to arbitration in Paris.", src)["grounded"]


def test_clean_quote_strips_prefix_and_ellipsis():
    assert clean_quote('Found: "The laws of Delaware..."') == "The laws of Delaware"


def test_token_f1():
    assert token_f1("laws of Delaware", "the laws of Delaware") == pytest.approx(2 * 1.0 * 0.75 / 1.75)
    assert token_f1("", "x") == 0.0


def test_paired_bootstrap_detects_clear_improvement():
    a = [_rec(i, "A", i % 2, 0) for i in range(200)]               # never finds anything
    b = [_rec(i, "A", i % 2, i % 2) for i in range(200)]            # perfect
    groups = {f"C{i}": f"C{i}" for i in range(10)}
    res = paired_bootstrap(a, b, groups, n=200)
    assert res["diff"] == pytest.approx(1.0) and res["p_value_one_sided"] == 0.0
    assert res["diff_ci"][0] > 0


def test_mcnemar_counts():
    a = [_rec(i, "A", 1, 0) for i in range(10)]
    b = [_rec(i, "A", 1, 1) for i in range(10)]
    res = mcnemar(a, b)
    assert res["only_b_correct"] == 10 and res["only_a_correct"] == 0 and res["p_value_two_sided"] < 0.01
