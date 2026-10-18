"""Full-document evaluation logic (alignment_plan D1) on tiny synthetic data."""

import numpy as np
import pytest

import document_eval as de
from prepare_data import CLAUSE_TYPES

GL, NC = "Governing Law", "Non-Compete"


def _chunk(cid, title, **statuses):
    return {"chunk_id": cid, "contract_title": title,
            "labels": {ct: {"status": statuses.get(ct, "negative"), "answers": []} for ct in CLAUSE_TYPES}}


def _margins(chunks, scores):
    """scores: {(chunk_id, clause): margin}; everything else strongly negative."""
    return {c["chunk_id"]: {ct: scores.get((c["chunk_id"], ct), -5.0) for ct in CLAUSE_TYPES} for c in chunks}


CHUNKS = [
    _chunk("A0", "A"), _chunk("A1", "A", **{GL: "positive"}), _chunk("A2", "A"),
    _chunk("B0", "B"), _chunk("B1", "B"),
    _chunk("C0", "C", **{NC: "positive"}), _chunk("C1", "C", **{GL: "partial"}),
]


def test_chunk_records_exclude_partial_pairs():
    recs = de.chunk_records(CHUNKS, _margins(CHUNKS, {}))
    assert len(recs) == len(CHUNKS) * 6 - 1
    assert not any(r["query_id"] == f"C1::{GL}" for r in recs)


def test_chunk_records_apply_threshold():
    m = _margins(CHUNKS, {("A1", GL): 1.5})
    assert {r["query_id"]: r["pred"] for r in de.chunk_records(CHUNKS, m, 0.0)}[f"A1::{GL}"] == 1
    assert {r["query_id"]: r["pred"] for r in de.chunk_records(CHUNKS, m, 2.0)}[f"A1::{GL}"] == 0


def test_contract_gold_pred_and_localisation():
    m = _margins(CHUNKS, {("A1", GL): 4.0, ("A2", GL): 1.0,   # flagged twice, best chunk is the right one
                          ("B0", GL): 2.0})                    # false alarm in a contract without the clause
    recs = {r["query_id"]: r for r in de.contract_records(CHUNKS, m)}
    a = recs[f"A::{GL}"]
    assert a["label"] == 1 and a["pred"] == 1 and a["n_flagged"] == 2 and a["top_chunk_on_clause"]
    b = recs[f"B::{GL}"]
    assert b["label"] == 0 and b["pred"] == 1
    assert recs[f"A::{NC}"]["label"] == 0 and recs[f"A::{NC}"]["pred"] == 0


def test_contract_with_only_partial_match_is_skipped():
    recs = de.contract_records(CHUNKS, _margins(CHUNKS, {}))
    assert not any(r["query_id"] == f"C::{GL}" for r in recs)


def test_wrong_top_chunk_is_not_localised():
    m = _margins(CHUNKS, {("A0", GL): 6.0, ("A1", GL): 1.0})   # flagged, but the strongest chunk is the wrong one
    a = next(r for r in de.contract_records(CHUNKS, m) if r["query_id"] == f"A::{GL}")
    assert a["pred"] == 1 and a["top_chunk_on_clause"] is False


def test_summary_counts_false_alarms():
    m = _margins(CHUNKS, {("A1", GL): 3.0, ("B0", GL): 2.0, ("B1", NC): 2.0})
    s = de.summarize(de.chunk_records(CHUNKS, m), de.contract_records(CHUNKS, m))
    assert s["n_contracts"] == 3
    assert s["wrong_clause_flags_per_contract"] == pytest.approx(2 / 3)   # B flags GL and NC wrongly
    assert s["contracts_with_no_wrong_flag"] == pytest.approx(2 / 3)
    assert s["top_chunk_on_clause_rate"] == 1.0


def test_threshold_selection_prefers_best_f1_then_closest_to_zero():
    rows = [{"threshold": -1.0, "chunk_f1": 0.5}, {"threshold": 0.0, "chunk_f1": 0.8},
            {"threshold": 2.0, "chunk_f1": 0.8}, {"threshold": 4.0, "chunk_f1": 0.6}]
    assert de.select_threshold(rows)["threshold"] == 0.0


def test_sweep_trades_recall_for_precision():
    m = _margins(CHUNKS, {("A1", GL): 3.0, ("B0", GL): 1.0, ("C0", NC): 3.0})
    rows = de.sweep(CHUNKS, m, [0.0, 2.0])
    assert rows[1]["chunk_precision"] >= rows[0]["chunk_precision"]
    assert rows[1]["chunk_recall"] == rows[0]["chunk_recall"]


def test_macro_from_counts_skips_clauses_without_positives():
    c = np.zeros((len(CLAUSE_TYPES), 4))
    c[0] = [5, 0, 0, 5]          # clause 0: perfect
    c[1] = [10, 3, 0, 0]         # clause 1: no positives -> skipped even though it has false positives
    assert de.macro_from_counts(c)["f1"] == pytest.approx(1.0)


def test_bootstrap_detects_clear_difference():
    groups = {f"g{i}": f"g{i}" for i in range(30)}
    good = {g: np.tile([[8, 0, 0, 2]], (len(CLAUSE_TYPES), 1)).astype(float) for g in groups}
    bad = {g: np.tile([[4, 4, 1, 1]], (len(CLAUSE_TYPES), 1)).astype(float) for g in groups}
    res = de.paired_group_bootstrap(bad, good, n=300)
    assert res["diff_f1"] > 0.3 and res["diff_f1_ci"][0] > 0 and res["p_value_one_sided"] == 0.0
    assert res["n_groups"] == 30


def test_counts_by_group_layout():
    recs = [{"contract_title": "A", "clause_type": GL, "label": 1, "pred": 1},
            {"contract_title": "A", "clause_type": GL, "label": 0, "pred": 1},
            {"contract_title": "A", "clause_type": GL, "label": 1, "pred": 0},
            {"contract_title": "A", "clause_type": GL, "label": 0, "pred": 0}]
    c = de.counts_by_group(recs, {"A": "g"})["g"][CLAUSE_TYPES.index(GL)]
    assert list(c) == [1, 1, 1, 1]  # tn, fp, fn, tp


def test_average_precision_prefers_better_ranking_and_ignores_threshold():
    groups = {"A": "g1", "B": "g2", "C": "g3"}
    good = _margins(CHUNKS, {("A1", GL): 9.0, ("B0", GL): -1.0, ("C0", NC): 9.0})     # positives ranked first
    bad = _margins(CHUNKS, {("A1", GL): 0.5, ("B0", GL): 4.0, ("B1", GL): 3.0, ("C0", NC): 0.2, ("A0", NC): 3.0})
    ap_good = de.macro_ap(de.build_ap_data(CHUNKS, good, groups, "chunk"), ["g1", "g2", "g3"])[0]
    ap_bad = de.macro_ap(de.build_ap_data(CHUNKS, bad, groups, "chunk"), ["g1", "g2", "g3"])[0]
    assert ap_good == pytest.approx(1.0) and ap_bad < 0.8


def test_contract_level_ap_uses_best_chunk_score():
    groups = {"A": "g1", "B": "g2", "C": "g3"}
    m = _margins(CHUNKS, {("A1", GL): 5.0, ("B0", GL): 1.0})
    data = de.build_ap_data(CHUNKS, m, groups, "contract")
    y, s = data["g1"][GL]
    assert list(y) == [1] and s[0] == 5.0          # contract A has the clause; score = max chunk margin
    y, s = data["g2"][GL]
    assert list(y) == [0] and s[0] == 1.0


def test_ap_bootstrap_reports_improvement():
    groups = {f"c{i}": f"c{i}" for i in range(12)}
    chunks = [_chunk(f"c{i}-{j}", f"c{i}", **({GL: "positive"} if j == 0 else {})) for i in range(12) for j in range(3)]
    good = {c["chunk_id"]: {ct: (8.0 if c["labels"][GL]["status"] == "positive" and ct == GL else -5.0) for ct in CLAUSE_TYPES} for c in chunks}
    noisy = {c["chunk_id"]: {ct: (float(hash(c["chunk_id"]) % 7) - 3) for ct in CLAUSE_TYPES} for c in chunks}
    res = de.ap_bootstrap(de.build_ap_data(chunks, noisy, groups, "chunk"), de.build_ap_data(chunks, good, groups, "chunk"), n=50)
    assert res["diff_ap"] > 0.3 and res["b_ap"] == pytest.approx(1.0)
