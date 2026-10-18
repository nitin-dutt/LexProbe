"""The blend that the API serves (alignment_plan D7): config, maths, reproduction of the reported result."""

import contextlib
import copy
import json
import math
import os

import numpy as np
import pytest

import blend
import calibration_experiments as ce
import chunk_scoring
from inference import analyze_document, analyze_scored
from prepare_data import CLAUSE_TYPES

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONFIG = os.path.join(ROOT, "configs", "blend_v1.json")


def _cfg():
    return blend.load_blend(CONFIG)


# ── config ─────────────────────────────────────────────────────────────────────
def test_config_is_valid_and_complete():
    cfg = _cfg()
    assert cfg["rule"] == "max" and math.isfinite(cfg["threshold"])
    assert set(cfg["clause_params"]) == set(CLAUSE_TYPES)


def test_config_matches_the_stage1_report():
    """The deployed weights must be exactly the ones the reported result was computed with."""
    report = json.load(open(os.path.join(ROOT, "reports", "eval", "recovery_stage1.json"), encoding="utf-8"))
    cfg = _cfg()
    for ct in CLAUSE_TYPES:
        for k in ("w_v4", "w_baseline", "intercept"):
            assert cfg["clause_params"][ct][k] == pytest.approx(report["blend_parameters_fitted_on_all_validation"][ct][k], abs=1e-9)
    assert cfg["threshold"] == report["selected"]["threshold"] and cfg["rule"] == report["selected"]["rule"]


def test_config_reproduces_the_reported_test_result_from_saved_scores():
    """Blend the saved v4 and zero-shot test margins with the exported config -> the reported contract F1."""
    cfg = _cfg()
    test = ce.Split("test")
    v4, base = test.scores("lora_v4"), test.scores("baseline")
    S = np.zeros_like(v4)
    for i in range(len(v4)):
        scores = blend.blend_scores(dict(zip(CLAUSE_TYPES, v4[i])), dict(zip(CLAUSE_TYPES, base[i])), cfg)
        S[i] = [scores[ct] for ct in CLAUSE_TYPES]
    cs, gold, keep = ce.contract_arrays(test.segments, S, test.status, cfg["rule"])
    f1 = ce.contract_macro(cs, gold, keep, [cfg["threshold"]])[0, 2]
    report = json.load(open(os.path.join(ROOT, "reports", "eval", "recovery_stage1.json"), encoding="utf-8"))
    assert f1 == pytest.approx(report["selected"]["test_contract_f1"], abs=1e-9)
    assert f1 == pytest.approx(0.828, abs=1e-3)


# ── maths and validation ───────────────────────────────────────────────────────
def _toy_cfg():
    return {"name": "t", "rule": "max", "threshold": 0.0, "adapter": "x",
            "clause_params": {ct: {"w_v4": 1.0, "w_baseline": 0.5, "intercept": -2.0} for ct in CLAUSE_TYPES}}


def test_blend_scores_are_the_weighted_sum():
    cfg = _toy_cfg()
    out = blend.blend_scores({ct: 4.0 for ct in CLAUSE_TYPES}, {ct: -2.0 for ct in CLAUSE_TYPES}, cfg)
    assert all(v == pytest.approx(1.0 * 4.0 + 0.5 * -2.0 - 2.0) for v in out.values())


@pytest.mark.parametrize("mutate", [
    lambda c: c["clause_params"].pop("Governing Law"),
    lambda c: c["clause_params"]["Non-Compete"].update(w_v4=float("nan")),
    lambda c: c.update(rule="top2"),
    lambda c: c.update(threshold=float("inf")),
    lambda c: c.pop("adapter"),
])
def test_invalid_configs_are_rejected(mutate):
    cfg = copy.deepcopy(_toy_cfg())
    mutate(cfg)
    with pytest.raises(ValueError):
        blend.validate_blend(cfg)


# ── analysis logic ─────────────────────────────────────────────────────────────
def test_analyze_scored_matches_the_per_clause_form():
    clause = "This Agreement shall be governed by the laws of the State of Delaware."
    doc = ("The parties agree to cooperate in good faith. " * 120) + "\n\n" + clause + "\n\n" + ("Other terms apply. " * 80)
    decide = lambda ct, ch: 3.0 if ct == "Governing Law" and "governed by the laws" in ch else -3.0
    quote = lambda ct, ch: " " + clause
    a = analyze_document(doc, decide, quote, threshold=0.0)
    b = analyze_scored(doc, lambda ch: {ct: decide(ct, ch) for ct in CLAUSE_TYPES}, quote, threshold=0.0)
    assert a == b


# ── GPU glue, with a fake model ────────────────────────────────────────────────
class _FakeModel:
    def __init__(self):
        self.adapter_enabled = True

    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = True


def test_blend_passes_run_adapter_on_then_off_and_always_restore(monkeypatch):
    model, seen = _FakeModel(), []

    def fake_margins(m, tok, found, none, text, clause_types=CLAUSE_TYPES):
        seen.append(m.adapter_enabled)
        return {ct: float(len(seen)) for ct in clause_types}

    monkeypatch.setattr(chunk_scoring, "chunk_margins", fake_margins)
    v4, base = chunk_scoring.blend_chunk_margins(model, None, 1, 2, "text")
    assert seen == [True, False] and model.adapter_enabled
    assert v4["Governing Law"] == 1.0 and base["Governing Law"] == 2.0

    def boom(*a, **k):
        if not model.adapter_enabled:
            raise RuntimeError("fail while the adapter is off")
        return {ct: 0.0 for ct in CLAUSE_TYPES}

    monkeypatch.setattr(chunk_scoring, "chunk_margins", boom)
    with pytest.raises(RuntimeError):
        chunk_scoring.blend_chunk_margins(model, None, 1, 2, "text")
    assert model.adapter_enabled  # restored even after an exception
