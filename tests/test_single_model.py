"""The deployed v6b single-pass scoring config (src/single_model.py, configs/model_v6b.json)."""

import json
import os
import subprocess
import sys

import pytest

import single_model as sm

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = os.path.join(ROOT, "configs", "model_v6b.json")
CLAUSES = ["Anti-Assignment", "Auto-Renewal", "Governing Law", "Limitation of Liability",
           "Liquidated Damages", "Non-Compete", "Termination for Convenience"]


def toy():
    return {"name": "t", "profile": "v4", "adapter": "x", "rule": "max", "threshold": -2.0,
            "clause_shift": {"Governing Law": 2.0, "Non-Compete": 4.0}}


def test_scores_are_margin_minus_shift():
    out = sm.single_scores({"Governing Law": 5.0, "Non-Compete": 1.0, "Extra": 9.0}, toy())
    assert out == {"Governing Law": 3.0, "Non-Compete": -3.0}  # only the config's clauses, shifted


def test_missing_clause_margin_is_an_error():
    with pytest.raises(KeyError):
        sm.single_scores({"Governing Law": 1.0}, toy())


@pytest.mark.parametrize("change", [{"rule": "top2"}, {"clause_shift": {}}, {"clause_shift": {"Governing Law": "x"}}])
def test_bad_configs_are_refused(change):
    cfg = {**toy(), **change}
    with pytest.raises(ValueError):
        sm.validate_single(cfg)
    with pytest.raises(ValueError):
        sm.validate_single({k: v for k, v in toy().items() if k != "threshold"})


def test_shipped_config_is_valid_and_matches_the_validated_report():
    cfg = sm.load_single(CFG)
    assert cfg["profile"] == "v4" and sorted(cfg["clause_shift"]) == CLAUSES
    assert cfg["threshold"] == -2.0 and cfg["adapter"].endswith("lora-v6b/final_adapter")
    report = json.load(open(os.path.join(ROOT, cfg["source_report"]), encoding="utf-8"))
    assert report["selected"]["candidate"] == "lora_v4:shift" and report["selected"]["threshold"] == cfg["threshold"]
    assert cfg["test_result_at_selection"]["contract_f1"] == pytest.approx(report["selected"]["test_contract_f1"])
    assert cfg["selection"]["stable_across_fold_seeds"] == ["lora_v4:shift"]


SCRIPT = r"""
import json, sys
sys.path.insert(0, "src")
import numpy as np
import calibration_experiments as ce
import single_model as sm
from prepare_data import CLAUSE_TYPES
cfg = sm.load_single("configs/model_v6b.json")
assert list(CLAUSE_TYPES) == sorted(cfg["clause_shift"]), CLAUSE_TYPES
test = ce.Split("test")
ce.ADAPTER_SYSTEM = "lora_v6b"
m = test.scores("lora_v4")
S = np.zeros_like(m)
for i in range(len(m)):
    sc = sm.single_scores(dict(zip(CLAUSE_TYPES, m[i])), cfg)
    S[i] = [sc[c] for c in CLAUSE_TYPES]
cs, gold, keep = ce.contract_arrays(test.segments, S, test.status, cfg["rule"])
print("F1=%.9f" % ce.contract_macro(cs, gold, keep, [cfg["threshold"]])[0, 2])
"""


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "reports", "eval_v4", "doc_lora_v6b_test.json")), reason="v6b test scores not present")
def test_config_reproduces_the_reported_test_result_from_saved_scores():
    """Needs the 7-clause profile, so it runs in its own process (the test process uses the 6-clause default)."""
    env = {**os.environ, "LEXPROBE_PROFILE": "v4", "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-800:]
    f1 = float(out.stdout.strip().split("F1=")[1])
    report = json.load(open(os.path.join(ROOT, "reports", "eval_v4", "recovery_stage1_v6b.json"), encoding="utf-8"))
    assert f1 == pytest.approx(report["selected"]["test_contract_f1"], abs=1e-9)
    assert f1 == pytest.approx(0.867, abs=1e-3)
