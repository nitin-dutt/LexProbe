"""
Single-pass scoring config for the deployed v6b model (alignment_plan, 2026-10-20 / 2026-10-23).

v6b alone, no zero-shot second pass: the score of a (chunk, clause) pair is the first-token margin
logit(Found) - logit(None) minus a per-clause constant (the log-odds gap between the training mix and the real mix,
computed from training data only), and a clause is flagged when ANY chunk's score exceeds `threshold`
(chosen on validation, never on test). One forward pass per chunk: about half the compute of the v4+zero-shot blend.

The config is exported from the validated report (reports/eval_v4/recovery_stage1_v6b.json), not recomputed:
    LEXPROBE_PROFILE=v4 python src/single_model.py --report reports/eval_v4/recovery_stage1_v6b.json \
        --adapter models/llama-3.1-8b-cuad-lora-v6b/final_adapter --out configs/model_v6b.json
"""

import argparse
import json
import os

REQUIRED = ("name", "profile", "adapter", "rule", "threshold", "clause_shift")


def load_single(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    validate_single(cfg)
    return cfg


def validate_single(cfg: dict) -> None:
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"single-model config is missing {missing}")
    if cfg["rule"] != "max":
        raise ValueError(f"unsupported rule {cfg['rule']!r}: only 'max' (flag if ANY chunk exceeds the threshold) is implemented")
    if not isinstance(cfg["clause_shift"], dict) or not cfg["clause_shift"]:
        raise ValueError("clause_shift must be a non-empty {clause: shift} mapping")
    if any(not isinstance(v, (int, float)) for v in cfg["clause_shift"].values()):
        raise ValueError("clause_shift values must be numbers")


def single_scores(margins: dict, cfg: dict) -> dict:
    """{clause: margin - shift} for the clauses of the config (the margins dict must contain every one of them)."""
    missing = [c for c in cfg["clause_shift"] if c not in margins]
    if missing:
        raise KeyError(f"margins are missing clauses: {missing}")
    return {c: margins[c] - cfg["clause_shift"][c] for c in cfg["clause_shift"]}


def export_from_report(report_path: str, adapter: str, out_path: str, clauses: list[str]) -> dict:
    """Write the config for the candidate that validation selected. Only the single-adapter candidates are supported."""
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    sel = rep["selected"]
    if sel["candidate"] != "lora_v4:shift":
        raise ValueError(f"selected candidate is {sel['candidate']!r}; this exporter handles the prior-shift candidate only")
    shift = rep["prior_shift"]
    if len(shift) != len(clauses):
        raise ValueError(f"report has {len(shift)} shifts but {len(clauses)} clauses were given (run with LEXPROBE_PROFILE=v4)")
    primary = next(g for k, g in rep["gates"].items() if k.startswith("PRIMARY"))
    stress = next(g for k, g in rep["gates"].items() if k.startswith("STRESS"))
    cfg = {
        "name": "v6b",
        "description": "v6b adapter alone (one pass per chunk). Score = first-token margin minus a per-clause prior shift; "
                       "a clause is flagged when ANY chunk's score exceeds the threshold.",
        "profile": "v4",
        "adapter": adapter,
        "rule": sel["rule"],
        "threshold": sel["threshold"],
        "clause_shift": {c: float(s) for c, s in zip(clauses, shift)},
        "fitted_on": "shift: training mix vs all training chunks (no validation or test data); threshold: validation only",
        "selection": {"candidate": sel["candidate"], "objective": sel["objective"],
                      "val_contract_f1_out_of_fold": sel["val_contract_f1"], "val_chunk_f1_out_of_fold": sel["val_chunk_f1"],
                      "stable_across_fold_seeds": sorted({v["selected"] for v in rep["selection_stability"].values()})},
        "test_result_at_selection": {"contract_f1": sel["test_contract_f1"], "chunk_f1": sel["test_chunk_f1"],
                                     "primary_gate_diff": primary["diff"], "primary_gate_ci": primary["ci"],
                                     "stress_gate_diff": stress["diff"], "stress_gate_ci": stress["ci"]},
        "source_report": report_path.replace("\\", "/"),
        "caveats": ["Liquidated Damages is weak (contract F1 0.57 on 12 test positives)",
                    "test split is 76 contracts: differences of a few points between model versions are within noise",
                    "threshold and shifts were fitted for dataset profile v4 (7 clauses); re-fit if the adapter or data change"],
    }
    validate_single(cfg)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from prepare_data import CLAUSE_TYPES  # needs LEXPROBE_PROFILE=v4 so the clause order matches the report
    cfg = export_from_report(args.report, args.adapter, args.out, list(CLAUSE_TYPES))
    print(f"wrote {args.out}: threshold {cfg['threshold']}, shifts {cfg['clause_shift']}")


if __name__ == "__main__":
    main()
