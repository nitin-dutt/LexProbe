"""
The blend (alignment_plan D7, recovery Stage 1): combine the v4 and zero-shot first-token margins
with per-clause logistic weights fitted on validation. Pure Python, no torch.

  score(clause) = w_v4[clause] * margin_v4 + w_baseline[clause] * margin_zero_shot + intercept[clause]

The weights, threshold and contract rule live in configs/blend_v1.json (written by
`python src/calibration_experiments.py --export-blend configs/blend_v1.json`).
"""

import json
import math

from prepare_data import CLAUSE_TYPES

REQUIRED_KEYS = ("name", "rule", "threshold", "clause_params", "adapter")


def load_blend(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    validate_blend(cfg)
    return cfg


def validate_blend(cfg: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"blend config is missing keys: {missing}")
    if cfg["rule"] != "max":
        raise ValueError(f"unsupported contract rule {cfg['rule']!r} (only 'max' is implemented)")
    if not math.isfinite(cfg["threshold"]):
        raise ValueError("blend threshold must be finite")
    for ct in CLAUSE_TYPES:
        p = cfg["clause_params"].get(ct)
        if p is None:
            raise ValueError(f"blend config has no parameters for clause {ct!r}")
        for k in ("w_v4", "w_baseline", "intercept"):
            if not math.isfinite(p[k]):
                raise ValueError(f"non-finite {k} for {ct!r}")


def blend_scores(margins_v4: dict, margins_baseline: dict, cfg: dict) -> dict:
    """{clause: blended score} from the two margin dicts {clause: margin}."""
    out = {}
    for ct in CLAUSE_TYPES:
        p = cfg["clause_params"][ct]
        out[ct] = p["w_v4"] * margins_v4[ct] + p["w_baseline"] * margins_baseline[ct] + p["intercept"]
    return out
