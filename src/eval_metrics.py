"""
Pure evaluation helpers (no torch): classification metrics, evidence grounding,
span overlap and paired bootstrap (alignment_plan D2, D3, D4).
"""

import random
import re
from collections import defaultdict
from difflib import SequenceMatcher


# ── Classification ─────────────────────────────────────────────────────────────
def prf(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f1}


def per_clause_metrics(records: list[dict], pred_key: str = "pred") -> dict:
    """{clause: prf} plus "macro" (mean over clauses that have positives)."""
    by_clause = defaultdict(list)
    for r in records:
        by_clause[r["clause_type"]].append(r)
    out = {}
    for ct, rs in sorted(by_clause.items()):
        if not any(r["label"] for r in rs):
            continue
        out[ct] = prf([r["label"] for r in rs], [r[pred_key] for r in rs])
        out[ct]["support_pos"] = sum(r["label"] for r in rs)
    keys = [k for k in out]
    # Undefined (NaN) when no clause has positives, e.g. an unlucky bootstrap draw
    out["macro"] = {m: (sum(out[k][m] for k in keys) / len(keys)) if keys else float("nan")
                    for m in ("precision", "recall", "f1")}
    return out


# ── Evidence grounding (D2) ────────────────────────────────────────────────────
def norm(text: str) -> str:
    return " ".join(text.split()).lower()


def clean_quote(generated: str) -> str:
    """Strip the 'Found:' prefix, surrounding quotes and trailing ellipses."""
    q = generated.strip()
    q = re.sub(r"^found:\s*", "", q, flags=re.I)
    return q.strip().strip('"“”').rstrip(".… ").strip()


def grounding(quote: str, source: str) -> dict:
    """
    exact    — normalised quote is a substring of the normalised source
    coverage — fraction of the quote covered by its longest common block with the source
    grounded — exact, or coverage >= 0.9 (tolerates small tokenisation / punctuation drift)
    """
    q, s = norm(quote), norm(source)
    if not q:
        return {"exact": False, "coverage": 0.0, "grounded": False}
    exact = q in s
    if exact:
        cov = 1.0
    else:
        m = SequenceMatcher(None, q, s, autojunk=False).find_longest_match(0, len(q), 0, len(s))
        cov = m.size / len(q)
    return {"exact": exact, "coverage": cov, "grounded": exact or cov >= 0.9}


def token_f1(pred: str, gold: str) -> float:
    """CUAD/SQuAD-style token overlap F1."""
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return 0.0
    common = defaultdict(int)
    for t in g:
        common[t] += 1
    overlap = 0
    for t in p:
        if common[t] > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


# ── Paired bootstrap (D3) ──────────────────────────────────────────────────────
def paired_bootstrap(records_a: list[dict], records_b: list[dict], group_of: dict, n: int = 2000,
                     seed: int = 42) -> dict:
    """
    Resample GROUPS (near-duplicate contract groups) with replacement and recompute macro-F1
    for both systems on the same draw. Records must be aligned (same query order).
    Returns point estimates, 95% CIs for each system and for the difference (b - a),
    and the one-sided p-value P(diff <= 0).
    """
    assert [r["query_id"] for r in records_a] == [r["query_id"] for r in records_b]
    groups = defaultdict(list)
    for i, r in enumerate(records_a):
        groups[group_of[r["contract_title"]]].append(i)
    keys = sorted(groups)
    rng = random.Random(seed)

    def macro(recs, idx):
        return per_clause_metrics([recs[i] for i in idx])["macro"]["f1"]

    fa, fb, diffs = [], [], []
    for _ in range(n):
        idx = [i for k in rng.choices(keys, k=len(keys)) for i in groups[k]]
        a, b = macro(records_a, idx), macro(records_b, idx)
        if a != a or b != b:  # NaN: draw had no positives
            continue
        fa.append(a); fb.append(b); diffs.append(b - a)

    def ci(v):
        v = sorted(v)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]

    all_idx = list(range(len(records_a)))
    return {
        "n_bootstrap": n,
        "n_groups": len(keys),
        "a_macro_f1": macro(records_a, all_idx), "a_ci": ci(fa),
        "b_macro_f1": macro(records_b, all_idx), "b_ci": ci(fb),
        "diff": macro(records_b, all_idx) - macro(records_a, all_idx), "diff_ci": ci(diffs),
        "n_valid_draws": len(diffs),
        "p_value_one_sided": sum(1 for d in diffs if d <= 0) / len(diffs),
    }


def mcnemar(records_a: list[dict], records_b: list[dict]) -> dict:
    """Exact McNemar test on per-query correctness (b better than a?)."""
    from math import comb
    b_only = sum(1 for a, b in zip(records_a, records_b) if a["pred"] != a["label"] and b["pred"] == b["label"])
    a_only = sum(1 for a, b in zip(records_a, records_b) if a["pred"] == a["label"] and b["pred"] != b["label"])
    n = a_only + b_only
    k = min(a_only, b_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only_b_correct": b_only, "only_a_correct": a_only, "p_value_two_sided": p}
