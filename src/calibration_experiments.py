"""
Stage 1 of the recovery plan (alignment_plan D7): can the gap be closed WITHOUT retraining?
CPU only, on SAVED scores (reports/eval/doc_<system>_<split>.json). Nothing here loads a model.

Candidates (score transforms)
  ident   raw first-token margin                                   (what the current references use)
  shift   margin - per-clause prior log-odds shift (v4 only: it was trained at 1.5 negatives per positive)
  platt   per-clause logistic calibration a*margin + b            (fitted on validation)
  blend   per-clause logistic on [v4 margin, baseline margin]      (fitted on validation)
Contract rule: "max" (flag if any chunk is high) or "top2" (mean of the two highest chunks)
Threshold objective: "zero" (fixed 0, no tuning), "chunk" (max chunk macro-F1 on validation),
                     "contract" (max contract macro-F1 on validation)

Discipline (pre-registered in alignment_plan D7 before this was run)
  - Every choice is made on VALIDATION only. Fitted calibrators use 5-fold out-of-fold validation scores
    (folds = near-duplicate contract groups), so a variant is never judged on data it was fitted on.
  - ONE v4-based variant is selected by validation contract-F1. Test numbers of the other variants are
    printed for understanding but never used to choose.
  - Primary gate: selected candidate minus the approved zero-shot reference (baseline, raw margin, global
    threshold tuned on validation), contract-level macro-F1, paired group-bootstrap 95% CI lower bound > 0.
  - Stress test (reported, not the gate): same comparison against the BEST baseline variant.

Usage:
  python src/calibration_experiments.py
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

import document_eval as de
from prepare_data import CLAUSE_TYPES

K = len(CLAUSE_TYPES)
# Same grid as the references in document_level.json. A wider grid (up to +14) selects +12.0 for the
# zero-shot baseline and a weaker reference (test contract F1 0.686 vs 0.739); the pre-registered reference
# must not be changed after the fact, so the original grid is kept for every candidate.
GRID = np.arange(-3.0, 8.01, 0.5)
N_FOLDS = 5
STATUS_CODE = {"positive": 1, "negative": 0, "partial": -1}
OUT = "reports/eval/recovery_stage1"


# ── Data ───────────────────────────────────────────────────────────────────────
class Split:
    """One split's chunks as arrays. Chunks of a contract are contiguous (asserted)."""

    def __init__(self, name: str):
        self.name = name
        self.chunks = de.load_json(f"data/processed/chunks_{name}.json")
        self.ids = [c["chunk_id"] for c in self.chunks]
        self.titles = [c["contract_title"] for c in self.chunks]
        self.status = np.array([[STATUS_CODE[c["labels"][ct]["status"]] for ct in CLAUSE_TYPES] for c in self.chunks])
        self.segments = segments(self.titles)

    def scores(self, system: str) -> np.ndarray:
        m = de.load_json(f"reports/eval/doc_{system}_{self.name}.json")["margins"]
        return np.array([[m[i][ct] for ct in CLAUSE_TYPES] for i in self.ids], dtype=float)


def segments(titles: list[str]) -> list[tuple[str, int, int]]:
    """[(title, start, end)] for contiguous runs; a title may not reappear later."""
    out, seen, i = [], set(), 0
    while i < len(titles):
        j = i
        while j < len(titles) and titles[j] == titles[i]:
            j += 1
        assert titles[i] not in seen, f"chunks of {titles[i]} are not contiguous"
        seen.add(titles[i])
        out.append((titles[i], i, j))
        i = j
    return out


# ── Metrics on arrays ──────────────────────────────────────────────────────────
def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def chunk_macro(S: np.ndarray, st: np.ndarray, grid) -> np.ndarray:
    """(len(grid), 3) macro precision / recall / F1 over chunk-level pairs (partial pairs excluded)."""
    clauses = [k for k in range(S.shape[1]) if (st[:, k] == 1).any()]
    out = np.zeros((len(grid), 3))
    for gi, t in enumerate(grid):
        vals = []
        for k in clauses:
            keep = st[:, k] != -1
            y, pred = st[keep, k] == 1, S[keep, k] > t
            vals.append(_f1((y & pred).sum(), (~y & pred).sum(), (y & ~pred).sum()))
        out[gi] = np.mean(vals, axis=0)
    return out


def contract_arrays(segs, S: np.ndarray, st: np.ndarray, rule: str):
    """Per (contract, clause): score, gold label, keep-flag (contracts with only a partial match are skipped)."""
    C = len(segs)
    cs, gold, keep = np.zeros((C, S.shape[1])), np.zeros((C, S.shape[1]), bool), np.ones((C, S.shape[1]), bool)
    for i, (_, a, b) in enumerate(segs):
        for k in range(S.shape[1]):
            v = np.sort(S[a:b, k])[::-1]
            cs[i, k] = v[0] if (rule == "max" or len(v) < 2) else float(v[:2].mean())
            gold[i, k] = (st[a:b, k] == 1).any()
            keep[i, k] = gold[i, k] or not (st[a:b, k] == -1).any()
    return cs, gold, keep


def contract_macro(cs, gold, keep, grid) -> np.ndarray:
    clauses = [k for k in range(cs.shape[1]) if (gold[:, k] & keep[:, k]).any()]
    out = np.zeros((len(grid), 3))
    for gi, t in enumerate(grid):
        vals = []
        for k in clauses:
            m = keep[:, k]
            y, pred = gold[m, k], cs[m, k] > t
            vals.append(_f1((y & pred).sum(), (~y & pred).sum(), (y & ~pred).sum()))
        out[gi] = np.mean(vals, axis=0)
    return out


def best_threshold(f1_curve: np.ndarray, grid) -> float:
    """Argmax of macro-F1; ties go to the threshold closest to 0."""
    order = sorted(range(len(grid)), key=lambda i: (-round(float(f1_curve[i]), 6), abs(grid[i])))
    return float(grid[order[0]])


# ── Calibrators ────────────────────────────────────────────────────────────────
def fit_calibrators(X: np.ndarray, st: np.ndarray, min_pos: int = 3):
    """
    X: (n, K, F) features. Returns per clause (coef (F,), intercept) from a logistic regression on the
    non-partial pairs. Falls back to the first feature with unit slope when a clause has too few positives.
    """
    from sklearn.linear_model import LogisticRegression

    F = X.shape[2]
    params = []
    for k in range(X.shape[1]):
        keep = st[:, k] != -1
        y = (st[keep, k] == 1).astype(int)
        if y.sum() < min_pos or y.sum() == len(y):
            coef = np.zeros(F)
            coef[0] = 1.0
            params.append((coef, 0.0))
            continue
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X[keep, k, :], y)
        params.append((lr.coef_[0].copy(), float(lr.intercept_[0])))
    return params


def apply_calibrators(X: np.ndarray, params) -> np.ndarray:
    return np.stack([X[:, k, :] @ params[k][0] + params[k][1] for k in range(X.shape[1])], axis=1)


def group_folds(groups: list, n_folds: int = N_FOLDS, seed: int = 42) -> dict:
    """Assign each group to a fold (disjoint, near-balanced)."""
    uniq = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(uniq))
    return {uniq[g]: i % n_folds for i, g in enumerate(order)}


def oof_scores(X: np.ndarray, st: np.ndarray, chunk_groups: list, fitted: bool, seed: int = 42) -> np.ndarray:
    """Out-of-fold calibrated scores for the validation split (or the raw first feature if not fitted)."""
    if not fitted:
        return X[:, :, 0]
    fold_of = group_folds(chunk_groups, seed=seed)
    fold = np.array([fold_of[g] for g in chunk_groups])
    out = np.zeros(X.shape[:2])
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        params = fit_calibrators(X[tr], st[tr])
        out[te] = apply_calibrators(X[te], params)
    return out


def prior_shift() -> np.ndarray:
    """Per-clause log-odds shift between the training mix and the all-chunks mix of the training contracts."""
    train_q = de.load_json("data/processed/train.json")
    chunks = de.load_json("data/processed/chunks_train.json")
    shifts = []
    for ct in CLAUSE_TYPES:
        tp = sum(1 for q in train_q if q["clause_type"] == ct and q["answers"])
        tn = sum(1 for q in train_q if q["clause_type"] == ct and not q["answers"])
        dp = sum(1 for c in chunks if c["labels"][ct]["status"] == "positive")
        dn = sum(1 for c in chunks if c["labels"][ct]["status"] == "negative")
        shifts.append(math.log((tp / tn) / (dp / dn)))
    return np.array(shifts)


# ── Experiment ─────────────────────────────────────────────────────────────────
def records(split: Split, cs, gold, keep, t):
    recs = []
    for i, (title, _, _) in enumerate(split.segments):
        for k, ct in enumerate(CLAUSE_TYPES):
            if keep[i, k]:
                recs.append({"contract_title": title, "clause_type": ct, "label": int(gold[i, k]), "pred": int(cs[i, k] > t)})
    return recs


def chunk_records_arrays(split: Split, S: np.ndarray, t: float) -> list:
    recs = []
    for i, title in enumerate(split.titles):
        for k, ct in enumerate(CLAUSE_TYPES):
            s = split.status[i, k]
            if s != -1:
                recs.append({"contract_title": title, "clause_type": ct, "label": int(s == 1), "pred": int(S[i, k] > t)})
    return recs


def as_margins(split: Split, S: np.ndarray) -> dict:
    return {cid: {ct: float(S[i, k]) for k, ct in enumerate(CLAUSE_TYPES)} for i, cid in enumerate(split.ids)}


def secondary(test: Split, gm: dict, Xt: dict, S_sel, sel, ref, bootstrap: int, ap_bootstrap: int) -> dict:
    """Secondary metrics for the selected candidate vs the approved reference (reported, not gates)."""
    from collections import Counter

    from eval_metrics import per_clause_metrics

    cr = chunk_records_arrays(test, S_sel, sel["threshold"])
    cs, gold, keep = contract_arrays(test.segments, S_sel, test.status, sel["rule"])
    kr = records(test, cs, gold, keep, sel["threshold"])
    ref_S = Xt["baseline"]
    rcr = chunk_records_arrays(test, ref_S, ref["threshold"])
    pc, pk = per_clause_metrics(cr), per_clause_metrics(kr)
    fp_chunks = sum(1 for r in cr if r["pred"] and not r["label"])
    wrong = Counter(r["contract_title"] for r in kr if r["pred"] and not r["label"])
    n_contracts = len(test.segments)
    ex_ld = lambda m: float(np.mean([m[c]["f1"] for c in CLAUSE_TYPES if c in m and c != "Liquidated Damages"]))
    boot_chunk = de.paired_group_bootstrap(de.counts_by_group(rcr, gm), de.counts_by_group(cr, gm), bootstrap)
    ap = {}
    for level in ("chunk", "contract"):
        ap[level] = de.ap_bootstrap(de.build_ap_data(test.chunks, as_margins(test, ref_S), gm, level),
                                    de.build_ap_data(test.chunks, as_margins(test, S_sel), gm, level), ap_bootstrap)
    return {
        "chunk_macro": pc["macro"], "contract_macro": pk["macro"],
        "per_clause_chunk_f1": {c: pc[c]["f1"] for c in CLAUSE_TYPES if c in pc},
        "per_clause_contract_f1": {c: pk[c]["f1"] for c in CLAUSE_TYPES if c in pk},
        "falsely_flagged_chunks_per_contract": fp_chunks / n_contracts,
        "wrong_clause_flags_per_contract": sum(wrong.values()) / n_contracts,
        "chunk_f1_excluding_liquidated_damages": ex_ld(pc), "contract_f1_excluding_liquidated_damages": ex_ld(pk),
        "chunk_f1_diff_vs_reference": {"diff": boot_chunk["diff_f1"], "ci": boot_chunk["diff_f1_ci"]},
        "average_precision_vs_reference_baseline": {
            lvl: {"reference": v["a_ap"], "candidate": v["b_ap"], "diff": v["diff_ap"], "ci": v["diff_ap_ci"],
                  "per_clause_candidate": v["per_clause_b"], "per_clause_reference": v["per_clause_a"]}
            for lvl, v in ap.items()},
    }


def selection_stability(val: Split, Xv: dict, val_groups: list, shift: np.ndarray, seeds=(1, 2, 3, 4, 5)) -> dict:
    """Would the same v4-based candidate be selected under other fold assignments? (validation only)"""
    order = ["lora_v4:ident", "lora_v4:shift", "lora_v4:platt", "blend:platt"]
    out = {}
    for seed in seeds:
        fv = np.stack([Xv["lora_v4"]], axis=2)
        bv = np.stack([Xv["lora_v4"], Xv["baseline"]], axis=2)
        scores = {"lora_v4:ident": Xv["lora_v4"], "lora_v4:shift": Xv["lora_v4"] - shift,
                  "lora_v4:platt": oof_scores(fv, val.status, val_groups, True, seed),
                  "blend:platt": oof_scores(bv, val.status, val_groups, True, seed)}
        best = {}
        for name, Sv in scores.items():
            chunk_curve = chunk_macro(Sv, val.status, GRID)[:, 2]
            vals = []
            for rule in ("max", "top2"):
                cv = contract_arrays(val.segments, Sv, val.status, rule)
                contract_curve = contract_macro(*cv, GRID)[:, 2]
                for objective in ("zero", "chunk", "contract"):
                    t = 0.0 if objective == "zero" else best_threshold(chunk_curve if objective == "chunk" else contract_curve, GRID)
                    vals.append(float(contract_macro(*cv, [t])[0, 2]))
            best[name] = max(vals)
        out[seed] = {"best_val_contract_f1": best, "selected": max(order, key=lambda n: (round(best[n], 6), -order.index(n)))}
    return out


def run(bootstrap: int = 2000, ap_bootstrap: int = 500) -> dict:
    from compare_results import group_map

    val, test = Split("val"), Split("test")
    gm = group_map(set(val.titles) | set(test.titles))
    val_groups = [gm[t] for t in val.titles]

    Xv = {s: val.scores(s) for s in ("baseline", "lora_v4")}
    Xt = {s: test.scores(s) for s in ("baseline", "lora_v4")}
    shift = prior_shift()

    # candidate -> (val OOF scores, test scores)
    cands = {}
    for s in ("baseline", "lora_v4"):
        cands[f"{s}:ident"] = (Xv[s], Xt[s])
        fv = np.stack([Xv[s]], axis=2)
        ft = np.stack([Xt[s]], axis=2)
        cands[f"{s}:platt"] = (oof_scores(fv, val.status, val_groups, True),
                               apply_calibrators(ft, fit_calibrators(fv, val.status)))
    cands["lora_v4:shift"] = (Xv["lora_v4"] - shift, Xt["lora_v4"] - shift)
    bv = np.stack([Xv["lora_v4"], Xv["baseline"]], axis=2)
    bt = np.stack([Xt["lora_v4"], Xt["baseline"]], axis=2)
    cands["blend:platt"] = (oof_scores(bv, val.status, val_groups, True), apply_calibrators(bt, fit_calibrators(bv, val.status)))

    rows = []
    for name, (Sv, St) in cands.items():
        chunk_curve = chunk_macro(Sv, val.status, GRID)[:, 2]
        for rule in ("max", "top2"):
            cv = contract_arrays(val.segments, Sv, val.status, rule)
            ct_ = contract_arrays(test.segments, St, test.status, rule)
            contract_curve = contract_macro(*cv, GRID)[:, 2]
            for objective in ("zero", "chunk", "contract"):
                t = 0.0 if objective == "zero" else best_threshold(chunk_curve if objective == "chunk" else contract_curve, GRID)
                rows.append({
                    "candidate": name, "rule": rule, "objective": objective, "threshold": t,
                    "val_chunk_f1": float(chunk_macro(Sv, val.status, [t])[0, 2]),
                    "val_contract_f1": float(contract_macro(*cv, [t])[0, 2]),
                    "test_chunk_f1": float(chunk_macro(St, test.status, [t])[0, 2]),
                    "test_contract_f1": float(contract_macro(*ct_, [t])[0, 2]),
                    "_test_arrays": ct_,
                })

    def row(candidate, rule, objective):
        return next(r for r in rows if (r["candidate"], r["rule"], r["objective"]) == (candidate, rule, objective))

    # sanity: the identity rows must reproduce the numbers already reported (document_level.json)
    ref_json = de.load_json("reports/eval/document_level.json")["operating_points"]["tuned"]
    sanity = {}
    for s, key in (("baseline", "baseline"), ("lora_v4", "lora_v4")):
        r = row(f"{s}:ident", "max", "chunk")
        exp_t = ref_json["thresholds"][key]
        exp_f1 = ref_json["summary"][key]["contract"]["macro"]["f1"]
        sanity[s] = {"threshold": r["threshold"], "expected_threshold": exp_t, "test_contract_f1": r["test_contract_f1"],
                     "expected_contract_f1": exp_f1,
                     "ok": abs(r["threshold"] - exp_t) < 1e-9 and abs(r["test_contract_f1"] - exp_f1) < 1e-3}

    # selection among v4-based candidates by validation contract-F1 (ties: simpler first)
    order = ["lora_v4:ident", "lora_v4:shift", "lora_v4:platt", "blend:platt"]
    v4_rows = [r for r in rows if r["candidate"] in order]
    selected = max(v4_rows, key=lambda r: (round(r["val_contract_f1"], 6), -order.index(r["candidate"])))
    base_rows = [r for r in rows if r["candidate"].startswith("baseline")]
    best_base = max(base_rows, key=lambda r: round(r["val_contract_f1"], 6))
    ref = row("baseline:ident", "max", "chunk")

    def counts(r):
        cs, gold, keep = r["_test_arrays"]
        return de.counts_by_group(records(test, cs, gold, keep, r["threshold"]), gm)

    gates = {}
    for label, cand, reference in (("current v4 (sanity)", row("lora_v4:ident", "max", "chunk"), ref),
                                   ("PRIMARY: selected candidate vs approved reference", selected, ref),
                                   ("STRESS: selected candidate vs best baseline variant", selected, best_base)):
        b = de.paired_group_bootstrap(counts(reference), counts(cand), bootstrap)
        gates[label] = {"candidate": f"{cand['candidate']} / {cand['rule']} / {cand['objective']} @ {cand['threshold']:+.1f}",
                        "reference": f"{reference['candidate']} / {reference['rule']} / {reference['objective']} @ {reference['threshold']:+.1f}",
                        "reference_test_contract_f1": reference["test_contract_f1"], "candidate_test_contract_f1": cand["test_contract_f1"],
                        "diff": b["diff_f1"], "ci": b["diff_f1_ci"], "pass": bool(b["diff_f1_ci"][0] > 0)}
    sec = secondary(test, gm, Xt, cands[selected["candidate"]][1], selected, ref, bootstrap, ap_bootstrap)
    stab = selection_stability(val, Xv, val_groups, shift)
    fitted_params = fit_calibrators(bv, val.status)
    fitted = {ct: {"w_v4": float(fitted_params[k][0][0]), "w_baseline": float(fitted_params[k][0][1]),
                   "intercept": float(fitted_params[k][1])} for k, ct in enumerate(CLAUSE_TYPES)}
    for r in rows:
        r.pop("_test_arrays")
    for r in (selected, best_base, ref):
        r.pop("_test_arrays", None)
    return {"secondary": sec, "selection_stability": stab, "blend_parameters_fitted_on_all_validation": fitted, "rows": rows, "selected": selected, "best_baseline": best_base, "approved_reference": ref,
            "sanity": sanity, "gates": gates, "prior_shift": shift.tolist(), "n_bootstrap": bootstrap}


def write_report(res: dict):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    L = ["# Recovery Stage 1: calibration experiments on saved scores\n",
         "Selection on validation only (out-of-fold for fitted calibrators). Test columns are shown for understanding; only the PRIMARY row is a gate.\n",
         "## Sanity: identity rows reproduce the reported numbers\n"]
    for s, v in res["sanity"].items():
        L.append(f"- {s}: threshold {v['threshold']:+.1f} (expected {v['expected_threshold']:+.1f}), test contract F1 {v['test_contract_f1']:.3f} "
                 f"(expected {v['expected_contract_f1']:.3f}) -> {'OK' if v['ok'] else 'MISMATCH'}")
    L += ["", "## Gates\n"]
    for label, g in res["gates"].items():
        L.append(f"- **{label}**: {g['candidate']}  vs  {g['reference']}: test contract F1 {g['reference_test_contract_f1']:.3f} -> "
                 f"{g['candidate_test_contract_f1']:.3f}, diff {g['diff']:+.3f} (95% CI {g['ci'][0]:+.3f} to {g['ci'][1]:+.3f}) -> **{'PASS' if g['pass'] else 'FAIL'}**")
    sec = res["secondary"]
    L += ["", "## Secondary metrics of the selected candidate (reported, not gates)\n",
          f"- Chunk macro P/R/F1: {sec['chunk_macro']['precision']:.3f} / {sec['chunk_macro']['recall']:.3f} / {sec['chunk_macro']['f1']:.3f}; "
          f"difference vs approved reference {sec['chunk_f1_diff_vs_reference']['diff']:+.3f} "
          f"(95% CI {sec['chunk_f1_diff_vs_reference']['ci'][0]:+.3f} to {sec['chunk_f1_diff_vs_reference']['ci'][1]:+.3f})",
          f"- Contract macro P/R/F1: {sec['contract_macro']['precision']:.3f} / {sec['contract_macro']['recall']:.3f} / {sec['contract_macro']['f1']:.3f}",
          f"- Falsely flagged chunks per contract: {sec['falsely_flagged_chunks_per_contract']:.2f}; "
          f"wrong clause flags per contract (of 6): {sec['wrong_clause_flags_per_contract']:.2f}",
          f"- Excluding Liquidated Damages: chunk F1 {sec['chunk_f1_excluding_liquidated_damages']:.3f}, "
          f"contract F1 {sec['contract_f1_excluding_liquidated_damages']:.3f}",
          "- Average precision (ranking, threshold-free) vs raw zero-shot margins: " + "; ".join(
              f"{lvl} {v['reference']:.3f} -> {v['candidate']:.3f} (diff {v['diff']:+.3f}, CI {v['ci'][0]:+.3f} to {v['ci'][1]:+.3f})"
              for lvl, v in sec["average_precision_vs_reference_baseline"].items()),
          "", "| Clause | chunk F1 | contract F1 | blend weight v4 | blend weight baseline | intercept |", "|---|---|---|---|---|---|"]
    for ct in CLAUSE_TYPES:
        fp = res["blend_parameters_fitted_on_all_validation"][ct]
        L.append(f"| {ct} | {sec['per_clause_chunk_f1'].get(ct, float('nan')):.3f} | {sec['per_clause_contract_f1'].get(ct, float('nan')):.3f} | "
                 f"{fp['w_v4']:+.2f} | {fp['w_baseline']:+.2f} | {fp['intercept']:+.2f} |")
    L += ["", "## Is the selection stable? (validation only, 5 other fold assignments)\n",
          "| fold seed | selected | best val contract F1: v4:ident / v4:shift / v4:platt / blend:platt |", "|---|---|---|"]
    for seed, v in res["selection_stability"].items():
        b = v["best_val_contract_f1"]
        L.append(f"| {seed} | {v['selected']} | {b['lora_v4:ident']:.3f} / {b['lora_v4:shift']:.3f} / {b['lora_v4:platt']:.3f} / {b['blend:platt']:.3f} |")
    L += ["", "## All variants (sorted by validation contract-F1)\n",
          "| candidate | rule | threshold objective | t | val chunk F1 | val contract F1 | test chunk F1 | test contract F1 |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(res["rows"], key=lambda r: -r["val_contract_f1"]):
        L.append(f"| {r['candidate']} | {r['rule']} | {r['objective']} | {r['threshold']:+.1f} | {r['val_chunk_f1']:.3f} | "
                 f"{r['val_contract_f1']:.3f} | {r['test_chunk_f1']:.3f} | {r['test_contract_f1']:.3f} |")
    with open(OUT + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:46]))


def export_blend(res: dict, path: str):
    """Write the selected blend (weights fitted on ALL validation scores) as a config the API can load."""
    sel = res["selected"]
    assert sel["candidate"] == "blend:platt", f"selected candidate is {sel['candidate']}, not the blend"
    val_meta = de.load_json("reports/eval/doc_lora_v4_val.json")
    primary = next(g for k, g in res["gates"].items() if k.startswith("PRIMARY"))
    stress = next(g for k, g in res["gates"].items() if k.startswith("STRESS"))
    cfg = {
        "name": "blend_v1",
        "description": "Per-clause logistic blend of the v4 (adapter on) and zero-shot (adapter off) first-token margins. "
                       "A clause is flagged when ANY chunk's blended score exceeds the threshold.",
        "fitted_on": "validation split (all contracts), dataset " + val_meta["dataset_version"],
        "adapter": val_meta["adapter"],
        "rule": sel["rule"],
        "threshold": sel["threshold"],
        "clause_params": res["blend_parameters_fitted_on_all_validation"],
        "selection": {"objective": sel["objective"], "val_contract_f1_out_of_fold": sel["val_contract_f1"],
                      "val_chunk_f1_out_of_fold": sel["val_chunk_f1"]},
        "test_result_at_selection": {"contract_f1": sel["test_contract_f1"], "chunk_f1": sel["test_chunk_f1"],
                                     "primary_gate_diff": primary["diff"], "primary_gate_ci": primary["ci"],
                                     "stress_gate_diff": stress["diff"], "stress_gate_ci": stress["ci"]},
        "source_report": "reports/eval/recovery_stage1.md",
        "caveats": ["two passes per chunk (adapter on and off): about 1.8x compute",
                    "Auto-Renewal relies mostly on the zero-shot score; Liquidated Damages remains weak",
                    "weights fitted on 77 validation contracts; re-fit if the adapter or dataset changes"],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"blend config written to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--export-blend", metavar="PATH", help="also write the selected blend as a JSON config (e.g. configs/blend_v1.json)")
    args = parser.parse_args()
    res = run(args.bootstrap)
    write_report(res)
    if args.export_blend:
        export_blend(res, args.export_blend)


if __name__ == "__main__":
    main()
