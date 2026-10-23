"""
Compare saved chunk-level scores of several models by ranking quality, with ranges (CPU only).

Why this and not a single F1: on 76 test contracts (and ~12 Liquidated Damages positives) gaps of 0.01-0.05 F1 are
noise. Average precision (AP) of the raw margins needs no threshold and no calibration, and resampling whole contracts
gives a 95% range for every model and for each difference against a reference model.

Usage (profile v4 = 7 clauses; systems are label=path pairs, first one is the reference):
  LEXPROBE_PROFILE=v4 python src/compare_models.py --split test --out reports/eval_v4/model_comparison \
      baseline=reports/eval_v4/doc_baseline_test.json v6=reports/eval_v4/doc_lora_v6_test.json
Clauses a model has no margins for are skipped; the macro uses the clauses common to all systems.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from prepare_data import CLAUSE_TYPES, DATA_DIR


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """AP with ties broken pessimistically (negatives first), so saturated/tied scores cannot look better than they are."""
    if y.sum() == 0:
        return float("nan")
    order = np.lexsort((y, -s))  # sort by score desc; within equal scores negatives (y=0) first
    ys = y[order]
    tp = np.cumsum(ys)
    return float((tp / np.arange(1, len(ys) + 1))[ys == 1].mean())


def load_margins(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["margins"]


def build(chunks: list[dict], margins: dict[str, dict], clauses: list[str]):
    """Per clause: arrays (y, scores per system, contract index) over chunks that are not 'partial'."""
    titles = sorted({c["contract_title"] for c in chunks})
    t_index = {t: i for i, t in enumerate(titles)}
    out = {}
    for ct in clauses:
        keep = [c for c in chunks if c["labels"][ct]["status"] != "partial"]
        y = np.array([1 if c["labels"][ct]["status"] == "positive" else 0 for c in keep])
        s = {name: np.array([m[c["chunk_id"]][ct] for c in keep]) for name, m in margins.items()}
        g = np.array([t_index[c["contract_title"]] for c in keep])
        out[ct] = (y, s, g)
    return out, len(titles)


def macro_ap(data, systems, clauses, contract_weights=None):
    """macro AP over clauses; `contract_weights` (resample counts per contract) turns the data into a bootstrap draw."""
    res = {name: [] for name in systems}
    for ct in clauses:
        y, s, g = data[ct]
        if contract_weights is not None:
            reps = contract_weights[g]
            idx = np.repeat(np.arange(len(y)), reps)
        else:
            idx = np.arange(len(y))
        for name in systems:
            res[name].append(average_precision(y[idx], s[name][idx]))
    return {n: float(np.nanmean(v)) for n, v in res.items()}, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("systems", nargs="+", help="label=path_to_doc_scores.json ; the first is the reference")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = dict(s.split("=", 1) for s in args.systems)
    names = list(paths)
    margins = {n: load_margins(p) for n, p in paths.items()}
    chunks = json.load(open(os.path.join(DATA_DIR, f"chunks_{args.split}.json"), encoding="utf-8"))
    first = chunks[0]["chunk_id"]
    clauses = [ct for ct in CLAUSE_TYPES if all(ct in m[first] for m in margins.values())]
    data, n_contracts = build(chunks, margins, clauses)

    point, per_clause = macro_ap(data, names, clauses)
    rng = np.random.default_rng(0)
    boots = defaultdict(list)
    for _ in range(args.draws):
        w = np.bincount(rng.integers(0, n_contracts, n_contracts), minlength=n_contracts)
        m, _ = macro_ap(data, names, clauses, w)
        for n in names:
            boots[n].append(m[n])
    ref = names[0]
    lines = [f"# Model comparison on the {args.split} split ({n_contracts} contracts, {len(chunks)} chunks)", "",
             f"Raw-margin average precision (threshold-free), clauses: {', '.join(clauses)}. 95% ranges from {args.draws} contract resamples. Reference: `{ref}`.", "",
             "| model | macro AP | 95% range | diff vs reference | 95% range of diff | verdict |", "|---|---|---|---|---|---|"]
    summary = {}
    for n in names:
        lo, hi = np.percentile(boots[n], [2.5, 97.5])
        d = np.array(boots[n]) - np.array(boots[ref])
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        verdict = "-" if n == ref else ("better" if dlo > 0 else "worse" if dhi < 0 else "same within noise")
        lines.append(f"| {n} | {point[n]:.3f} | {lo:.3f} to {hi:.3f} | {point[n] - point[ref]:+.3f} | {dlo:+.3f} to {dhi:+.3f} | {verdict} |")
        summary[n] = {"macro_ap": point[n], "range": [lo, hi], "diff_vs_ref": point[n] - point[ref], "diff_range": [dlo, dhi], "verdict": verdict}
    lines += ["", "Per clause (point estimates):", "", "| clause | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for i, ct in enumerate(clauses):
        lines.append(f"| {ct} (positives {int(data[ct][0].sum())}) | " + " | ".join(f"{per_clause[n][i]:.3f}" for n in names) + " |")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "clauses": clauses, "systems": summary,
                   "per_clause": {n: dict(zip(clauses, per_clause[n])) for n in names}}, f, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
