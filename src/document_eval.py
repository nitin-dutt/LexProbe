"""
Full-document evaluation analysis (alignment_plan D1). CPU only.

Inputs
  data/processed/chunks_<split>.json     every chunk of every contract with a label per clause
  reports/eval/doc_<system>_<split>.json {chunk_id: {clause: margin}} from src/evaluate_documents.py

Two realistic views (the sampled test set has 1 positive : 2 negatives; a deployed system sees
every chunk, ~98.5% of (chunk, clause) pairs are negative):
  chunk level     every (chunk, clause) pair; "partial" pairs (clause cut by a chunk edge) are excluded
  contract level  per (contract, clause): flagged if ANY chunk is flagged; gold if ANY chunk is positive
                  + localisation: is the highest-scoring chunk (the one the API quotes) really the clause?

Decision thresholds are chosen on the VALIDATION split only (separately for each system), then
applied to test. Also reported: average precision, a threshold-free ranking metric, so the
comparison does not depend on how conservative each model happens to be.
Confidence intervals: paired bootstrap over near-duplicate contract groups (per-group confusion
counts for F1; resampled score arrays for average precision).

Usage:
  python src/document_eval.py --baseline reports/eval/doc_baseline_test.json \
      --system reports/eval/doc_lora_v4_test.json \
      --val-baseline reports/eval/doc_baseline_val.json --val reports/eval/doc_lora_v4_val.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from eval_metrics import per_clause_metrics
from prepare_data import CLAUSE_TYPES

K = len(CLAUSE_TYPES)
CT_INDEX = {c: i for i, c in enumerate(CLAUSE_TYPES)}


# ── Records ────────────────────────────────────────────────────────────────────
def chunk_records(chunks: list[dict], margins: dict, thr: float = 0.0) -> list[dict]:
    """One record per (chunk, clause); partial pairs are excluded (neither positive nor negative)."""
    recs = []
    for ch in chunks:
        m = margins[ch["chunk_id"]]
        for ct in CLAUSE_TYPES:
            status = ch["labels"][ct]["status"]
            if status == "partial":
                continue
            recs.append({"query_id": f"{ch['chunk_id']}::{ct}", "contract_title": ch["contract_title"],
                         "clause_type": ct, "label": int(status == "positive"),
                         "pred": int(m[ct] > thr), "margin": m[ct]})
    return recs


def contract_records(chunks: list[dict], margins: dict, thr: float = 0.0) -> list[dict]:
    """One record per (contract, clause). Contracts with only a partial match and no positive are skipped."""
    by_contract = defaultdict(list)
    for ch in chunks:
        by_contract[ch["contract_title"]].append(ch)

    recs = []
    for title, chs in by_contract.items():
        for ct in CLAUSE_TYPES:
            statuses = [ch["labels"][ct]["status"] for ch in chs]
            gold = "positive" in statuses
            if not gold and "partial" in statuses:
                continue
            flagged = [ch for ch in chs if margins[ch["chunk_id"]][ct] > thr]
            best = max(chs, key=lambda ch: margins[ch["chunk_id"]][ct])
            recs.append({
                "query_id": f"{title}::{ct}", "contract_title": title, "clause_type": ct,
                "label": int(gold), "pred": int(bool(flagged)), "n_flagged": len(flagged),
                # the chunk the API would quote is the highest-scoring one; partial = clause cut by the chunk edge
                "top_chunk_on_clause": best["labels"][ct]["status"] in ("positive", "partial"),
            })
    return recs


# ── Summaries ──────────────────────────────────────────────────────────────────
def summarize(chunk_recs: list[dict], contract_recs: list[dict]) -> dict:
    n_contracts = len({r["contract_title"] for r in contract_recs})
    negatives = sum(1 for r in chunk_recs if not r["label"])
    fp_chunks = sum(1 for r in chunk_recs if r["pred"] and not r["label"])
    flagged_tp = [r for r in contract_recs if r["label"] and r["pred"]]
    false_flags = defaultdict(int)
    for r in contract_recs:
        false_flags[r["contract_title"]] += int(r["pred"] and not r["label"])
    return {
        "chunk": per_clause_metrics(chunk_recs),
        "contract": per_clause_metrics(contract_recs),
        "chunk_negatives": negatives,
        "chunk_false_positive_rate": fp_chunks / negatives if negatives else 0.0,
        "false_flagged_chunks_per_contract": fp_chunks / n_contracts,
        "wrong_clause_flags_per_contract": sum(false_flags.values()) / n_contracts,
        "contracts_with_no_wrong_flag": sum(1 for v in false_flags.values() if v == 0) / n_contracts,
        "top_chunk_on_clause_rate": (sum(r["top_chunk_on_clause"] for r in flagged_tp) / len(flagged_tp)) if flagged_tp else None,
        "n_contracts": n_contracts,
    }


def sweep(chunks: list[dict], margins: dict, thresholds) -> list[dict]:
    rows = []
    for t in thresholds:
        cr, co = chunk_records(chunks, margins, t), contract_records(chunks, margins, t)
        c, k = per_clause_metrics(cr)["macro"], per_clause_metrics(co)["macro"]
        rows.append({"threshold": float(t), "chunk_precision": c["precision"], "chunk_recall": c["recall"],
                     "chunk_f1": c["f1"], "contract_precision": k["precision"], "contract_recall": k["recall"],
                     "contract_f1": k["f1"]})
    return rows


def select_threshold(rows: list[dict], key: str = "chunk_f1") -> dict:
    """Best row by `key`; ties go to the threshold closest to 0 (the model's natural boundary)."""
    return max(rows, key=lambda r: (round(r[key], 6), -abs(r["threshold"])))


# ── Bootstrap over groups, from per-group confusion counts ─────────────────────
def counts_by_group(records: list[dict], group_of: dict) -> dict:
    """{group: array (K, 4) of [tn, fp, fn, tp] per clause}."""
    out = defaultdict(lambda: np.zeros((K, 4)))
    for r in records:
        out[group_of[r["contract_title"]]][CT_INDEX[r["clause_type"]], 2 * r["label"] + r["pred"]] += 1
    return out


def macro_from_counts(c: np.ndarray) -> dict:
    """Macro P/R/F1 for count arrays shaped (..., K, 4); clauses without positives are skipped (NaN-safe)."""
    tn, fp, fn, tp = (c[..., i] for i in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    valid = (tp + fn) > 0
    n = valid.sum(-1)
    return {name: np.where(n > 0, (arr * valid).sum(-1) / np.maximum(n, 1), np.nan)
            for name, arr in (("precision", p), ("recall", r), ("f1", f1))}


def _ci(v) -> list[float]:
    v = np.asarray(v)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired_group_bootstrap(counts_a: dict, counts_b: dict, n: int = 2000, seed: int = 42) -> dict:
    keys = sorted(set(counts_a) | set(counts_b))
    zero = np.zeros((K, 4))
    A = np.stack([counts_a.get(k, zero) for k in keys])
    B = np.stack([counts_b.get(k, zero) for k in keys])
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(n, len(keys)))
    ma, mb = macro_from_counts(A[idx].sum(1)), macro_from_counts(B[idx].sum(1))
    point_a, point_b = macro_from_counts(A.sum(0)), macro_from_counts(B.sum(0))

    diff = mb["f1"] - ma["f1"]
    ok = ~np.isnan(diff)
    return {
        "n_bootstrap": n, "n_groups": len(keys),
        **{f"a_{m}": float(point_a[m]) for m in ("precision", "recall", "f1")},
        **{f"a_{m}_ci": _ci(ma[m]) for m in ("precision", "recall", "f1")},
        **{f"b_{m}": float(point_b[m]) for m in ("precision", "recall", "f1")},
        **{f"b_{m}_ci": _ci(mb[m]) for m in ("precision", "recall", "f1")},
        "diff_f1": float(point_b["f1"] - point_a["f1"]), "diff_f1_ci": _ci(diff),
        "p_value_one_sided": float((diff[ok] <= 0).mean()),
    }


# ── Average precision (threshold-free) ─────────────────────────────────────────
def build_ap_data(chunks: list[dict], margins: dict, group_of: dict, level: str) -> dict:
    """{group: {clause: (labels, scores)}} at chunk level, or contract level (score = max chunk margin)."""
    data = defaultdict(lambda: {ct: ([], []) for ct in CLAUSE_TYPES})
    if level == "chunk":
        for ch in chunks:
            g = group_of[ch["contract_title"]]
            for ct in CLAUSE_TYPES:
                status = ch["labels"][ct]["status"]
                if status != "partial":
                    data[g][ct][0].append(int(status == "positive"))
                    data[g][ct][1].append(margins[ch["chunk_id"]][ct])
    else:
        by_contract = defaultdict(list)
        for ch in chunks:
            by_contract[ch["contract_title"]].append(ch)
        for title, chs in by_contract.items():
            g = group_of[title]
            for ct in CLAUSE_TYPES:
                statuses = [c["labels"][ct]["status"] for c in chs]
                gold = "positive" in statuses
                if not gold and "partial" in statuses:
                    continue
                data[g][ct][0].append(int(gold))
                data[g][ct][1].append(max(margins[c["chunk_id"]][ct] for c in chs))
    return {g: {ct: (np.array(v[0]), np.array(v[1], dtype=float)) for ct, v in d.items()} for g, d in data.items()}


def macro_ap(data: dict, groups: list) -> tuple[float, dict]:
    """Macro average precision over clauses that have positives, for the given (possibly repeated) groups."""
    from sklearn.metrics import average_precision_score

    per = {}
    for ct in CLAUSE_TYPES:
        y = np.concatenate([data[g][ct][0] for g in groups])
        if y.sum() == 0:
            continue
        per[ct] = float(average_precision_score(y, np.concatenate([data[g][ct][1] for g in groups])))
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def ap_bootstrap(data_a: dict, data_b: dict, n: int = 1000, seed: int = 42) -> dict:
    keys = sorted(set(data_a) & set(data_b))
    point_a, per_a = macro_ap(data_a, keys)
    point_b, per_b = macro_ap(data_b, keys)
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _ in range(n):
        groups = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        x, y = macro_ap(data_a, groups)[0], macro_ap(data_b, groups)[0]
        if x == x and y == y:  # skip draws with no positives at all (NaN)
            a.append(x)
            b.append(y)
    a, b = np.array(a), np.array(b)
    return {"n_bootstrap": n, "a_ap": point_a, "a_ap_ci": _ci(a), "b_ap": point_b, "b_ap_ci": _ci(b),
            "diff_ap": point_b - point_a, "diff_ap_ci": _ci(b - a), "p_value_one_sided": float(((b - a) <= 0).mean()),
            "per_clause_a": per_a, "per_clause_b": per_b}


# ── Report ─────────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clause_table(sa: dict, sb: dict, level: str, names, support: dict) -> list[str]:
    na, nb = names
    L = [f"| Clause | Positives | {na} P | {na} R | {na} F1 | {nb} P | {nb} R | {nb} F1 |", "|---|---|---|---|---|---|---|---|"]
    for ct in CLAUSE_TYPES:
        if ct not in sa[level]:
            continue
        a, b = sa[level][ct], sb[level][ct]
        flag = " ⚠️" if support[ct] < 30 else ""
        L.append(f"| {ct}{flag} | {support[ct]} | {a['precision']:.3f} | {a['recall']:.3f} | {a['f1']:.3f} | "
                 f"{b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} |")
    a, b = sa[level]["macro"], sb[level]["macro"]
    L.append(f"| **Macro** | | {a['precision']:.3f} | {a['recall']:.3f} | **{a['f1']:.3f}** | "
             f"{b['precision']:.3f} | {b['recall']:.3f} | **{b['f1']:.3f}** |")
    return L


def fmt_ci(ci) -> str:
    return f"{ci[0]:.3f}–{ci[1]:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--val", help="validation margins of --system (threshold selection)")
    parser.add_argument("--val-baseline", help="validation margins of --baseline (threshold selection)")
    parser.add_argument("--out", default="reports/eval/document_level")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--ap-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    from compare_results import group_map

    A, B = load_json(args.baseline), load_json(args.system)
    split = A["split"]
    chunks = load_json(f"data/processed/chunks_{split}.json")
    ids = {c["chunk_id"] for c in chunks}
    for s in (A, B):
        missing = ids - set(s["margins"])
        assert not missing, f"{s['system']}: {len(missing)} chunks without scores"
    na, nb = A["system"], B["system"]
    titles = {c["contract_title"] for c in chunks}
    group_of = group_map(titles)

    support_chunks, gold_contracts = defaultdict(int), defaultdict(set)
    for ch in chunks:
        for ct in CLAUSE_TYPES:
            if ch["labels"][ct]["status"] == "positive":
                support_chunks[ct] += 1
                gold_contracts[ct].add(ch["contract_title"])
    support_contracts = {ct: len(gold_contracts[ct]) for ct in CLAUSE_TYPES}

    report = {"systems": [na, nb], "split": split, "dataset_version": A["dataset_version"],
              "n_chunks": len(chunks), "n_contracts": len(titles), "n_groups": len(set(group_of.values())),
              "chunk_positives": dict(support_chunks), "contract_positives": support_contracts}

    # Thresholds: natural 0 for both; optionally tuned on validation (separately per system)
    tuned = {}
    vchunks = load_json("data/processed/chunks_val.json") if (args.val or args.val_baseline) else None
    report["threshold_selection"] = {}
    for name, path in ((na, args.val_baseline), (nb, args.val)):
        if path:
            rows = sweep(vchunks, load_json(path)["margins"], np.arange(-3.0, 8.01, 0.5))
            best = select_threshold(rows, "chunk_f1")
            tuned[name] = best["threshold"]
            report["threshold_selection"][name] = {"selected_on": "validation, max chunk-level macro-F1",
                                                   "selected_threshold": best["threshold"], "sweep": rows}

    def evaluate(system, t):
        m = system["margins"]
        cr, co = chunk_records(chunks, m, t), contract_records(chunks, m, t)
        return cr, co, summarize(cr, co)

    def compare(ta, tb):
        ca, oa, sa = evaluate(A, ta)
        cb, ob, sb = evaluate(B, tb)
        boot = {"chunk": paired_group_bootstrap(counts_by_group(ca, group_of), counts_by_group(cb, group_of), args.bootstrap),
                "contract": paired_group_bootstrap(counts_by_group(oa, group_of), counts_by_group(ob, group_of), args.bootstrap)}
        return {"thresholds": {na: ta, nb: tb}, "summary": {na: sa, nb: sb}, "bootstrap": boot}

    report["operating_points"] = {"natural": compare(0.0, 0.0)}
    if na in tuned and nb in tuned:
        report["operating_points"]["tuned"] = compare(tuned[na], tuned[nb])

    report["average_precision"] = {
        level: ap_bootstrap(build_ap_data(chunks, A["margins"], group_of, level),
                            build_ap_data(chunks, B["margins"], group_of, level), args.ap_bootstrap)
        for level in ("chunk", "contract")}

    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    # ── Markdown ───────────────────────────────────────────────────────────────
    nat = report["operating_points"]["natural"]
    neg = nat["summary"][na]["chunk_negatives"]
    pos = sum(support_chunks.values())
    L = [f"# Full-document evaluation — {na} vs {nb}\n",
         f"Dataset {A['dataset_version']} **{split}** split: **{len(titles)} contracts** ({report['n_groups']} near-duplicate groups), "
         f"**{len(chunks):,} chunks**, every chunk scored for all 6 clauses: {pos} positive and {neg:,} negative "
         f"(chunk, clause) pairs (**{pos / (pos + neg):.1%} prevalence**; 13 partial pairs excluded). "
         "A clause is flagged when its first-token margin exceeds the threshold.\n"]

    def op_rows(op):
        t, out = op["thresholds"], []
        for name in (na, nb):
            s = op["summary"][name]
            out.append(f"| {name} @ {t[name]:+.1f} | {s['chunk']['macro']['precision']:.3f} | {s['chunk']['macro']['recall']:.3f} | "
                       f"**{s['chunk']['macro']['f1']:.3f}** | {s['false_flagged_chunks_per_contract']:.1f} | "
                       f"{s['contract']['macro']['precision']:.3f} | {s['contract']['macro']['recall']:.3f} | "
                       f"**{s['contract']['macro']['f1']:.3f}** | {s['wrong_clause_flags_per_contract']:.2f} |")
        return out

    L += ["## Summary: operating points\n",
          "| System @ threshold | Chunk P | Chunk R | Chunk F1 | Falsely flagged chunks / contract | Contract P | Contract R | Contract F1 | Wrong clause flags / contract |",
          "|---|---|---|---|---|---|---|---|---|"]
    L += op_rows(nat)
    if "tuned" in report["operating_points"]:
        L += op_rows(report["operating_points"]["tuned"])
    L.append("")

    for key, title in (("natural", "Natural threshold 0"), ("tuned", "Thresholds tuned on validation")):
        if key not in report["operating_points"]:
            continue
        op = report["operating_points"][key]
        sa, sb = op["summary"][na], op["summary"][nb]
        bc, ko = op["bootstrap"]["chunk"], op["bootstrap"]["contract"]
        t = op["thresholds"]
        L += [f"## {title} ({na} {t[na]:+.1f}, {nb} {t[nb]:+.1f})\n", "### Chunk level, every chunk of every contract\n"]
        L += clause_table(sa, sb, "chunk", (na, nb), support_chunks)
        L += ["", f"- Macro-F1 {na} **{bc['a_f1']:.3f}** (95% CI {fmt_ci(bc['a_f1_ci'])}) → {nb} **{bc['b_f1']:.3f}** (95% CI {fmt_ci(bc['b_f1_ci'])}); "
                  f"difference {bc['diff_f1']:+.3f} (95% CI {bc['diff_f1_ci'][0]:+.3f} to {bc['diff_f1_ci'][1]:+.3f}), one-sided p = {bc['p_value_one_sided']:.4f}",
              f"- Negative pairs flagged: {na} {round(sa['chunk_false_positive_rate'] * neg):,} ({sa['chunk_false_positive_rate']:.2%}) vs "
              f"{nb} {round(sb['chunk_false_positive_rate'] * neg):,} ({sb['chunk_false_positive_rate']:.2%})", "",
              "### Contract level: is each clause flagged in each contract?\n"]
        L += clause_table(sa, sb, "contract", (na, nb), support_contracts)
        L += ["", f"- Macro-F1 {na} **{ko['a_f1']:.3f}** ({fmt_ci(ko['a_f1_ci'])}) → {nb} **{ko['b_f1']:.3f}** ({fmt_ci(ko['b_f1_ci'])}); "
                  f"difference {ko['diff_f1']:+.3f} (95% CI {ko['diff_f1_ci'][0]:+.3f} to {ko['diff_f1_ci'][1]:+.3f})",
              f"- Contracts with no wrong clause flag: {na} {sa['contracts_with_no_wrong_flag']:.0%} vs {nb} {sb['contracts_with_no_wrong_flag']:.0%}",
              f"- Highest-scoring chunk (the one the API quotes) really is the clause: {na} {sa['top_chunk_on_clause_rate']:.1%} vs "
              f"{nb} {sb['top_chunk_on_clause_rate']:.1%} of correctly flagged contract-clauses\n"]

    L += ["## Threshold-free ranking quality: average precision (AP)\n",
          f"| Clause | Chunk AP {na} | Chunk AP {nb} | Contract AP {na} | Contract AP {nb} |", "|---|---|---|---|---|"]
    apc, apk = report["average_precision"]["chunk"], report["average_precision"]["contract"]
    for ct in CLAUSE_TYPES:
        L.append(f"| {ct} | {apc['per_clause_a'][ct]:.3f} | {apc['per_clause_b'][ct]:.3f} | {apk['per_clause_a'][ct]:.3f} | {apk['per_clause_b'][ct]:.3f} |")
    L.append(f"| **Macro** | **{apc['a_ap']:.3f}** | **{apc['b_ap']:.3f}** | **{apk['a_ap']:.3f}** | **{apk['b_ap']:.3f}** |")
    L += ["", f"- Chunk-level macro AP difference {apc['diff_ap']:+.3f} (95% CI {apc['diff_ap_ci'][0]:+.3f} to {apc['diff_ap_ci'][1]:+.3f}); "
              f"contract-level {apk['diff_ap']:+.3f} (95% CI {apk['diff_ap_ci'][0]:+.3f} to {apk['diff_ap_ci'][1]:+.3f})", ""]

    for name, sel in report["threshold_selection"].items():
        keep = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, sel["selected_threshold"]}
        L += [f"## Validation sweep — {name} (selected threshold {sel['selected_threshold']:+.1f})\n",
              "| Threshold | Chunk P | Chunk R | Chunk F1 | Contract F1 |", "|---|---|---|---|---|"]
        L += [f"| {r['threshold']:+.1f} | {r['chunk_precision']:.3f} | {r['chunk_recall']:.3f} | {r['chunk_f1']:.3f} | {r['contract_f1']:.3f} |"
              for r in sel["sweep"] if r["threshold"] in keep]
        L.append("")
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
