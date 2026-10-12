"""
Compare two evaluation result files from src/evaluate.py (alignment_plan D3).

  - per-clause precision / recall / F1 for both systems
  - paired bootstrap over near-duplicate contract GROUPS: 95% CIs for each system's
    macro-F1 and for the difference, plus a one-sided p-value
  - exact McNemar test on per-query correctness
  - evidence grounding: share of generated quotes found in the source chunk,
    and span-overlap F1 against the gold clause on true positives
  - classes with < MIN_SUPPORT positives are flagged as insufficient support

Writes reports/eval/comparison.json and reports/eval/comparison.md.

Usage:
  python src/compare_results.py reports/eval/baseline_test.json reports/eval/lora_v4_test.json
"""

import json
import sys
from collections import defaultdict

from eval_metrics import mcnemar, paired_bootstrap, per_clause_metrics

MIN_SUPPORT = 30
OUT_JSON = "reports/eval/comparison.json"
OUT_MD = "reports/eval/comparison.md"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_map(titles: set[str]) -> dict:
    """Map contract title -> near-duplicate group (from the dataset manifest)."""
    with open("data/processed/manifest.json", "r", encoding="utf-8") as f:
        pairs = json.load(f)["near_duplicate_pairs"]
    parent = {t: t for t in titles}

    def find(t):
        while parent[t] != t:
            t = parent[t]
        return t

    for p in pairs:
        if p["a"] in parent and p["b"] in parent:
            parent[find(p["b"])] = find(p["a"])
    return {t: find(t) for t in titles}


def evidence_stats(records):
    quoted = [r for r in records if "quote" in r]
    tps = [r for r in quoted if r["label"] == 1]
    if not quoted:
        return None
    return {
        "quotes": len(quoted),
        "grounded_rate": sum(r["grounded"] for r in quoted) / len(quoted),
        "exact_rate": sum(r["exact"] for r in quoted) / len(quoted),
        "tp_quotes": len(tps),
        "tp_span_f1_mean": sum(r["span_f1"] for r in tps) / len(tps) if tps else 0.0,
        "tp_grounded_rate": sum(r["grounded"] for r in tps) / len(tps) if tps else 0.0,
    }


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    A, B = load(a_path), load(b_path)
    ra, rb = A["records"], B["records"]
    na, nb = A["system"], B["system"]

    ma, mb = per_clause_metrics(ra), per_clause_metrics(rb)
    groups = group_map({r["contract_title"] for r in ra})
    boot = paired_bootstrap(ra, rb, groups)
    mc = mcnemar(ra, rb)
    ev = {na: evidence_stats(ra), nb: evidence_stats(rb)}
    fmt = {na: sum(r["top_token_ok"] for r in ra) / len(ra), nb: sum(r["top_token_ok"] for r in rb) / len(rb)}

    support = defaultdict(lambda: [0, 0])
    for r in ra:
        support[r["clause_type"]][0 if r["label"] else 1] += 1

    report = {
        "systems": [na, nb],
        "dataset_version": A["dataset_version"],
        "n_queries": len(ra),
        "per_clause": {na: ma, nb: mb},
        "support": {k: {"pos": v[0], "neg": v[1]} for k, v in support.items()},
        "paired_bootstrap_groups": boot,
        "mcnemar": mc,
        "format_compliance_top_token": fmt,
        "evidence": ev,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── Markdown ───────────────────────────────────────────────────────────────
    L = []
    L.append(f"# Test-set evaluation — {na} vs {nb}\n")
    L.append(f"Dataset {A['dataset_version']} test split: **{len(ra)} queries** "
             f"({sum(r['label'] for r in ra)} positive / {sum(1 - r['label'] for r in ra)} negative), "
             f"{boot['n_groups']} contract groups. Decision = first-token logit `Found` vs `None`.\n")
    L.append(f"| Clause | Pos/Neg | {na} P | {na} R | {na} F1 | {nb} P | {nb} R | {nb} F1 | ΔF1 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for ct in [k for k in ma if k != "macro"]:
        s = support[ct]
        flag = " ⚠️" if s[0] < MIN_SUPPORT else ""
        a, b = ma[ct], mb[ct]
        L.append(f"| {ct}{flag} | {s[0]}/{s[1]} | {a['precision']:.3f} | {a['recall']:.3f} | {a['f1']:.3f} | "
                 f"{b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} | {b['f1'] - a['f1']:+.3f} |")
    a, b = ma["macro"], mb["macro"]
    L.append(f"| **Macro** | | {a['precision']:.3f} | {a['recall']:.3f} | **{a['f1']:.3f}** | "
             f"{b['precision']:.3f} | {b['recall']:.3f} | **{b['f1']:.3f}** | **{b['f1'] - a['f1']:+.3f}** |")
    L.append(f"\n⚠️ = fewer than {MIN_SUPPORT} positive test queries — per-class numbers are unreliable.\n")
    L.append("## Significance (paired, resampling contract groups)\n")
    L.append(f"- {na} macro-F1 **{boot['a_macro_f1']:.3f}** (95% CI {boot['a_ci'][0]:.3f}–{boot['a_ci'][1]:.3f})")
    L.append(f"- {nb} macro-F1 **{boot['b_macro_f1']:.3f}** (95% CI {boot['b_ci'][0]:.3f}–{boot['b_ci'][1]:.3f})")
    L.append(f"- Difference **{boot['diff']:+.3f}** (95% CI {boot['diff_ci'][0]:+.3f} to {boot['diff_ci'][1]:+.3f}), "
             f"one-sided p = {boot['p_value_one_sided']:.4f} ({boot['n_bootstrap']} draws)")
    L.append(f"- McNemar: only {nb} correct on {mc['only_b_correct']} queries, only {na} correct on "
             f"{mc['only_a_correct']}; two-sided p = {mc['p_value_two_sided']:.2e}\n")
    L.append("## Format compliance (unconstrained top token is `Found`/`None`)\n")
    L.append(f"- {na}: {fmt[na]:.1%} · {nb}: {fmt[nb]:.1%}\n")
    if ev[na] or ev[nb]:
        L.append("## Evidence quality (quotes generated for predicted positives)\n")
        L.append("| System | Quotes | Grounded in source | Exact substring | TP span-F1 vs gold |")
        L.append("|---|---|---|---|---|")
        for n in (na, nb):
            e = ev[n]
            if e:
                L.append(f"| {n} | {e['quotes']} | {e['grounded_rate']:.1%} | {e['exact_rate']:.1%} | {e['tp_span_f1_mean']:.3f} |")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
