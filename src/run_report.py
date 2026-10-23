"""
Detailed report of a training + scoring run, built from the files the run leaves behind
(CPU only; nothing here touches the GPU).

Sources: the run's console log, the orchestrator log, <run dir>/start_metrics.json,
the newest checkpoint's trainer_state.json (full loss/eval history), final_adapter/training_summary.json.

Usage:
  python src/run_report.py --log logs/v6_run.log --run-dir models/llama-3.1-8b-cuad-lora-v6 \
      --out reports/run_v6_report.md [--steps-total 1110]
"""

import argparse
import glob
import json
import os
import re
import statistics
from datetime import datetime


def read(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().replace("\r", "\n")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_ts(line):
    m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None


def stages(text):
    """[(name, start, end|None, status)] from STARTING / DONE / FAILED lines (duplicates removed)."""
    seen, out, open_ = set(), [], {}
    for line in text.splitlines():
        if line in seen or not line.startswith("["):
            continue
        seen.add(line)
        t = parse_ts(line)
        m = re.search(r"STARTING: (.*)", line)
        if m:
            open_[m.group(1)] = t
            out.append([m.group(1), t, None, "running"])
            continue
        m = re.search(r"DONE: (.*) in [\d.]+ minutes", line)
        if m:
            for s in reversed(out):
                if s[0] == m.group(1) and s[3] == "running":
                    s[2], s[3] = t, "done"
                    break
            continue
        m = re.search(r"FAILED: (.*) \(exit code (\d+)\)", line)
        if m:
            for s in reversed(out):
                if s[0] == m.group(1) and s[3] == "running":
                    s[2], s[3] = t, f"FAILED (exit {m.group(2)})"
                    break
    return out


def step_speeds(text):
    """seconds per optimizer step from the tqdm lines of the training bars (ignores eval bars)."""
    vals = []
    for m in re.finditer(r"\|\s*(\d+)/(\d+) \[[\d:]+<[\d:?]+,\s*([\d.]+)s/it\]", text):
        step, total, spi = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if total > 200:
            vals.append((step, spi))
    return vals


def history(run_dir):
    states = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json")),
                    key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    if not states:
        return None
    return load_json(states[-1])


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def build(args):
    text = read(args.log)
    L = ["# Run report", "", f"_Generated {datetime.now():%Y-%m-%d %H:%M} from `{args.log}` and `{args.run_dir}`._", ""]

    # --- timeline -----------------------------------------------------------------
    L += ["## Timeline", "", "| stage | start | end | minutes | status |", "|---|---|---|---|---|"]
    st = stages(read("logs/overnight_run.log")) if os.path.exists("logs/overnight_run.log") else []
    log_stages = stages(text)
    for name, a, b, status in (log_stages or st):
        mins = f"{(b - a).total_seconds() / 60:.1f}" if a and b else "-"
        L.append(f"| {name} | {a:%m-%d %H:%M} | {b:%H:%M} | {mins} | {status} |" if b else f"| {name} | {a:%m-%d %H:%M} | - | {mins} | {status} |")
    fails = [s for s in log_stages if s[3].startswith("FAILED")]
    L += ["", f"Failures / restarts: **{len(fails)}**" + (" (" + "; ".join(f"{s[0]} at {s[2]:%H:%M}" for s in fails) + ")" if fails else ""), ""]

    # --- incidents ----------------------------------------------------------------
    inc = sorted({l.strip()[:200] for l in text.splitlines()
                  if re.search(r"SLOW STEPS|out of memory|OutOfMemory|CUBLAS_STATUS|PREFLIGHT FAILED|NaN|nan loss", l)})
    L += ["## Incidents", ""] + ([f"- `{i}`" for i in inc] if inc else ["- none logged"]) + [""]

    # --- speed ----------------------------------------------------------------------
    sp = step_speeds(text)
    if sp:
        per = [v for _, v in sp]
        # tqdm shows a running average; report the last value per 100-step block
        blocks = {}
        for step, v in sp:
            blocks[step // 100] = v
        L += ["## Training speed", "", f"- steps logged: {len(sp)}; running-average s/step: median {statistics.median(per):.1f}, max {max(per):.1f}",
              "- running average at the end of each 100 steps: " + ", ".join(f"{k * 100}: {v:.1f}s" for k, v in sorted(blocks.items())), ""]

    # --- scores per checkpoint ---------------------------------------------------------
    hist = history(args.run_dir)
    start = load_json(os.path.join(args.run_dir, "start_metrics.json")) if os.path.exists(os.path.join(args.run_dir, "start_metrics.json")) else {}
    if hist:
        by_step = {}
        for e in hist["log_history"]:
            by_step.setdefault(e["step"], {}).update(e)
        evals = {s: e for s, e in by_step.items() if "eval_realistic_ap" in e}
        clauses = sorted({k.replace("eval_realistic_ap_", "") for e in evals.values() for k in e if k.startswith("eval_realistic_ap_")})
        L += ["## Checkpoint scores (validation, realistic-prevalence average precision = the selection metric)", ""]
        head = "| step | AP (macro) | " + " | ".join(c[:14] for c in clauses) + " | eval loss | decision F1 |"
        L += [head, "|" + "---|" * (len(clauses) + 5)]
        if start:
            L.append("| start | **" + fmt(start.get("eval_realistic_ap")) + "** | " + " | ".join(fmt(start.get(f"eval_realistic_ap_{c}")) for c in clauses)
                     + f" | {fmt(start.get('eval_eval_loss', start.get('eval_loss')), 4)} | {fmt(start.get('eval_decision_macro_f1'))} |")
        best = hist.get("best_global_step")
        for s in sorted(evals):
            e = evals[s]
            mark = " **(best)**" if s == best else ""
            L.append(f"| {s}{mark} | {fmt(e.get('eval_realistic_ap'))} | " + " | ".join(fmt(e.get(f"eval_realistic_ap_{c}")) for c in clauses)
                     + f" | {fmt(e.get('eval_loss'), 4)} | {fmt(e.get('eval_decision_macro_f1'))} |")
        tl = [(e["step"], e["loss"]) for e in hist["log_history"] if "loss" in e]
        if tl:
            L += ["", "Training loss (every logged step): " + ", ".join(f"{s}: {v:.4f}" for s, v in tl[:: max(1, len(tl) // 14)]), ""]
        L += [f"Best checkpoint: step {best}, AP {fmt(hist.get('best_metric'), 4)}; start AP {fmt(start.get('eval_realistic_ap'), 4)}.", ""]
    summ_path = os.path.join(args.run_dir, "final_adapter", "training_summary.json")
    if os.path.exists(summ_path):
        sm = load_json(summ_path)
        L += ["## Training summary", "", f"- improved over start: **{sm.get('improved_over_start')}** (start {fmt(sm.get('start_metric'), 4)} -> best {fmt(sm.get('best_metric'), 4)})",
              f"- best checkpoint: `{sm.get('best_checkpoint')}`", f"- initialised from: `{sm.get('init_adapter')}`",
              f"- dataset {sm.get('dataset_version')}, steps/epoch {sm.get('steps_per_epoch')}, total steps {sm.get('total_steps')}", ""]

    # --- scoring durations ----------------------------------------------------------------
    sc = re.findall(r"\[([\w]+)\] done in ([\d.]+) min -> (\S+)", text)
    if sc:
        L += ["## Chunk scoring", "", "| system | minutes | output |", "|---|---|---|"] + [f"| {a} | {b} | `{c}` |" for a, b, c in sc] + [""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    md = build(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {args.out} ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
