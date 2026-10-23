"""
Overnight Training Pipeline - LexProbe
======================================
Run this once before bed. It will:
  1. Check the token budget for the config (no model load)
  2. Fine-tune QLoRA from the YAML config (eval + checkpoint every N steps)
  3. Evaluate the new adapter vs the zero-shot baseline on the frozen test set
     (then: python src/compare_results.py reports/eval/baseline_test.json reports/eval/<name>_test.json)

The dataset is NOT rebuilt here: evaluation sets are frozen once built
(alignment_plan A1). Build it separately with `python src/prepare_data.py`.

Usage:
  python run_overnight.py                                  # configs/qlora_v4.yaml
  python run_overnight.py --config configs/qlora_v4.yaml
  python run_overnight.py --resume                         # continue training after a crash
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

import yaml

LOG_FILE = "logs/overnight_run.log"
PROFILE = "v3"  # set from the config's `profile:` in __main__


def log(msg, to_file=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if to_file:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def preflight(max_used_mib=1500, min_free_gb=20, min_power_w=40):
    """Refuse to start if the GPU is busy (shared with another project) or the disk is nearly full."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.split()[0]
        used = int(out)
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        log("PREFLIGHT FAILED: could not read GPU memory with nvidia-smi.")
        sys.exit(2)
    free_gb = shutil.disk_usage(".").free / 1e9
    # A low power cap (laptop power mode, charger) made one run ~20x slower, so refuse to start under it
    limit_w = None
    try:
        q = subprocess.run(["nvidia-smi", "-q", "-d", "POWER"], capture_output=True, text=True).stdout
        m = re.search(r"Current Power Limit\s*:\s*([\d.]+) W", q)
        limit_w = float(m.group(1)) if m else None
    except (OSError, ValueError):
        pass
    log(f"Preflight: GPU memory in use {used} MiB (limit {max_used_mib}), disk free {free_gb:.0f} GB (min {min_free_gb}), GPU power limit {limit_w} W (min {min_power_w})")
    if limit_w is not None and limit_w < min_power_w:
        log("PREFLIGHT FAILED: the GPU power limit is very low (laptop power mode or charger?). Training would crawl. Nothing was started.")
        sys.exit(2)
    if used > max_used_mib or free_gb < min_free_gb:
        log("PREFLIGHT FAILED: GPU is in use by something else or the disk is nearly full. Nothing was started.")
        sys.exit(2)


def run_step(name, cmd_args, exit_on_fail=True):
    log("=" * 60)
    log(f"STARTING: {name}")
    log(f"Command:  {' '.join(cmd_args)}")
    log("=" * 60)

    start = time.time()

    # Stream output to BOTH terminal and log file simultaneously (tee behavior)
    # Force UTF-8 in child process to prevent cp1252 crashes on Windows terminals
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["LEXPROBE_PROFILE"] = PROFILE
    child_env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        [sys.executable, "-u", *cmd_args],     # -u = unbuffered output
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,             # Merge stderr into stdout
        text=True,
        encoding="utf-8",
        bufsize=1,                            # Line-buffered
        env=child_env,
    )

    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        for line in process.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)           # Live terminal output
            log_f.write(line + "\n")          # Also write to log file
            log_f.flush()

    process.wait()
    elapsed = time.time() - start

    if process.returncode != 0 and not exit_on_fail:
        log(f"FAILED: {name} (exit code {process.returncode}) after {elapsed/60:.1f} min (will retry)")
        return None
    if process.returncode != 0:
        log(f"FAILED: {name} (exit code {process.returncode}) after {elapsed/60:.1f} min")
        log("Stopping pipeline - check logs/overnight_run.log for details.")
        # Restore sleep before exiting on failure
        subprocess.run(["powercfg", "/change", "standby-timeout-ac", "30"], check=False)
        subprocess.run(["powercfg", "/change", "monitor-timeout-ac", "15"], check=False)
        sys.exit(process.returncode)
    else:
        log(f"DONE: {name} in {elapsed/60:.1f} minutes")

    return elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qlora_v4.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    log("LexProbe Overnight Pipeline Started")
    log(f"   Python: {sys.executable}")
    log(f"   Config: {args.config}")

    # Prevent Windows from sleeping during training (root cause of the v3 crash)
    log("Disabling Windows sleep for duration of training...")
    subprocess.run(["powercfg", "/change", "standby-timeout-ac", "0"], check=False)
    subprocess.run(["powercfg", "/change", "monitor-timeout-ac", "0"], check=False)
    log("Sleep disabled. Will restore to 30min/15min after pipeline completes.")

    pipeline_start = time.time()

    with open(args.config, "r", encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)
    PROFILE = run_cfg.get("profile", "v3")  # dataset profile: v3 (6 clauses) or v4 (7 clauses)
    adapter_dir = run_cfg["output_dir"]
    final_adapter = os.path.join(adapter_dir, "final_adapter")
    preflight()

    finetune_args = ["src/finetune.py", "--config", args.config] + (["--resume"] if args.resume else [])
    if run_cfg.get("post_eval") == "documents":
        # Score every chunk of val, then test, ONCE. Test is scored here and not again: thresholds and
        # weights are fitted afterwards on val only. Systems: adapter only (baseline scores already saved,
        # profile v3) or baseline+adapter (a new clause set needs the baseline re-scored).
        ename = run_cfg.get("eval_name", "lora_new")
        systems = run_cfg.get("post_eval_systems", "adapter")
        evals = []
        for split in ("val", "test"):
            evals.append((f"Score every {split} chunk ({systems}; {ename})",
                          ["src/evaluate_documents.py", "--adapter", final_adapter, "--name", ename, "--split", split, "--systems", systems]))
        if run_cfg.get("post_eval_sampled"):
            # decisions + quotes on the sampled test: gives the verbatim-quote rate (target T6)
            evals.append((f"Sampled test with quotes (baseline vs {ename})", ["src/evaluate.py", "--adapter", final_adapter, "--name", ename] + (["--skip-baseline"] if systems == "adapter" else [])))
    elif run_cfg.get("post_eval") == "none":
        evals = []
    else:
        evals = [("Evaluate on frozen test set (baseline vs adapter)", ["src/evaluate.py", "--adapter", final_adapter])]
    total_steps = 2 + (1 if run_cfg.get("init_adapter") and run_cfg.get("validation", {}).get("eval_at_start") else 0) + len(evals)

    if not args.resume:
        run_step(f"Step 1/{total_steps} - Token budget check", ["src/finetune.py", "--config", args.config, "--check-only"])
        if run_cfg.get("init_adapter") and run_cfg.get("validation", {}).get("eval_at_start"):
            # Separate process on purpose: scoring and then training in one process made training ~20x slower
            run_step("Score the starting adapter (the bar to beat)", ["src/finetune.py", "--config", args.config, "--start-eval-only"])

    # Training: if it dies (e.g. out of memory because another program took the GPU), resume from the
    # latest checkpoint up to MAX_RETRIES times. A crash of the whole PC needs `--resume` by hand.
    MAX_RETRIES = 3
    resume = args.resume
    for attempt in range(1, MAX_RETRIES + 2):
        cmd = ["src/finetune.py", "--config", args.config] + (["--resume"] if resume else [])
        if run_step(f"Step 2/{total_steps} - QLoRA fine-tuning (attempt {attempt})", cmd, exit_on_fail=False) is not None:
            break
        if attempt > MAX_RETRIES:
            log("Training failed and the retries are used up. Stopping.")
            subprocess.run(["powercfg", "/change", "standby-timeout-ac", "30"], check=False)
            subprocess.run(["powercfg", "/change", "monitor-timeout-ac", "15"], check=False)
            sys.exit(1)
        resume = bool(glob.glob(os.path.join(adapter_dir, "checkpoint-*")))  # no checkpoint yet: restart from scratch
        log("Waiting 60 s, then " + ("resuming from the latest checkpoint..." if resume else "restarting (no checkpoint yet)..."))
        time.sleep(60)

    if not os.path.exists(os.path.join(final_adapter, "adapter_model.safetensors")):
        log("Training finished but final_adapter was not written. Stopping.")
        sys.exit(1)
    for name, cmd in evals:
        run_step(name, cmd)

    total = time.time() - pipeline_start

    # Restore Windows sleep settings
    log("Restoring Windows sleep settings (30min standby / 15min monitor)...")
    subprocess.run(["powercfg", "/change", "standby-timeout-ac", "30"], check=False)
    subprocess.run(["powercfg", "/change", "monitor-timeout-ac", "15"], check=False)

    log("=" * 60)
    log(f"ALL STEPS COMPLETE - Total time: {total/3600:.2f} hours")
    log(f"   Adapter + training_summary.json: see output_dir in {args.config}")
    log(f"   Full log: {LOG_FILE}")
    log("=" * 60)
