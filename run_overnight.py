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
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml

LOG_FILE = "logs/overnight_run.log"


def log(msg, to_file=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if to_file:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_step(name, cmd_args):
    log("=" * 60)
    log(f"STARTING: {name}")
    log(f"Command:  {' '.join(cmd_args)}")
    log("=" * 60)

    start = time.time()

    # Stream output to BOTH terminal and log file simultaneously (tee behavior)
    # Force UTF-8 in child process to prevent cp1252 crashes on Windows terminals
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
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
        adapter_dir = yaml.safe_load(f)["output_dir"]

    finetune_args = ["src/finetune.py", "--config", args.config] + (["--resume"] if args.resume else [])
    steps = [
        ("Step 1/3 - Token budget check", ["src/finetune.py", "--config", args.config, "--check-only"]),
        ("Step 2/3 - QLoRA fine-tuning", finetune_args),
        ("Step 3/3 - Evaluate on frozen test set (baseline vs adapter)",
         ["src/evaluate.py", "--adapter", os.path.join(adapter_dir, "final_adapter")]),
    ]
    if args.resume:
        steps = steps[1:]

    for name, cmd in steps:
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
