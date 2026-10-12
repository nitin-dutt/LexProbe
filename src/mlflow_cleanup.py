"""
Mark orphaned MLflow runs (stuck in RUNNING after a crash) as KILLED (alignment_plan A6).

Dry run by default — lists what would change. Nothing is deleted; runs keep their
params, metrics and artifacts, only their status is corrected.

Usage:
  python src/mlflow_cleanup.py                  # list RUNNING runs older than 24h
  python src/mlflow_cleanup.py --apply          # mark them KILLED
  python src/mlflow_cleanup.py --older-than-hours 6 --apply
"""

import argparse
import time
from datetime import datetime

from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--older-than-hours", type=float, default=24)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    client = MlflowClient()
    cutoff_ms = (time.time() - args.older_than_hours * 3600) * 1000

    stale = []
    for exp in client.search_experiments():
        for run in client.search_runs([exp.experiment_id], filter_string="attributes.status = 'RUNNING'"):
            if run.info.start_time < cutoff_ms:
                stale.append((exp.name, run))

    if not stale:
        print("No stale RUNNING runs found.")
        return

    for exp_name, run in stale:
        started = datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M")
        print(f"  [{exp_name}] {run.info.run_name}  ({run.info.run_id})  started {started}")

    if args.apply:
        for _, run in stale:
            client.set_terminated(run.info.run_id, status="KILLED")
        print(f"Marked {len(stale)} run(s) as KILLED.")
    else:
        print(f"{len(stale)} stale run(s). Re-run with --apply to mark them KILLED.")


if __name__ == "__main__":
    main()
