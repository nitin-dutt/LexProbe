"""
QLoRA fine-tuning of Llama-3.1-8B-Instruct for clause detection.

Everything about a run comes from a YAML config (alignment_plan A4), and the run is
reproducible from: config + dataset version (manifest hashes) + git commit, all logged
to MLflow and saved next to the adapter.

What this script guarantees:
  - same prompt format as evaluation/serving (src/prompting.py — C2, C4)
  - loss on the answer tokens only (C3)
  - no answer truncation; refuses to start if too many samples exceed the token budget (C1)
  - periodic validation with decision macro-F1; best checkpoint is kept (C5)
  - frequent checkpoints and real resume support (C6)

Usage:
  python src/finetune.py --config configs/qlora_v4.yaml
  python src/finetune.py --config configs/qlora_v4.yaml --resume      # continue after a crash
  python src/finetune.py --config configs/qlora_v4.yaml --check-only  # token report, no model load
"""

import os

# NOTE (2026-10-19): do NOT set PYTORCH_CUDA_ALLOC_CONF here. "garbage_collection_threshold:0.8,max_split_size_mb:256"
# was tried against a CUDA out-of-memory and made GPU scoring ~30x slower (20 items: 152 s vs 5 s) on this card.
# Memory headroom comes from src/answer_loss.py and prediction_loss_only instead (alignment_plan C7).

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time

import yaml
from dotenv import load_dotenv

from training_data import (decision_metrics, realistic_ap_metrics, select_realistic_val, select_val_subset,
                           tokenize_split, verify_train_mix)
from prompting import NEGATIVE_ANSWER, POSITIVE_PREFIX, encode_prompt


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v if not isinstance(v, list) else ",".join(map(str, v))
    return out


def git_state() -> dict:
    """Commit + dirty flag so the run can be traced to code (read-only git calls)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": True}


def load_dataset_files(cfg: dict) -> tuple[list[dict], list[dict], dict]:
    data_dir = cfg["data"]["dir"]
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        sys.exit(f"ERROR: {manifest_path} not found. Build the dataset first: python src/prepare_data.py")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected = cfg["data"]["expected_dataset_version"]
    import prepare_data
    if prepare_data.ACTIVE_PROFILE != expected:
        sys.exit(f"ERROR: config expects dataset {expected} but LEXPROBE_PROFILE={prepare_data.ACTIVE_PROFILE}. "
                 f"Set LEXPROBE_PROFILE={expected} (run_overnight.py does it from the config's `profile:`).")
    if manifest["dataset_version"] != expected:
        sys.exit(f"ERROR: dataset version is {manifest['dataset_version']}, config expects {expected}.")

    mix_dir = cfg["data"].get("train_mix_dir")
    if mix_dir:
        # Training-only mix built from the frozen chunks; val/test stay the DVC-pinned dataset
        with open(os.path.join(mix_dir, "manifest.json"), "r", encoding="utf-8") as f:
            mix_manifest = json.load(f)
        try:
            verify_train_mix(mix_manifest, manifest)
        except ValueError as e:
            sys.exit(f"ERROR: {e}")
        manifest = {**manifest, "train_mix": mix_manifest["mix"],
                    "files": {**manifest["files"], cfg["data"]["train_file"]: mix_manifest["train_sha256"]}}
        train_path = os.path.join(mix_dir, cfg["data"]["train_file"])
    else:
        train_path = os.path.join(data_dir, cfg["data"]["train_file"])
    with open(train_path, "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(os.path.join(data_dir, cfg["data"]["val_file"]), "r", encoding="utf-8") as f:
        val = json.load(f)
    return train, val, manifest


def print_report(name: str, r: dict):
    print(f"  {name:<5} {r['kept']:>5}/{r['samples']:<5} kept | dropped (too long): {r['dropped_too_long']} "
          f"({r['dropped_positive']} pos) | answers trimmed: {r['answers_trimmed']} | "
          f"tokens p50/p95/max: {r['tokens_p50']}/{r['tokens_p95']}/{r['tokens_max']} (limit {r['max_seq_length']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in output_dir")
    parser.add_argument("--check-only", action="store_true", help="Tokenise and report lengths, then exit")
    parser.add_argument("--start-eval-only", action="store_true",
                        help="Score the starting adapter on the validation sets, save start_metrics.json, then exit "
                             "(run in its own process: scoring before training in the same process slows training ~20x)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("ERROR: HF_TOKEN not set (put it in .env).")

    output_dir = cfg["output_dir"]
    final_dir = os.path.join(output_dir, "final_adapter")
    if os.path.exists(final_dir) and not args.resume and not args.check_only:
        sys.exit(f"ERROR: {final_dir} already exists. Use a new output_dir / run_name in the config.")

    from transformers import AutoTokenizer  # deferred so --help works without the ML stack

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], token=hf_token)
    pad = "<|finetune_right_pad_id|>"
    tokenizer.pad_token = pad if pad in tokenizer.get_vocab() else tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Data ───────────────────────────────────────────────────────────────────
    train_samples, val_samples, manifest = load_dataset_files(cfg)
    seq = cfg["sequence"]
    dcfg = cfg["data"]
    print(f"Dataset {manifest['dataset_version']}: {len(train_samples)} train / {len(val_samples)} val queries")

    train_examples, train_report = tokenize_split(
        tokenizer, train_samples, seq["max_seq_length"], dcfg["max_answer_chars"], dcfg["multi_span"])

    val_subset = select_val_subset(val_samples, cfg["validation"]["max_samples"], cfg["training"]["seed"])
    val_examples, val_report = tokenize_split(
        tokenizer, val_subset, seq["max_seq_length"], dcfg["max_answer_chars"], dcfg["multi_span"])

    print("Token budget report:")
    print_report("train", train_report)
    print_report("val", val_report)

    if train_report["dropped_fraction"] > seq["max_dropped_fraction"]:
        sys.exit(
            f"ERROR: {train_report['dropped_fraction']:.1%} of training samples exceed max_seq_length="
            f"{seq['max_seq_length']} (allowed {seq['max_dropped_fraction']:.1%}). Raise max_seq_length if VRAM "
            "allows, lower data.max_answer_chars, or rebuild the dataset with a smaller CHUNK_CHARS."
        )
    if args.check_only:
        print("--check-only: stopping before model load.")
        return

    # ── Model ──────────────────────────────────────────────────────────────────
    import mlflow
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    tcfg, lcfg = cfg["training"], cfg["lora"]

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
        token=hf_token,
    )
    # Not prepare_model_for_kbit_training: it upcasts embeddings / lm_head / norms to fp32
    # (+~2 GB), which overflows the 8 GB card into shared memory and makes every step
    # ~3-4x slower (alignment_plan C7). bf16 weights + checkpointing are enough here.
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    init_adapter = cfg.get("init_adapter")
    if init_adapter:
        # Continue from an existing adapter: weights only, fresh optimizer and schedule
        from peft import PeftModel
        if not os.path.exists(os.path.join(init_adapter, "adapter_model.safetensors")):
            sys.exit(f"ERROR: init_adapter {init_adapter} has no adapter_model.safetensors")
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        print(f"Warm start from {init_adapter}")
    else:
        model = get_peft_model(model, LoraConfig(
            r=lcfg["r"],
            lora_alpha=lcfg["alpha"],
            lora_dropout=lcfg["dropout"],
            target_modules=lcfg["target_modules"],
            task_type="CAUSAL_LM",
        ))
    model.print_trainable_parameters()

    # First answer token decides Found vs None (used for the validation decision metric)
    found_id = tokenizer(POSITIVE_PREFIX.strip(), add_special_tokens=False)["input_ids"][0]
    none_id = tokenizer(NEGATIVE_ANSWER, add_special_tokens=False)["input_ids"][0]
    val_prompts = [
        (encode_prompt(tokenizer, s["clause_type"], s["text"]), 1 if s["answers"] else 0, s["clause_type"])
        for s in val_subset
    ]

    # Realistic-prevalence validation set (weighted): the checkpoint metric when the training mix
    # is not the real mix. Read from the val chunks only; never touches test.
    vcfg = cfg["validation"]
    realistic_items, realistic_prompts = [], []
    if vcfg.get("realistic"):
        r = vcfg["realistic"]
        with open(os.path.join(cfg["data"]["dir"], "chunks_val.json"), "r", encoding="utf-8") as f:
            val_chunks = json.load(f)
        from prepare_data import CLAUSE_TYPES
        realistic_items = select_realistic_val(val_chunks, CLAUSE_TYPES, r["pos_per_clause"],
                                               r["hard_per_clause"], r["easy_per_clause"], tcfg["seed"])
        realistic_prompts = [encode_prompt(tokenizer, it["clause_type"], it["text"]) for it in realistic_items]
        print(f"Realistic validation: {len(realistic_items)} items "
              f"({sum(it['label'] for it in realistic_items)} positive), weighted to the real mix")

    class DecisionEvalTrainer(Trainer):
        """Adds decision macro-F1 on the val subset to every evaluation (C5)."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if tcfg.get("answer_only_logits") and model.training:
                from answer_loss import answer_only_loss  # same loss, ~1 GB less GPU memory (src/answer_loss.py)
                return answer_only_loss(model, inputs, num_items_in_batch, return_outputs)
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            was_training = self.model.training
            self.model.eval()
            y_true, y_pred, clauses = [], [], []
            with torch.no_grad():
                for ids, label, ct in val_prompts:
                    logits = self.model(input_ids=torch.tensor([ids], device=self.model.device),
                                        logits_to_keep=1).logits[0, -1]
                    y_pred.append(int(logits[found_id] > logits[none_id]))
                    y_true.append(label)
                    clauses.append(ct)
                margins = []
                for ids in realistic_prompts:
                    logits = self.model(input_ids=torch.tensor([ids], device=self.model.device),
                                        logits_to_keep=1).logits[0, -1]
                    margins.append(float(logits[found_id] - logits[none_id]))
            if was_training:
                self.model.train()
            extra = {f"{metric_key_prefix}_{k}": v for k, v in decision_metrics(y_true, y_pred, clauses).items()}
            if realistic_items:
                extra.update({f"{metric_key_prefix}_{k}": v
                              for k, v in realistic_ap_metrics(realistic_items, margins).items()})
            self.log(extra)
            metrics.update(extra)
            # Give the evaluation's cached GPU blocks back. Without this the 8 GB card is full when
            # training resumes, Windows spills to shared memory and steps go from ~10 s to ~180 s.
            gc.collect()
            torch.cuda.empty_cache()
            return metrics

    class SlowStepGuard(TrainerCallback):
        """Stop (exit code 75) if steps become >6x slower than normal for 3 steps in a row, so the
        orchestrator can restart from the last checkpoint instead of crawling for hours."""

        def __init__(self):
            self.times, self.last, self.slow = [], None, 0

        def on_step_end(self, args, state, control, **kwargs):
            now = time.time()
            if self.last is not None:
                dt = now - self.last
                if len(self.times) >= 5 and dt > 6 * sorted(self.times)[len(self.times) // 2] and dt > 45:
                    self.slow += 1
                    if self.slow >= 3:
                        try:
                            q = subprocess.run(["nvidia-smi", "-q", "-d", "POWER"], capture_output=True, text=True).stdout
                            print("GPU power at the time:", [l.strip() for l in q.splitlines() if "Power Limit" in l or "Instantaneous" in l][:3], flush=True)
                        except OSError:
                            pass
                        print(f"SLOW STEPS: {dt:.0f} s/step vs median {sorted(self.times)[len(self.times) // 2]:.0f} s "
                              f"-> exiting so the run can restart from the last checkpoint", flush=True)
                        os._exit(75)
                else:
                    self.slow = 0
                    if dt < 60:
                        self.times.append(dt)
            self.last = now

        def on_evaluate(self, args, state, control, **kwargs):
            self.last = None  # evaluation time is not step time

    class MLflowCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            for k, v in (logs or {}).items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=state.global_step)

    steps_per_epoch = math.ceil(len(train_examples) / (tcfg["per_device_batch_size"] * tcfg["grad_accum"]))
    total_steps = steps_per_epoch * tcfg["num_epochs"]
    max_steps = tcfg.get("max_steps", -1)  # > 0 only for smoke tests
    if max_steps > 0:
        total_steps = max_steps

    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=tcfg["seed"],
        data_seed=tcfg["seed"],
        num_train_epochs=tcfg["num_epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=1,
        # Only the eval LOSS is needed from the Trainer (decision F1 / AP are computed in evaluate() below).
        # Without this the Trainer keeps (samples x tokens x 128k-vocab) logits in GPU memory, which spills.
        prediction_loss_only=True,
        gradient_accumulation_steps=tcfg["grad_accum"],
        learning_rate=float(tcfg["learning_rate"]),
        warmup_steps=max(1, int(tcfg["warmup_ratio"] * total_steps)),
        lr_scheduler_type=tcfg["lr_scheduler"],
        optim=tcfg["optimizer"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model=cfg["validation"]["metric_for_best_model"],
        greater_is_better=True,
        logging_steps=tcfg["logging_steps"],
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = DecisionEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_examples),
        eval_dataset=Dataset.from_list(val_examples),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        processing_class=tokenizer,
        callbacks=[MLflowCallback(), SlowStepGuard()],
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    provenance = {
        "config_file":     args.config,
        "dataset_version": manifest["dataset_version"],
        "train_sha256":    manifest["files"].get(cfg["data"]["train_file"]),
        "val_sha256":      manifest["files"].get(cfg["data"]["val_file"]),
        "steps_per_epoch": steps_per_epoch,
        "total_steps":     total_steps,
        **git_state(),
    }

    if args.start_eval_only:
        start_file = os.path.join(output_dir, "start_metrics.json")
        metrics = trainer.evaluate()
        os.makedirs(output_dir, exist_ok=True)
        json.dump({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, open(start_file, "w", encoding="utf-8"))
        print(f"Starting adapter scored -> {start_file}: realistic_ap = {metrics.get('eval_realistic_ap')}")
        return

    mlflow.set_experiment(cfg.get("mlflow_experiment", "LexProbe-FineTuning"))
    run_name = cfg["run_name"] + ("-resumed" if args.resume else "")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten(cfg))
        mlflow.log_params(provenance)
        mlflow.log_metrics({f"train_tokens_{k}": v for k, v in train_report.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(args.config, artifact_path="config")

        print(f"\nTraining {total_steps} steps ({steps_per_epoch}/epoch), eval + checkpoint every "
              f"{tcfg['eval_steps']} steps...")
        start_metrics = {}
        start_file = os.path.join(output_dir, "start_metrics.json")
        if init_adapter and vcfg.get("eval_at_start") and os.path.exists(start_file):
            start_metrics = json.load(open(start_file, encoding="utf-8"))  # measured in a separate process
            print(f"Starting adapter: { {k: round(v, 4) for k, v in start_metrics.items() if k.startswith('eval_realistic_ap') or k == 'eval_decision_macro_f1'} }")
        trainer.train(resume_from_checkpoint=True if args.resume else None)

        trainer.model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        summary = {
            **provenance,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "metric_for_best_model": cfg["validation"]["metric_for_best_model"],
            "train_token_report": train_report,
            "val_token_report": val_report,
            "start_metric": start_metrics.get("eval_" + cfg["validation"]["metric_for_best_model"]),
            "init_adapter": init_adapter,
        }
        sm, bm = summary["start_metric"], summary["best_metric"]
        if sm is not None and bm is not None:
            summary["improved_over_start"] = bm > sm
            print(f"{cfg['validation']['metric_for_best_model']}: start {sm:.4f} -> best {bm:.4f} "
                  f"({'improved' if bm > sm else 'NOT improved: apply the stop rule'})")
        with open(os.path.join(final_dir, "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(final_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        mlflow.log_metric("best_" + cfg["validation"]["metric_for_best_model"], trainer.state.best_metric or 0.0)
        mlflow.log_artifacts(final_dir, artifact_path="lora_adapter")

    print(f"\nBest checkpoint: {trainer.state.best_model_checkpoint} "
          f"({cfg['validation']['metric_for_best_model']} = {trainer.state.best_metric})")
    print(f"Adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
