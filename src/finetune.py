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

import argparse
import json
import math
import os
import subprocess
import sys

import yaml
from dotenv import load_dotenv

from training_data import decision_metrics, select_val_subset, tokenize_split
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
    if manifest["dataset_version"] != expected:
        sys.exit(f"ERROR: dataset version is {manifest['dataset_version']}, config expects {expected}.")

    with open(os.path.join(data_dir, cfg["data"]["train_file"]), "r", encoding="utf-8") as f:
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

    class DecisionEvalTrainer(Trainer):
        """Adds decision macro-F1 on the val subset to every evaluation (C5)."""

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
            if was_training:
                self.model.train()
            extra = {f"{metric_key_prefix}_{k}": v for k, v in decision_metrics(y_true, y_pred, clauses).items()}
            self.log(extra)
            metrics.update(extra)
            return metrics

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
        callbacks=[MLflowCallback()],
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

    mlflow.set_experiment(cfg.get("mlflow_experiment", "LexProbe-FineTuning"))
    run_name = cfg["run_name"] + ("-resumed" if args.resume else "")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten(cfg))
        mlflow.log_params(provenance)
        mlflow.log_metrics({f"train_tokens_{k}": v for k, v in train_report.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(args.config, artifact_path="config")

        print(f"\nTraining {total_steps} steps ({steps_per_epoch}/epoch), eval + checkpoint every "
              f"{tcfg['eval_steps']} steps...")
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
        }
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
