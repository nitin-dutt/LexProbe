"""
Test-set evaluation: zero-shot baseline vs fine-tuned adapter (alignment_plan D2, D4).

Loads the 4-bit base model + adapter ONCE; the baseline is the same model with the
adapter disabled, so both systems see identical weights, prompts and inputs.

Pass 1 — decisions (fast): one forward pass per query; the decision is Found vs None from
         the first answer-token logits. Also records whether the unconstrained top token
         was "Found"/"None" at all (format compliance — matters for the zero-shot model).
Pass 2 — evidence (slower): for queries a system predicts positive, generate the quote
         after a forced "Found:" prefix and check it is grounded in the chunk text.

Writes reports/eval/<name>_test.json per system. Compare with src/compare_results.py.

Usage:
  python src/evaluate.py --adapter models/llama-3.1-8b-cuad-lora-v4/final_adapter
  python src/evaluate.py --adapter ... --skip-generation     # decisions only
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

from eval_metrics import clean_quote, grounding, per_clause_metrics, token_f1
from prompting import NEGATIVE_ANSWER, POSITIVE_PREFIX, encode_prompt

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "reports/eval"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--name", default="lora_v4")
    parser.add_argument("--split-file", default="data/processed/test.json")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N queries (smoke test)")
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    with open(args.split_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    if args.limit:
        queries = queries[: args.limit]
    with open("data/processed/manifest.json", "r", encoding="utf-8") as f:
        dataset_version = json.load(f)["dataset_version"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                               bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="auto", token=hf_token,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    found_id = tokenizer(POSITIVE_PREFIX.strip(), add_special_tokens=False)["input_ids"][0]
    none_id = tokenizer(NEGATIVE_ANSWER, add_special_tokens=False)["input_ids"][0]
    prefix_ids = tokenizer(POSITIVE_PREFIX.rstrip(), add_special_tokens=False)["input_ids"]
    eot_ids = [tokenizer.convert_tokens_to_ids("<|eot_id|>"), tokenizer.eos_token_id]

    systems = {"baseline": True, args.name: False}  # name -> adapter disabled?
    prompts = [encode_prompt(tokenizer, q["clause_type"], q["text"]) for q in queries]
    results = {name: [] for name in systems}

    def run(disabled, fn):
        if disabled:
            with model.disable_adapter():
                return fn()
        return fn()

    # ── Pass 1: decisions ──────────────────────────────────────────────────────
    for name, disabled in systems.items():
        t0 = time.time()

        def decide():
            out = []
            with torch.no_grad():
                for i, (q, ids) in enumerate(zip(queries, prompts)):
                    logits = model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1).logits[0, -1]
                    top = int(logits.argmax())
                    margin = float(logits[found_id] - logits[none_id])
                    out.append({
                        "query_id":       q["query_id"],
                        "contract_title": q["contract_title"],
                        "clause_type":    q["clause_type"],
                        "label":          1 if q["answers"] else 0,
                        "pred":           1 if margin > 0 else 0,
                        "margin":         margin,
                        "top_token_ok":   top in (found_id, none_id),
                    })
                    if (i + 1) % 200 == 0:
                        print(f"  [{name}] decisions {i + 1}/{len(queries)}", flush=True)
            return out

        results[name] = run(disabled, decide)
        m = per_clause_metrics(results[name])["macro"]
        print(f"[{name}] decisions done in {(time.time() - t0) / 60:.1f} min — macro P/R/F1 "
              f"{m['precision']:.3f}/{m['recall']:.3f}/{m['f1']:.3f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)

    def save():
        for name in systems:
            path = os.path.join(OUT_DIR, f"{name}_test.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"system": name, "adapter": None if systems[name] else args.adapter,
                           "dataset_version": dataset_version, "split_file": args.split_file,
                           "decision": "first-token logit Found vs None",
                           "records": results[name]}, f, indent=1)

    save()
    print(f"Decisions saved to {OUT_DIR}/", flush=True)
    if args.skip_generation:
        return

    # ── Pass 2: evidence for predicted positives ───────────────────────────────
    for name, disabled in systems.items():
        t0 = time.time()
        todo = [i for i, r in enumerate(results[name]) if r["pred"] == 1]

        def generate():
            with torch.no_grad():
                for n, i in enumerate(todo):
                    ids = prompts[i] + prefix_ids
                    inp = torch.tensor([ids], device="cuda")
                    out = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp),
                                         max_new_tokens=args.max_new_tokens, do_sample=False,
                                         eos_token_id=eot_ids, pad_token_id=tokenizer.eos_token_id)
                    quote = clean_quote(tokenizer.decode(out[0][len(ids):], skip_special_tokens=True))
                    rec = results[name][i]
                    rec["quote"] = quote
                    rec.update(grounding(quote, queries[i]["text"]))
                    if rec["label"] == 1:
                        rec["span_f1"] = max(token_f1(quote, a["text"]) for a in queries[i]["answers"])
                    if (n + 1) % 50 == 0:
                        print(f"  [{name}] evidence {n + 1}/{len(todo)}", flush=True)

        run(disabled, generate)
        print(f"[{name}] evidence done in {(time.time() - t0) / 60:.1f} min", flush=True)
        save()

    print("All done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
