"""
Full-document scoring: every chunk of every contract, all 6 clause types (alignment_plan D1).

The chunk-level test set samples negatives (1 positive : 2 negatives). A deployed system sees
EVERY chunk of a contract, where ~98.5% of (chunk, clause) pairs are negative, so precision
measured on the sampled set is optimistic. This script scores the complete chunk set.

Speed: the prompt order is  system + excerpt  |  question(clause). The first part is identical
for all 6 clauses, so it is run once and its KV cache is reused for the 6 short questions
(~327 shared tokens vs ~47 per question). Tokenising the two halves separately was checked
to give exactly the same token ids as tokenising the full prompt for all 17,970 test pairs.

Writes reports/eval/doc_<system>_<split>.json = {chunk_id: {clause: margin}} where
margin = logit(Found) - logit(None). Analysis happens on the CPU in src/document_eval.py.
Resumable: progress is saved to a .partial file every 250 chunks.

Usage:
  python src/evaluate_documents.py --adapter models/llama-3.1-8b-cuad-lora-v4/final_adapter --verify 15
  python src/evaluate_documents.py --adapter models/llama-3.1-8b-cuad-lora-v4/final_adapter
  python src/evaluate_documents.py --adapter ... --split val --systems lora_v4
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

from prepare_data import CLAUSE_TYPES, DATA_DIR, EVAL_DIR
from prompting import NEGATIVE_ANSWER, POSITIVE_PREFIX, encode_prompt, split_prompt

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = EVAL_DIR
SAVE_EVERY = 250


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--name", default="lora_v4")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--systems", default="baseline,adapter", help="comma list of: baseline, adapter")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N chunks (smoke test)")
    parser.add_argument("--verify", type=int, default=0, help="Compare cached vs full-forward margins on N chunks, then exit")
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    with open(f"{DATA_DIR}/chunks_{args.split}.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if args.limit:
        chunks = chunks[: args.limit]
    with open(f"{DATA_DIR}/manifest.json", "r", encoding="utf-8") as f:
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

    def margin_of(logits):
        return float(logits[found_id] - logits[none_id])

    @torch.no_grad()
    def margins_cached(text: str) -> dict:
        """All 6 clause margins for one chunk: prefix once, then 6 short questions on a cache trimmed back each time."""
        prefix, _ = split_prompt(tokenizer, CLAUSE_TYPES[0], text)
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        out = model(input_ids=torch.tensor([prefix_ids], device="cuda"), use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        res = {}
        for ct in CLAUSE_TYPES:
            _, suffix = split_prompt(tokenizer, ct, text)
            sids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
            o = model(input_ids=torch.tensor([sids], device="cuda"), past_key_values=cache,
                      use_cache=True, logits_to_keep=1)
            res[ct] = margin_of(o.logits[0, -1])
            cache.crop(-len(sids))  # drop the question tokens, keep the shared excerpt
        return res

    @torch.no_grad()
    def margins_full(text: str) -> dict:
        res = {}
        for ct in CLAUSE_TYPES:
            ids = encode_prompt(tokenizer, ct, text)
            res[ct] = margin_of(model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1).logits[0, -1])
        return res

    # ── Verification: cached path must agree with the plain forward pass ───────────
    if args.verify:
        diffs, flips, t_full, t_cached = [], 0, 0.0, 0.0
        for ch in chunks[: args.verify]:
            torch.cuda.synchronize(); t = time.time(); full = margins_full(ch["text"])
            torch.cuda.synchronize(); t_full += time.time() - t; t = time.time()
            cached = margins_cached(ch["text"])
            torch.cuda.synchronize(); t_cached += time.time() - t
            for ct in CLAUSE_TYPES:
                diffs.append(abs(full[ct] - cached[ct]))
                flips += (full[ct] > 0) != (cached[ct] > 0)
        print(f"verified {len(diffs)} (chunk, clause) pairs | max |Δmargin| {max(diffs):.3f} | mean {sum(diffs) / len(diffs):.4f} "
              f"| decision flips {flips}")
        print(f"time per chunk: full {t_full / args.verify:.2f}s -> cached {t_cached / args.verify:.2f}s "
              f"({t_full / t_cached:.1f}x faster)")
        return

    # ── Full scoring ──────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    wanted = [s.strip() for s in args.systems.split(",")]
    systems = {}
    if "baseline" in wanted:
        systems["baseline"] = True       # name -> adapter disabled?
    if "adapter" in wanted:
        systems[args.name] = False

    for name, disabled in systems.items():
        final = os.path.join(OUT_DIR, f"doc_{name}_{args.split}.json")
        partial = final.replace(".json", ".partial.json")
        margins = {}
        if os.path.exists(partial):
            margins = json.load(open(partial, encoding="utf-8"))["margins"]
            print(f"[{name}] resuming: {len(margins)} chunks already scored", flush=True)

        t0, done0 = time.time(), len(margins)
        todo = [c for c in chunks if c["chunk_id"] not in margins]
        for i, ch in enumerate(todo, 1):
            if disabled:
                with model.disable_adapter():
                    margins[ch["chunk_id"]] = margins_cached(ch["text"])
            else:
                margins[ch["chunk_id"]] = margins_cached(ch["text"])
            if i % SAVE_EVERY == 0 or i == len(todo):
                with open(partial, "w", encoding="utf-8") as f:
                    json.dump({"margins": margins}, f)
                rate = i / (time.time() - t0)
                print(f"  [{name}] {len(margins)}/{len(chunks)} chunks | {rate:.2f} chunks/s | "
                      f"~{(len(todo) - i) / rate / 60:.0f} min left", flush=True)

        with open(final, "w", encoding="utf-8") as f:
            json.dump({"system": name, "adapter": None if disabled else args.adapter,
                       "dataset_version": dataset_version, "split": args.split, "n_chunks": len(chunks),
                       "decision": "first-token logit margin Found - None, prefix KV cache",
                       "elapsed_s": round(time.time() - t0, 1), "scored_this_run": len(todo) + 0 * done0,
                       "margins": margins}, f)
        if os.path.exists(partial):
            os.remove(partial)
        print(f"[{name}] done in {(time.time() - t0) / 60:.1f} min -> {final}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
