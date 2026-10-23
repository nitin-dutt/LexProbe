"""
Build the v5 TRAINING mix from the frozen chunks (CPU only, no model needed).

Why: v4 trained on 1.5 negatives per positive, mostly easy ones, while real contracts have
41-204. This builds a harder, larger training set aimed at the weak clauses:
  - every real positive is kept; weak clauses are repeated (--oversample),
  - negatives are `neg_per_pos` per UNIQUE positive, mostly "hard": chunks that mention the
    clause's own keywords or that contain a neighbouring clause, but are labelled negative.

Nothing here touches val/test: it reads data/processed/chunks_train.json only and writes to a
separate folder (data/train_mixes/), so the DVC-pinned dataset v3 stays frozen. The manifest
records the hashes of the untouched val/test files to prove it.

Usage:
    python src/train_mix.py                       # writes data/train_mixes/v5/
    python src/train_mix.py --neg-per-pos 3
"""
import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter

import prepare_data
from prepare_data import CLAUSE_TYPES, SEED

PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/train_mixes/v5"

NEG_PER_POS = 4.0
HARD_FRACTION = 0.7
OVERSAMPLE = {"Liquidated Damages": 2, "Non-Compete": 2, "Auto-Renewal": 2}

# Words that make a chunk LOOK like it holds the clause. A labelled-negative chunk that matches
# is a near-miss, which is exactly what the model confuses (alignment_plan D7 error review).
KEYWORDS = {
    "Auto-Renewal":                r"renew|successive|evergreen|extend(ed)? for",
    "Governing Law":              r"governed by|governing law|laws of the state",
    "Limitation of Liability":    r"limitation of liability|in no event|consequential|aggregate liability",
    "Liquidated Damages":         r"liquidated damages|late (fee|charge|payment)|penalt|per day|interest at",
    "Non-Compete":                r"non-?compet|shall not (directly|indirectly)|compete with|exclusiv",
    "Termination for Convenience": r"terminate.{0,60}(convenience|without cause|for any reason)|days.{0,10}notice",
    "Anti-Assignment":            r"assign|transfer|delegate|change of control|successors? and assigns",
}
_KW = {k: re.compile(v, re.I | re.S) for k, v in KEYWORDS.items()}


def is_near_miss(chunk: dict, label: str) -> bool:
    """A labelled-negative chunk that looks like a positive: own keywords, or a neighbouring clause."""
    if _KW[label].search(chunk["text"]):
        return True
    return any(chunk["labels"][o]["status"] == "positive" for o in CLAUSE_TYPES if o != label)


def _query(c: dict, label: str, hard: bool) -> dict:
    answers = c["labels"][label]["answers"]
    return {
        "query_id":       f"{c['chunk_id']}::{label}",
        "chunk_id":       c["chunk_id"],
        "contract_title": c["contract_title"],
        "clause_type":    label,
        "text":           c["text"],
        "answers":        answers,
        "explanation":    f"Found: {answers[0]['text']}" if answers else "None",
        "hard_negative":  hard,
    }


def sample_mix(chunks: list[dict], rng: random.Random, neg_per_pos: float = NEG_PER_POS,
               hard_fraction: float = HARD_FRACTION, oversample: dict | None = None) -> list[dict]:
    oversample = OVERSAMPLE if oversample is None else oversample
    out = []
    for label in CLAUSE_TYPES:
        pos = [c for c in chunks if c["labels"][label]["status"] == "positive"]
        neg = [c for c in chunks if c["labels"][label]["status"] == "negative"]  # 'partial' is excluded
        hard = [c for c in neg if is_near_miss(c, label)]
        hard_ids = {c["chunk_id"] for c in hard}
        easy = [c for c in neg if c["chunk_id"] not in hard_ids]

        n_neg = min(len(neg), round(neg_per_pos * len(pos)))
        n_hard = min(len(hard), round(n_neg * hard_fraction))
        n_easy = min(len(easy), n_neg - n_hard)
        n_hard = min(len(hard), n_neg - n_easy)

        for c in pos:
            out.extend(_query(c, label, False) for _ in range(max(1, oversample.get(label, 1))))
        out.extend(_query(c, label, True) for c in rng.sample(hard, n_hard))
        out.extend(_query(c, label, False) for c in rng.sample(easy, n_easy))
    rng.shuffle(out)
    return out


def load_synthetic(path: str, real_texts: set[str]) -> list[dict]:
    """
    Synthetic training rows (training only). Each row must be a full query dict (same fields as real
    rows) with synthetic=true, and every answer must be a verbatim substring of its text, otherwise the
    whole file is refused: the evidence-grounding rule of this project applies to training data too.
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        if not r.get("synthetic"):
            raise ValueError(f"{r.get('query_id')}: rows from a synthetic file must set synthetic=true")
        if r["clause_type"] not in CLAUSE_TYPES:
            raise ValueError(f"{r['query_id']}: unknown clause {r['clause_type']}")
        for a in r["answers"]:
            if r["text"][a["start"]:a["end"]] != a["text"]:
                raise ValueError(f"{r['query_id']}: answer is not a verbatim slice of the text")
        if r["text"] in real_texts:
            raise ValueError(f"{r['query_id']}: text duplicates a real chunk (val/test leakage risk)")
    return rows


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neg-per-pos", type=float, default=NEG_PER_POS)
    ap.add_argument("--hard-fraction", type=float, default=HARD_FRACTION)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--profile", choices=sorted(prepare_data.PROFILES), default="v3",
                    help="dataset profile (clause set); v4 reads data/processed_v4")
    ap.add_argument("--synthetic", help="JSON file of synthetic rows to add (training only; see load_synthetic)")
    ap.add_argument("--oversample", default="", help="e.g. 'Liquidated Damages=2,Non-Compete=2' (default: built-in)")
    args = ap.parse_args()
    global PROCESSED_DIR
    PROCESSED_DIR = prepare_data.apply_profile(args.profile)
    if args.output_dir == OUTPUT_DIR and args.profile != "v3":
        args.output_dir = f"data/train_mixes/{args.profile}"

    out_file = os.path.join(args.output_dir, "train.json")
    if os.path.exists(out_file) and not args.force:
        raise SystemExit(f"{out_file} exists; use --force to rebuild.")

    with open(os.path.join(PROCESSED_DIR, "chunks_train.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    rng = random.Random(f"{SEED}-train-v5")
    oversample = dict((k.strip(), int(v)) for k, v in (kv.split("=") for kv in args.oversample.split(",") if kv)) or None
    queries = sample_mix(chunks, rng, args.neg_per_pos, args.hard_fraction, oversample)
    if args.synthetic:
        real = {c["text"] for sp in ("train", "val", "test")
                for c in json.load(open(os.path.join(PROCESSED_DIR, f"chunks_{sp}.json"), encoding="utf-8"))}
        extra = load_synthetic(args.synthetic, real)
        queries = queries + extra
        random.Random(f"{SEED}-train-synthetic-shuffle").shuffle(queries)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False)

    frozen = {n: sha256(os.path.join(PROCESSED_DIR, n))
              for n in ("val.json", "test.json", "chunks_val.json", "chunks_test.json", "chunks_train.json")}
    per = Counter((q["clause_type"], bool(q["answers"])) for q in queries)
    hard = Counter(q["clause_type"] for q in queries if q["hard_negative"])
    manifest = {
        "mix": "v5", "seed": SEED, "neg_per_pos": args.neg_per_pos, "hard_fraction": args.hard_fraction,
        "oversample": oversample or OVERSAMPLE, "profile": args.profile, "n_queries": len(queries),
        "synthetic_file": args.synthetic, "n_synthetic": sum(1 for q in queries if q.get("synthetic")),
        "per_clause": {ct: {"positive_rows": per[(ct, True)], "negative_rows": per[(ct, False)],
                            "hard_negative_rows": hard[ct]} for ct in CLAUSE_TYPES},
        "train_sha256": sha256(out_file),
        "frozen_files_sha256": frozen,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"{'clause':30} {'pos rows':>9} {'neg rows':>9} {'hard neg':>9}")
    for ct in CLAUSE_TYPES:
        m = manifest["per_clause"][ct]
        print(f"{ct:30} {m['positive_rows']:>9} {m['negative_rows']:>9} {m['hard_negative_rows']:>9}")
    pos = sum(1 for q in queries if q["answers"])
    print(f"total {len(queries)} rows ({pos} positive, {len(queries) - pos} negative) -> {out_file}")


if __name__ == "__main__":
    main()
