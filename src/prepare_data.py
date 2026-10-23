"""
Leak-free data preparation for LexProbe (dataset v3).

Pipeline:
  1. Normalise whitespace in every contract, mapping CUAD answer offsets onto the normalised text.
  2. Group near-duplicate contracts (copies / re-filings of the same agreement) and split
     70/15/15 by GROUP, so neither a contract nor its copy can appear in two splits.
  3. Chunk each contract on natural boundaries into windows that fit the prompt budget.
     The stored chunk text is exactly what goes into the prompt — nothing is truncated later.
  4. Label every (chunk, clause) pair:
       positive — a gold span lies fully inside the chunk
       partial  — a gold span is cut by the chunk edge (excluded from query sets)
       negative — no gold span touches the chunk
     A span longer than any chunk is assigned, clipped, to the chunk that holds most of it.
  5. Write, per split:
       chunks_<split>.json — every chunk with labels for every clause (for document-level eval)
       <split>.json        — sampled (chunk, clause) queries for training / chunk-level eval
     Each split samples negatives with its OWN seeded RNG, and val/test use a fixed ratio,
     so changing the training sampling never changes the evaluation sets.
  6. Write manifest.json: config, label map, split membership, counts and file hashes.

Refuses to overwrite an existing dataset unless --force is given.

Usage:
  python src/prepare_data.py
  python src/prepare_data.py --force --train-neg-per-pos 2.0
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import zlib
from collections import Counter, defaultdict

from chunking import chunk_text, normalize_text

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_VERSION = "v3"
SEED            = 42
RAW_FILE        = "data/raw/CUAD_v1.json"
OUTPUT_DIR      = "data/processed"

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15}  # test gets the remainder

# Chunk size is bounded by the training token budget (~4 chars/token for legal text).
# 1500 chars ≈ 375 tokens of context; revisit together with MAX_SEQ_LENGTH (alignment_plan C1).
CHUNK_CHARS   = 1500
OVERLAP_CHARS = 300

# Contracts whose opening text overlaps this much are treated as one group (alignment_plan A8)
NEAR_DUP_JACCARD   = 0.5
NEAR_DUP_PREFIX    = 20_000  # chars of each contract compared
NEAR_DUP_SHINGLE_N = 8       # words per shingle

# Exact CUAD category -> LexProbe label (alignment_plan B2/B3/B4).
# Deliberately excluded: "Uncapped Liability" (opposite of a cap), "Competitive Restriction
# Exception" (an exception, not a restriction) and "Notice Period To Terminate Renewal".
CLAUSE_MAP = {
    "Governing Law":               "Governing Law",
    "Renewal Term":                "Auto-Renewal",
    "Liquidated Damages":          "Liquidated Damages",
    "Cap On Liability":            "Limitation of Liability",
    "Non-Compete":                 "Non-Compete",
    "Termination For Convenience": "Termination for Convenience",
}
CLAUSE_TYPES = sorted(set(CLAUSE_MAP.values()))

# Dataset profiles. v3 (6 clauses) is what the deployed model was built on and stays the default.
# v4 adds Anti-Assignment (374 CUAD contracts have it, 6x the data of Liquidated Damages). Splits
# depend only on contracts and near-duplicate groups, so v4 uses the SAME contracts per split as v3
# (tests/test_dataset_profiles.py). The profile is applied only by the prepare_data CLI.
PROFILES = {
    "v3": {"extra_map": {}, "output_dir": "data/processed", "eval_dir": "reports/eval"},
    "v4": {"extra_map": {"Anti-Assignment": "Anti-Assignment"}, "output_dir": "data/processed_v4", "eval_dir": "reports/eval_v4"},
}


def apply_profile(version: str) -> str:
    """Switch CLAUSE_MAP / CLAUSE_TYPES / DATASET_VERSION in place for this process; returns the output dir."""
    global DATASET_VERSION
    prof = PROFILES[version]
    CLAUSE_MAP.update(prof["extra_map"])
    CLAUSE_TYPES[:] = sorted(set(CLAUSE_MAP.values()))  # in place: other modules hold this same list
    DATASET_VERSION = version
    return prof["output_dir"]


# Every script imports this module, so one environment variable switches the whole toolchain
# (clause list, data folder, report folder): LEXPROBE_PROFILE=v4. Default v3 = the reported six clauses.
ACTIVE_PROFILE = os.environ.get("LEXPROBE_PROFILE", "v3")
if ACTIVE_PROFILE not in PROFILES:
    raise SystemExit(f"LEXPROBE_PROFILE={ACTIVE_PROFILE!r} is not one of {sorted(PROFILES)}")
if ACTIVE_PROFILE != "v3":
    apply_profile(ACTIVE_PROFILE)
DATA_DIR = PROFILES[ACTIVE_PROFILE]["output_dir"]
EVAL_DIR = PROFILES[ACTIVE_PROFILE]["eval_dir"]

# Negatives sampled per positive, per clause type. Half are "hard" negatives: chunks that
# contain a DIFFERENT target clause, which is where the model is most likely to confuse labels.
TRAIN_NEG_PER_POS   = 1.5  # tuned on val in Phase 2 (alignment_plan B6)
EVAL_NEG_PER_POS    = 2.0  # fixed — do not change without bumping DATASET_VERSION
HARD_NEG_FRACTION   = 0.5


# ── Loading & span mapping ─────────────────────────────────────────────────────
def extract_category(question: str) -> str:
    """CUAD questions quote the category name: 'Highlight the parts (if any) ... related to "X" ...'."""
    parts = question.split('"')
    return parts[1] if len(parts) >= 3 else ""


def load_contracts(path: str) -> tuple[list[dict], dict, Counter]:
    """
    Returns (contracts, category_questions, span_stats). Each contract:
      {title, text (normalised), spans: {label: [(start, end)]}}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)["data"]

    contracts = []
    category_questions = {}
    stats = Counter()

    for entry in raw:
        assert len(entry["paragraphs"]) == 1, "CUAD v1 has one paragraph per contract"
        para = entry["paragraphs"][0]
        text, offset_map = normalize_text(para["context"])
        spans = defaultdict(set)

        for qa in para["qas"]:
            category = extract_category(qa["question"])
            label = CLAUSE_MAP.get(category)
            if label is None:
                continue
            category_questions.setdefault(category, qa["question"])

            for ans in qa["answers"]:
                s = offset_map[ans["answer_start"]]
                e = offset_map[ans["answer_start"] + len(ans["text"])]
                # Trim whitespace left at span edges by normalisation
                while s < e and text[s].isspace():
                    s += 1
                while e > s and text[e - 1].isspace():
                    e -= 1
                if e <= s:
                    stats["empty_after_normalisation"] += 1
                    continue
                expected = " ".join(ans["text"].split())
                if " ".join(text[s:e].split()) != expected:
                    stats["mapping_mismatch"] += 1
                spans[label].add((s, e))
                stats["spans"] += 1

        contracts.append({
            "title": entry["title"],
            "text":  text,
            "spans": {label: sorted(v) for label, v in spans.items()},
        })

    return contracts, category_questions, stats


# ── Near-duplicate grouping ────────────────────────────────────────────────────
def _shingles(text: str) -> set[int]:
    words = re.findall(r"\w+", text[:NEAR_DUP_PREFIX].lower())
    n = NEAR_DUP_SHINGLE_N
    return {zlib.crc32(" ".join(words[i:i + n]).encode()) for i in range(max(1, len(words) - n + 1))}


def group_near_duplicates(contracts: list[dict]) -> tuple[list[int], list[dict]]:
    """Union-find over contracts whose shingle Jaccard >= NEAR_DUP_JACCARD. Returns (group_of, pairs)."""
    parent = list(range(len(contracts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sh = [_shingles(c["text"]) for c in contracts]
    pairs = []
    for i in range(len(contracts)):
        for j in range(i + 1, len(contracts)):
            a, b = sh[i], sh[j]
            inter = len(a & b)
            if not inter:
                continue
            jac = inter / len(a | b)
            if jac >= NEAR_DUP_JACCARD:
                parent[find(j)] = find(i)
                pairs.append({"a": contracts[i]["title"], "b": contracts[j]["title"], "jaccard": round(jac, 3)})

    return [find(i) for i in range(len(contracts))], pairs


def split_by_group(contracts: list[dict], group_of: list[int]) -> dict[str, list[dict]]:
    """Shuffle contracts with a dedicated RNG, then assign whole groups to splits in that order."""
    order = list(range(len(contracts)))
    random.Random(SEED).shuffle(order)

    groups: dict[int, list[int]] = {}
    for idx in order:
        groups.setdefault(group_of[idx], []).append(idx)

    n = len(contracts)
    train_cut = SPLIT_FRACTIONS["train"] * n
    val_cut = (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]) * n

    splits = {"train": [], "val": [], "test": []}
    assigned = 0
    for members in groups.values():
        name = "train" if assigned < train_cut else "val" if assigned < val_cut else "test"
        for idx in members:
            contracts[idx]["group_id"] = group_of[idx]
            splits[name].append(contracts[idx])
        assigned += len(members)
    return splits


# ── Chunking & labelling ───────────────────────────────────────────────────────
def build_chunks(contract: dict, stats: Counter) -> list[dict]:
    text = contract["text"]
    windows = chunk_text(text, CHUNK_CHARS, OVERLAP_CHARS)
    chunks = [{
        "chunk_id":       f"{contract['title']}::{i:04d}",
        "contract_title": contract["title"],
        "group_id":       contract["group_id"],
        "chunk_index":    i,
        "start":          s,
        "end":            e,
        "text":           text[s:e],
        "labels":         {label: {"status": "negative", "answers": []} for label in CLAUSE_TYPES},
    } for i, (s, e) in enumerate(windows)]

    for label, spans in contract["spans"].items():
        for (ss, se) in spans:
            containing = [c for c in chunks if c["start"] <= ss and se <= c["end"]]
            overlapping = [c for c in chunks if ss < c["end"] and se > c["start"]]

            if containing:
                targets = [(c, ss, se) for c in containing]
            else:
                # Span longer than a window: give it, clipped, to the chunk holding most of it
                best = max(overlapping, key=lambda c: min(se, c["end"]) - max(ss, c["start"]))
                targets = [(best, max(ss, best["start"]), min(se, best["end"]))]
                stats[f"clipped_spans::{label}"] += 1

            for c, s, e in targets:
                lab = c["labels"][label]
                lab["status"] = "positive"
                lab["answers"].append({
                    "start":   s - c["start"],
                    "end":     e - c["start"],
                    "text":    text[s:e],
                    "clipped": (s, e) != (ss, se),
                })

            target_ids = {id(c) for c, _, _ in targets}
            for c in overlapping:
                if id(c) not in target_ids and c["labels"][label]["status"] == "negative":
                    c["labels"][label]["status"] = "partial"

    for c in chunks:
        for label in CLAUSE_TYPES:
            lab = c["labels"][label]
            lab["answers"].sort(key=lambda a: a["start"])
            stats[f"{lab['status']}::{label}"] += 1
            # B1 invariant: every positive answer is inside the exact text the model will see
            for a in lab["answers"]:
                assert c["text"][a["start"]:a["end"]] == a["text"], c["chunk_id"]
    return chunks


# ── Query sampling ─────────────────────────────────────────────────────────────
def sample_queries(chunks: list[dict], rng: random.Random, neg_per_pos: float) -> list[dict]:
    """All positives + `neg_per_pos` negatives per positive (per clause type), half of them hard."""
    queries = []
    for label in CLAUSE_TYPES:
        pos = [c for c in chunks if c["labels"][label]["status"] == "positive"]
        neg = [c for c in chunks if c["labels"][label]["status"] == "negative"]
        hard = [c for c in neg if any(c["labels"][o]["status"] == "positive" for o in CLAUSE_TYPES if o != label)]
        hard_ids = {c["chunk_id"] for c in hard}
        easy = [c for c in neg if c["chunk_id"] not in hard_ids]

        n_neg = min(len(neg), round(neg_per_pos * len(pos)))
        n_hard = min(len(hard), round(n_neg * HARD_NEG_FRACTION))
        n_easy = min(len(easy), n_neg - n_hard)
        n_hard = min(len(hard), n_neg - n_easy)  # top up from hard if easy ran short

        chosen_neg = rng.sample(hard, n_hard) + rng.sample(easy, n_easy)

        for c in pos + chosen_neg:
            answers = c["labels"][label]["answers"]
            queries.append({
                "query_id":       f"{c['chunk_id']}::{label}",
                "chunk_id":       c["chunk_id"],
                "contract_title": c["contract_title"],
                "clause_type":    label,
                "text":           c["text"],
                "answers":        answers,
                # Kept for the existing training/eval scripts: first gold span, or "None"
                "explanation":    f"Found: {answers[0]['text']}" if answers else "None",
                "hard_negative":  (not answers) and c["chunk_id"] in hard_ids,
            })

    rng.shuffle(queries)
    return queries


# ── Output ─────────────────────────────────────────────────────────────────────
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def print_summary(split_contracts, split_chunks, split_queries):
    print("\n" + "=" * 92)
    print(f"{'Split':<7} {'Contracts':>9} {'Chunks':>7} {'Queries':>8} {'Pos':>6} {'Neg':>6}   per clause (pos/neg)")
    print("=" * 92)
    for name in ("train", "val", "test"):
        q = split_queries[name]
        pos = sum(1 for x in q if x["answers"])
        per = Counter((x["clause_type"], bool(x["answers"])) for x in q)
        detail = ", ".join(f"{ct.split()[0]} {per[(ct, True)]}/{per[(ct, False)]}" for ct in CLAUSE_TYPES)
        print(f"{name:<7} {len(split_contracts[name]):>9} {len(split_chunks[name]):>7} {len(q):>8} "
              f"{pos:>6} {len(q) - pos:>6}   {detail}")
    print("=" * 92)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing dataset in output_dir")
    parser.add_argument("--train-neg-per-pos", type=float, default=TRAIN_NEG_PER_POS)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v3",
                        help="v3 = the 6 reported clauses (default); v4 = adds Anti-Assignment, written to data/processed_v4")
    args = parser.parse_args()
    output_dir = apply_profile(args.profile)

    if not os.path.exists(RAW_FILE):
        sys.exit(f"Error: {RAW_FILE} not found. Run src/download_data.py first.")

    existing = [f for f in ("manifest.json", "train.json", "val.json", "test.json")
                if os.path.exists(os.path.join(output_dir, f))]
    if existing and not args.force:
        sys.exit(
            f"Refusing to overwrite existing dataset files in {output_dir}: {existing}.\n"
            "Evaluation sets must stay frozen once results are reported (alignment_plan A1).\n"
            "Re-run with --force only if you intend to create a new dataset version."
        )

    print(f"Loading and normalising {RAW_FILE}...")
    contracts, category_questions, stats = load_contracts(RAW_FILE)
    print(f"  {len(contracts)} contracts, {stats['spans']} target spans "
          f"({stats['mapping_mismatch']} offset-mapping mismatches)")

    print("Grouping near-duplicate contracts...")
    group_of, dup_pairs = group_near_duplicates(contracts)
    print(f"  {len(dup_pairs)} near-duplicate pairs -> {len(set(group_of))} groups")

    split_contracts = split_by_group(contracts, group_of)
    split_groups = {k: {c["group_id"] for c in v} for k, v in split_contracts.items()}
    assert not (split_groups["train"] & split_groups["val"]), "Group leakage: train/val"
    assert not (split_groups["train"] & split_groups["test"]), "Group leakage: train/test"
    assert not (split_groups["val"] & split_groups["test"]), "Group leakage: val/test"
    print("[OK] Train / val / test are disjoint at the near-duplicate-group level.")

    split_chunks, split_queries = {}, {}
    neg_ratio = {"train": args.train_neg_per_pos, "val": EVAL_NEG_PER_POS, "test": EVAL_NEG_PER_POS}
    for name in ("train", "val", "test"):
        split_chunks[name] = [ch for c in split_contracts[name] for ch in build_chunks(c, stats)]
        # Independent RNG per split: changing train sampling cannot alter val/test (alignment_plan A3)
        rng = random.Random(f"{SEED}-{name}-queries")
        split_queries[name] = sample_queries(split_chunks[name], rng, neg_ratio[name])

    os.makedirs(output_dir, exist_ok=True)
    files = {}
    for name in ("train", "val", "test"):
        for fname, obj in ((f"chunks_{name}.json", split_chunks[name]), (f"{name}.json", split_queries[name])):
            path = os.path.join(output_dir, fname)
            write_json(obj, path)
            files[fname] = sha256(path)

    manifest = {
        "dataset_version": DATASET_VERSION,
        "source": {"file": RAW_FILE, "sha256": sha256(RAW_FILE)},
        "config": {
            "seed": SEED,
            "split_fractions": SPLIT_FRACTIONS,
            "chunk_chars": CHUNK_CHARS,
            "overlap_chars": OVERLAP_CHARS,
            "near_dup": {"jaccard": NEAR_DUP_JACCARD, "prefix_chars": NEAR_DUP_PREFIX,
                         "shingle_words": NEAR_DUP_SHINGLE_N},
            "neg_per_pos": neg_ratio,
            "hard_neg_fraction": HARD_NEG_FRACTION,
        },
        "clause_map": CLAUSE_MAP,
        "cuad_category_questions": category_questions,
        "near_duplicate_pairs": dup_pairs,
        "splits": {name: sorted(c["title"] for c in split_contracts[name]) for name in split_contracts},
        "stats": dict(sorted(stats.items())),
        "files": files,
    }
    write_json(manifest, os.path.join(output_dir, "manifest.json"))

    print_summary(split_contracts, split_chunks, split_queries)
    clipped = sum(v for k, v in stats.items() if k.startswith("clipped_spans"))
    partial = sum(v for k, v in stats.items() if k.startswith("partial::"))
    print(f"Spans longer than a chunk (clipped): {clipped} | partial (chunk, clause) pairs excluded: {partial}")
    print(f"Wrote dataset {DATASET_VERSION} to {output_dir}/ (manifest.json records config + hashes).")
    print("Next: dvc add data/processed  (then commit the .dvc file)")


if __name__ == "__main__":
    main()
