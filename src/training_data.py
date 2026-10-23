"""
Tokenisation and validation helpers for QLoRA fine-tuning (alignment_plan C1, C3, C5).

Kept free of torch / model imports so the logic can be unit-tested with a fake tokenizer.
"""

import random
from collections import Counter

from prompting import END_OF_TURN, encode_prompt, format_target

IGNORE_INDEX = -100  # label value ignored by the cross-entropy loss


def build_example(tokenizer, sample: dict, max_seq_length: int, max_answer_chars: int,
                  multi_span: str = "first") -> dict | None:
    """
    One training example with completion-only labels (C3): prompt tokens are masked with
    IGNORE_INDEX so the loss covers only the answer ("Found: <quote>" or "None") + end-of-turn.

    Returns None if the full sequence would exceed max_seq_length. The context is never
    truncated: the gold clause can sit anywhere in the chunk, so cutting the context could
    silently turn a positive into an unanswerable sample (B1/C1).
    """
    prompt_ids = encode_prompt(tokenizer, sample["clause_type"], sample["text"])
    target = format_target(sample["answers"], max_answer_chars, multi_span) + END_OF_TURN
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_seq_length:
        return None
    return {
        "input_ids":      input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels":         [IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def tokenize_split(tokenizer, samples: list[dict], max_seq_length: int, max_answer_chars: int,
                   multi_span: str = "first") -> tuple[list[dict], dict]:
    """Tokenise a split and report the token-length distribution and anything dropped."""
    examples, lengths = [], []
    dropped = Counter()
    trimmed = 0

    for s in samples:
        if s["answers"]:
            full = s["answers"][0]["text"] if multi_span == "first" else " ... ".join(a["text"] for a in s["answers"])
            trimmed += len(full) > max_answer_chars
        ex = build_example(tokenizer, s, max_seq_length, max_answer_chars, multi_span)
        if ex is None:
            dropped["positive" if s["answers"] else "negative"] += 1
            continue
        examples.append(ex)
        lengths.append(len(ex["input_ids"]))

    lengths.sort()
    n = len(samples)
    report = {
        "samples":            n,
        "kept":               len(examples),
        "dropped_too_long":   sum(dropped.values()),
        "dropped_positive":   dropped["positive"],
        "dropped_negative":   dropped["negative"],
        "dropped_fraction":   (sum(dropped.values()) / n) if n else 0.0,
        "answers_trimmed":    trimmed,
        "tokens_p50":         _percentile(lengths, 0.50),
        "tokens_p95":         _percentile(lengths, 0.95),
        "tokens_max":         lengths[-1] if lengths else 0,
        "max_seq_length":     max_seq_length,
    }
    return examples, report


def select_val_subset(samples: list[dict], max_samples: int, seed: int) -> list[dict]:
    """Fixed, seeded subset of val used for periodic evaluation during training."""
    ordered = sorted(samples, key=lambda s: s.get("query_id", ""))
    random.Random(f"{seed}-val-subset").shuffle(ordered)
    return ordered[:max_samples]


def decision_metrics(y_true: list[int], y_pred: list[int], clause_types: list[str]) -> dict:
    """
    Per-clause F1 and macro F1 for the Found(1) vs None(0) decision.
    Clause types with no positives in the subset are skipped: their F1 is undefined,
    and counting them as 0 would distort the macro average and checkpoint selection.
    """
    metrics = {}
    f1s = []
    for ct in sorted(set(clause_types)):
        idx = [i for i, c in enumerate(clause_types) if c == ct]
        if not any(y_true[i] == 1 for i in idx):
            continue
        tp = sum(1 for i in idx if y_true[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in idx if y_true[i] == 0 and y_pred[i] == 1)
        fn = sum(1 for i in idx if y_true[i] == 1 and y_pred[i] == 0)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        safe = ct.replace(" ", "_").replace("/", "_")
        metrics[f"decision_f1_{safe}"] = f1
        f1s.append(f1)
    metrics["decision_macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    return metrics


# ── Realistic-prevalence validation (v5 checkpoint selection) ──────────────────
def select_realistic_val(chunks: list[dict], clause_types: list[str], pos_per_clause: int,
                         hard_per_clause: int, easy_per_clause: int, seed: int) -> list[dict]:
    """
    A small validation set whose METRIC behaves like the real mix (about 1.5% positives).

    We cannot score every val chunk during training (too slow), so we sample and give each item
    a weight: weight = (stratum size) / (stratum sampled). Strata per clause are positives,
    near-miss negatives and easy negatives. A weighted metric on this sample then estimates the
    metric on ALL val chunks, which a 1:2 enriched sample cannot (it overstates precision).
    """
    from train_mix import is_near_miss  # local import: keeps this module light for tests

    items = []
    for ct in clause_types:
        pos = sorted((c for c in chunks if c["labels"][ct]["status"] == "positive"), key=lambda c: c["chunk_id"])
        neg = [c for c in chunks if c["labels"][ct]["status"] == "negative"]
        hard = sorted((c for c in neg if is_near_miss(c, ct)), key=lambda c: c["chunk_id"])
        hard_ids = {c["chunk_id"] for c in hard}
        easy = sorted((c for c in neg if c["chunk_id"] not in hard_ids), key=lambda c: c["chunk_id"])
        rng = random.Random(f"{seed}-realistic-val-{ct}")
        for pool, n, label in ((pos, pos_per_clause, 1), (hard, hard_per_clause, 0), (easy, easy_per_clause, 0)):
            take = rng.sample(pool, min(n, len(pool)))
            for c in take:
                items.append({"clause_type": ct, "text": c["text"], "label": label,
                              "weight": len(pool) / len(take), "chunk_id": c["chunk_id"]})
    return items


def weighted_average_precision(y_true: list[int], scores: list[float], weights: list[float]) -> float:
    """Average precision where each item counts `weight` times. Ties are broken by input order."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = seen = 0.0
    total_pos = sum(w for y, w in zip(y_true, weights) if y == 1)
    if total_pos == 0:
        return 0.0
    ap = 0.0
    for i in order:
        seen += weights[i]
        if y_true[i] == 1:
            tp += weights[i]
            ap += weights[i] * (tp / seen)
    return ap / total_pos


def realistic_ap_metrics(items: list[dict], margins: list[float]) -> dict:
    """Per-clause weighted AP and their macro average, from the margins of `items`."""
    out, aps = {}, []
    for ct in sorted({it["clause_type"] for it in items}):
        idx = [i for i, it in enumerate(items) if it["clause_type"] == ct]
        if not any(items[i]["label"] == 1 for i in idx):
            continue
        ap = weighted_average_precision([items[i]["label"] for i in idx], [margins[i] for i in idx],
                                        [items[i]["weight"] for i in idx])
        out[f"realistic_ap_{ct.replace(' ', '_')}"] = ap
        aps.append(ap)
    out["realistic_ap"] = sum(aps) / len(aps) if aps else 0.0
    return out


def verify_train_mix(mix_manifest: dict, dataset_manifest: dict) -> None:
    """Refuse a training mix whose frozen inputs differ from the dataset the run evaluates against."""
    frozen = mix_manifest["frozen_files_sha256"]
    for name in ("val.json", "test.json", "chunks_val.json", "chunks_test.json", "chunks_train.json"):
        if frozen.get(name) != dataset_manifest["files"].get(name):
            raise ValueError(f"train mix was built against a different {name} than the dataset in use")
