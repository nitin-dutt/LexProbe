"""
Hand-written synthetic Liquidated Damages examples (training only), written by a stronger model (Claude) instead of
the local 8B generator, whose output was rejected in two external reviews (src/synthetic_ld.py, alignment_plan.md).

Source files: data/synthetic/hand/*.txt, blocks of
    === POS | <kind> | <contract type>      the quote is wrapped in [[[ ]]] inside the excerpt
    === NEG | <kind> | <contract type>      near-miss topics with NO agreed-damages provision
This script checks every block and writes data/synthetic/hand_v1.json in the training-row format
(same loader as train_mix.load_synthetic). Nothing here needs a GPU.

Checks (a block that fails is reported and left out, never silently fixed):
  - exactly one marked quote per POS, none in NEG; the quote is a verbatim slice of the excerpt;
  - no template artefacts: [PLACEHOLDER], ____, REDACTED, [***], TBD;
  - POS quote starts and ends like a sentence, has a trigger word and a liquidated-damages style keyword;
  - NEG has no liquidated-damages style wording at all;
  - near-duplicate (5-word shingles) of any real chunk or of another hand-written block is refused;
  - length windows as in synthetic_ld.
"""

import argparse
import glob
import json
import os
import random
import re
from collections import Counter

from prepare_data import DATA_DIR, _shingles
from synthetic_ld import filter_candidate, make_row, parse_marked, PHRASE

SRC_GLOB = "data/synthetic/hand/*.txt"
OUT = "data/synthetic/hand_v1.json"
ARTEFACTS = re.compile(r"\[[A-Za-z*][A-Za-z *./_-]{0,30}\]|_{2,}|REDACTED|\bTBD\b|\[\*+\]", re.I)
NEG_BANNED = re.compile(r"liquidated|penalt|termination fee|cancellation charge|termination charge|termination payment|as damages|pre-?estimate|deposit.{0,40}(retain|forfeit)", re.I)
HEADER = re.compile(r"^=== (POS|NEG) \| ([\w-]+) \| (.+)$")


def parse_blocks(text: str, source: str) -> list[dict]:
    blocks, cur = [], None
    for line in text.splitlines():
        m = HEADER.match(line.strip())
        if m:
            cur = {"label": m.group(1), "kind": m.group(2), "type": m.group(3).strip(), "lines": [], "source": source}
            blocks.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
    for b in blocks:
        b["text"] = "\n".join(b.pop("lines")).strip()
    return blocks


def check_block(b: dict, seen: list, real: list) -> tuple[bool, str, tuple | None, str]:
    """(ok, reason, answer span or None, final text)"""
    raw = b["text"]
    if b["label"] == "POS":
        parsed = parse_marked(raw)
        if parsed is None:
            return False, "markers", None, raw
        text, a, e = parsed
        span = (a, e)
    else:
        if "[[[" in raw or "]]]" in raw:
            return False, "markers_in_negative", None, raw
        text, span = raw, None
    if ARTEFACTS.search(text):
        return False, "template_artifact", span, text
    if b["label"] == "NEG" and NEG_BANNED.search(text):
        return False, "negative_has_ld_wording", span, text
    keep, reason = filter_candidate(text, span, b["label"] == "POS", seen, real, require_keyword=False)
    return keep, reason, span, text


def load_real_shingles() -> list[set]:
    out = []
    for sp in ("train", "val", "test"):
        out += [_shingles(c["text"]) for c in json.load(open(os.path.join(DATA_DIR, f"chunks_{sp}.json"), encoding="utf-8"))]
    return out


def build(paths: list[str]) -> tuple[list[dict], list[tuple], Counter]:
    real = load_real_shingles()
    blocks = []
    for p in paths:
        blocks += parse_blocks(open(p, encoding="utf-8").read(), os.path.basename(p))
    rows, rejected, seen, kinds = [], [], [], Counter()
    for i, b in enumerate(blocks, 1):
        ok, reason, span, text = check_block(b, seen, real)
        tag = f"{b['source']}#{i} {b['label']} {b['kind']}"
        if not ok:
            rejected.append((tag, reason))
            continue
        seen.append(_shingles(text))
        spec = {"type": b["type"], "style": "hand-written", "trigger": b["kind"], "position": "-"}
        row = make_row(len(rows) + 1, text, span, spec)
        row["query_id"] = row["chunk_id"] = f"syn::hand::{len(rows) + 1:04d}"
        row["author"] = "hand-written by Claude (Anthropic); training only"
        rows.append(row)
        kinds[(b["label"], b["kind"])] += 1
    return rows, rejected, kinds


TAIL_GUARD = re.compile(r"remed|damages|liab|reimburs|refund|credit|set-?off|deduct|offset|sole|exclusive|in addition|compensat|penalt|forfeit|retain|\bpay|reduc|waiv|may elect|shall not", re.I)


def roughen(rows: list[dict], seed: int = 7, p_start: float = 0.9, p_end: float = 0.6, min_chars: int = 420) -> list[dict]:
    """
    Real chunks are cut at arbitrary character positions, so many start or end mid-sentence; hand-written excerpts start
    at a clean heading. Trim the start (and end) of a random subset at word boundaries, independent of the label, so
    "starts cleanly" says nothing about positive vs negative. The marked quote is never touched; offsets are recomputed.
    """
    out = []
    for r in rows:
        rng = random.Random(f"{seed}-{r['query_id']}")
        text = r["text"]
        a = r["answers"][0] if r["answers"] else None
        lo = (a["start"] - 15) if a else int(len(text) * 0.4)   # latest start cut (stay clear of the quote)
        hi = (a["end"] + 15) if a else int(len(text) * 0.6)     # earliest end cut
        cut_s, cut_e = 0, len(text)
        if rng.random() < p_start and lo > 20:
            c = rng.randint(10, lo)
            cut_s = text.find(" ", c) + 1 if text.find(" ", c) != -1 else 0
        if rng.random() < p_end and len(text) - hi > 20:
            c = rng.randint(hi, len(text) - 5)
            cut_e = text.rfind(" ", hi, c) if text.rfind(" ", hi, c) != -1 else len(text)
        if cut_e < len(text) and TAIL_GUARD.search(text[cut_e:]):
            cut_e = len(text)  # never cut off remedy/qualifier wording: it could change the label
        new = text[cut_s:cut_e].rstrip()
        if len(new) < min_chars or (a and not (cut_s <= a["start"] and a["end"] <= cut_s + len(new))):
            new, cut_s = text, 0
        r = {**r, "text": new}
        if a:
            r["answers"] = [{**a, "start": a["start"] - cut_s, "end": a["end"] - cut_s}]
            assert new[r["answers"][0]["start"]:r["answers"][0]["end"]] == a["text"]
        out.append(r)
    return out


def cue_report(rows: list[dict]) -> dict:
    pos = [r["answers"][0]["text"] for r in rows if r["answers"]]
    if not pos:
        return {}
    share = lambda p: round(sum(1 for q in pos if re.search(p, q, re.I)) / len(pos), 3)  # noqa: E731
    return {"positives": len(pos), "negatives": sum(1 for r in rows if not r["answers"]),
            "says_liquidated": share("liquidated"), "penalty": share("penalt"), "dollar": share(r"\$\s?\d"), "percent": share(r"\d\s?(percent|%)"),
            "per_period": share(r"per (day|week|month)|for each (day|week|month)"), "pre_estimate": round(sum(1 for r in rows if r["answers"] and PHRASE.search(r["text"])) / len(pos), 3),
            "schedule_deferral": share(r"schedule|order form|project plan|attachment"), "median_quote_chars": sorted(len(q) for q in pos)[len(pos) // 2]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    rows, rejected, kinds = build(sorted(glob.glob(SRC_GLOB)))
    rows = roughen(rows)
    for tag, reason in rejected:
        print(f"REJECTED {tag}: {reason}")
    json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {args.out}: {len(rows)} rows ({len(rejected)} rejected)")
    print("cues:", cue_report(rows))
    print("kinds:", dict(kinds))


if __name__ == "__main__":
    main()
