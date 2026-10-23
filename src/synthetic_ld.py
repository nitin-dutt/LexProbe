"""
Synthetic Liquidated Damages (LD) training data, written by the local Llama-3.1-8B-Instruct (base model,
adapter OFF), for TRAINING ONLY. Real CUAD has just 81 LD positives in train.

What it makes
  - positives: a ~1,000-1,500 character contract excerpt that contains one LD provision; the provision is
    wrapped in [[[ ]]] by the generator, so the evidence quote is a verbatim slice by construction;
  - hard negatives: excerpts built from neighbouring topics (indemnity, warranty remedies, termination,
    payment terms ...) that fix no agreed amount of damages -> label "None".

Guards (all enforced in code, see filter_candidate / load_synthetic in train_mix.py)
  - answer is a verbatim slice of the text; markers never leak into the text;
  - LD keyword + amount check on positives; no "liquidated damages" in negatives;
  - near-duplicate filter against every real chunk (train/val/test) and against earlier synthetic rows;
  - style examples are real LD clauses from the TRAIN split only.
Val/test are never used as style examples and synthetic rows are never added to them.

Usage (GPU; ask first, ~20-40 min):
  python src/synthetic_ld.py --n-pos 300 --n-neg 120
Outputs: data/synthetic/ld_v1.json (rows), data/synthetic/ld_v1_stats.json, data/synthetic/ld_v1_review.md
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

from prepare_data import SEED, _shingles

OUT_DIR = "data/synthetic"
CLAUSE = "Liquidated Damages"

CONTRACT_TYPES = [
    "software license agreement", "master services agreement", "supply and distribution agreement",
    "construction subcontract", "equipment lease", "franchise agreement", "manufacturing agreement",
    "consulting agreement", "reseller agreement", "co-marketing agreement", "joint venture agreement",
    "maintenance and support agreement", "outsourcing agreement", "licensing and royalty agreement",
    "hosting services agreement", "development and OEM agreement",
]
# A trigger and an amount are only combined when they make sense together: delay-type triggers take per-period or
# rate amounts; one-off triggers (early termination, breach of exclusivity ...) take one-time or remaining-term amounts.
DELAY_TRIGGERS = [
    "late delivery of the products or deliverables", "failure to meet the agreed completion date", "failure to achieve the service levels",
    "delay in commissioning or go-live", "failure to obtain required approvals on time",
]
ONEOFF_TRIGGERS = [
    "early termination by the customer without cause", "breach of the exclusivity undertaking", "failure to purchase the minimum committed quantity",
    "unauthorised disclosure of confidential information", "breach of the non-solicitation undertaking",
]
TRIGGERS = DELAY_TRIGGERS + ONEOFF_TRIGGERS
AMOUNTS_NUMERIC = {
    "delay": ["a fixed sum of {n} dollars (${amt}) for each day of delay", "{p} percent of the monthly fee for each week of delay",
              "${amt} for each calendar month in which the failure continues", "{p} percent of the total contract price, calculated per full week of delay"],
    "oneoff": ["a one-time amount of ${amt}", "${amt} per breach, payable within thirty days of demand",
               "{p} percent of the total contract price as a single payment"],
}
AMOUNTS_PLAIN = {  # real LD quotes rarely state a figure (9% have a $ amount)
    "delay": ["an amount equal to the fees payable for the period of delay", "compensation calculated at the rate stated in the service level schedule",
              "an agreed percentage of the unpaid fees, as set out in the schedule", "the sum specified in the applicable order form or schedule"],
    "oneoff": ["the sum specified in the applicable order form or schedule", "the agreed damages set out in the schedule of charges",
               "the fixed sum stated in the project plan", "damages equal to the minimum commitment for the remainder of the term",
               "an agreed percentage of the total contract price, as set out in the schedule"],
}
CAPS = ["subject to a maximum of {p} percent of the total fees", "with no cap other than the contract price", "capped at ${amt} in aggregate", ""]
NEIGHBOURS = [
    "payment terms with invoicing and a thirty day payment period", "a confidentiality obligation", "an indemnity for third-party claims",
    "a warranty with repair or replace remedies", "termination for material breach after a cure period", "a force majeure clause",
    "a limitation of liability for indirect damages", "notices and how they are given", "assignment and change of control",
    "intellectual property ownership of deliverables", "audit and record keeping rights", "insurance requirements",
]
NEGATIVE_THEMES = [  # topics close to LD that fix NO agreed amount of damages
    "an indemnity for third-party claims", "a warranty with repair or replace remedies", "termination for material breach after a cure period",
    "a limitation of liability for indirect damages", "payment terms with invoicing and a thirty day payment period", "milestone acceptance and sign-off procedures",
    "an obligation to use reasonable efforts to meet delivery dates", "injunctive relief and specific performance", "an obligation to cure defects", "a notice and dispute escalation procedure",
]
STYLES = ["formal legal drafting with numbered sections", "plain-English drafting with numbered sections", "dense, long sentences typical of older contracts"]

MIN_CHARS, MAX_CHARS = 450, 1800  # real chunks are at most 1,500 chars; stay close so length is not a giveaway
LD_KEYWORDS = re.compile(r"sole compensation|full settlement|compensation for (the )?delay|for each (full |complete )?(day|week|month)|reasonable (estimate|forecast)|termination (fee|charge|payment)|cancellation charge|retain the deposit|shortfall|liquidated|per (day|week|month)|penalt|agreed (sum|amount|damages)|fixed (sum|charge|damages)|pre-?estimat|as damages|for each (day|week|month)|agreed percentage|damages (equal|calculated)", re.I)
TRIGGER_WORDS = re.compile(r"delay|late|fail|breach|terminat|disclos|exclusiv|solicit|purchase|approv|service level|commission|go-live|non-?compliance|cancel|default|abandon|violat|prepa|accelerat|shortfall|fewer|below|less than", re.I)
HAS_AMOUNT = re.compile(r"\$\s?\d|\d\s?(percent|%)|\bdollars\b|fees payable", re.I)


# ── prompts ───────────────────────────────────────────────────────────────────
def pick_spec(rng: random.Random, positive: bool) -> dict:
    amt = rng.choice([500, 1000, 1500, 2500, 5000, 10000, 25000, 50000])
    kind = rng.choice(["delay", "oneoff"])
    trigger = rng.choice(DELAY_TRIGGERS if kind == "delay" else ONEOFF_TRIGGERS)
    numeric = rng.random() < 0.25  # real train LD quotes: 9% have a $ figure, 12% a percent, 3% per-day/week, 46% say "liquidated"
    amount = (rng.choice(AMOUNTS_NUMERIC[kind]).format(n=amt, amt=f"{amt:,}", p=rng.choice([1, 2, 5, 10])) if numeric
              else rng.choice(AMOUNTS_PLAIN[kind]))
    return {
        "type": rng.choice(CONTRACT_TYPES), "style": rng.choice(STYLES), "positive": positive, "kind": kind,
        "trigger": trigger, "amount": amount, "use_phrase": rng.random() < 0.5,
        "cap": rng.choice(CAPS).format(p=rng.choice([10, 15, 20, 25]), amt=f"{amt * 10:,}") if kind == "delay" else "",
        "neighbours": rng.sample(NEGATIVE_THEMES if not positive else NEIGHBOURS, 3),
        "preestimate": rng.random() < 0.3,  # most real clauses do not say "pre-estimate"; avoid a shortcut phrase
        "position": rng.choice(["first", "middle", "last"]), "words": rng.choice([140, 160, 180, 200]),
    }


def build_messages(spec: dict, style_examples: list[str]) -> list[dict]:
    system = ("You write realistic synthetic contract excerpts to train a clause classifier. "
              "Output only the excerpt, with no commentary.")
    ex = "\n".join(f"Example {i + 1} (style only, never copy): {e}" for i, e in enumerate(style_examples))
    if spec["positive"]:
        body = (f"Write about {spec['words']} words (hard limit: 1,400 characters) from a {spec['type']}, in {spec['style']}. It has 4 short numbered sections, including "
                f"{', '.join(spec['neighbours'])}. One section (the {spec['position']} one) is a liquidated damages provision: "
                f"the parties agree that on {spec['trigger']} one party pays {spec['amount']}"
                f"{', ' + spec['cap'] if spec['cap'] else ''}"
                + (", described as a genuine pre-estimate of loss rather than a penalty. " if spec["preestimate"]
                   else ". State the obligation plainly; do not use the words 'pre-estimate' or 'penalty'. ")
                + ("" if spec["use_phrase"] else "Do not use the words 'liquidated damages' in the provision; call it agreed damages, a fixed charge or similar. ")
                + "The provision itself must be short: one to three sentences. "
                + 
                f"Wrap exactly the full text of that one provision in [[[ and ]]] (use the markers once only).\n\n{ex}")
    else:
        body = (f"Write about {spec['words']} words (hard limit: 1,400 characters) from a {spec['type']}, in {spec['style']}. It has 4 short numbered sections covering "
                f"{', '.join(spec['neighbours'])}. None of the sections may state a pre-agreed sum, rate or percentage payable as damages for "
                f"breach or delay, and the words 'liquidated damages' must not appear. Do not use any [[[ ]]] markers.")
    return [{"role": "system", "content": system}, {"role": "user", "content": body}]


# ── parsing and filters (pure, unit-tested) ────────────────────────────────────
def parse_marked(raw: str):
    """(text, answer_start, answer_end) from a generation with exactly one [[[ ... ]]] pair, or None."""
    if raw.count("[[[") != 1 or raw.count("]]]") != 1:
        return None
    a, b = raw.index("[[["), raw.index("]]]")
    if b < a:
        return None
    before, inner, after = raw[:a], raw[a + 3:b].strip(), raw[b + 3:]
    if not inner:
        return None
    text = before + inner + after
    start = len(before)
    lead = len(text) - len(text.lstrip())  # drop outer leading whitespace; offsets move with it
    text = text.strip()
    start -= lead
    return text, start, start + len(inner)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def filter_candidate(text: str, answer: tuple | None, positive: bool, seen: list[set], real: list[set],
                     jac: float = 0.5, require_keyword: bool = True) -> tuple[bool, str]:
    """(keep, reason). `seen` = shingles of accepted synthetic texts, `real` = shingles of real chunks."""
    if len(text) < MIN_CHARS:
        return False, "length_short"
    if len(text) > MAX_CHARS:
        return False, "length_long"
    if "[[[" in text or "]]]" in text:
        return False, "marker_leak"
    if positive:
        if answer is None:
            return False, "no_markers"
        quote = text[answer[0]:answer[1]]
        if not 80 <= len(quote) <= 700:
            return False, "answer_length"
        if require_keyword and not LD_KEYWORDS.search(quote):
            return False, "not_ld_like"  # a heuristic for the 8B generator; hand-written blocks are vouched for by their author
        if not re.match(r"""^[A-Z0-9"“(]""", quote) or not re.search(r"""[.;"”)]$""", quote):
            return False, "quote_fragment"  # must start like a sentence and end like one (reviewer: "[ The agreed fixed charge ...")
        if not TRIGGER_WORDS.search(quote):
            return False, "quote_without_trigger"
    else:
        if re.search(r"liquidated damages", text, re.I):
            return False, "negative_mentions_ld"
        if re.search(r"service credits?", text, re.I):
            return False, "negative_service_credit"  # too close to liquidated damages to label reliably
        if HAS_AMOUNT.search(text) and re.search(r"per (day|week)|penalt|as damages", text, re.I):
            return False, "negative_looks_like_ld"
    sh = _shingles(text)
    if any(jaccard(sh, s) >= jac for s in seen):
        return False, "duplicate_synthetic"
    if any(jaccard(sh, s) >= jac for s in real):
        return False, "duplicate_real"
    return True, "ok"


PHRASE = re.compile(r"pre-?estimate", re.I)


def cap_phrase_share(rows: list[dict], max_share: float = 0.35, seed: int = 1) -> list[dict]:
    """Keep every positive without the shortcut phrase and only as many with it as max_share allows. Negatives untouched."""
    pos = [r for r in rows if r["answers"]]
    neg = [r for r in rows if not r["answers"]]
    plain = [r for r in pos if not PHRASE.search(r["text"])]
    phrased = [r for r in pos if PHRASE.search(r["text"])]
    keep = int(max_share * len(plain) / (1 - max_share)) if plain else 0
    phrased = random.Random(seed).sample(phrased, min(len(phrased), keep))
    kept = sorted(plain + phrased + neg, key=lambda r: r["query_id"])
    return kept


# Shortcut-cue features and their ceilings, set from the REAL train LD quotes (9% $ figure, 12% percent, 3% per-period,
# 1% "pre-estimate", 46% "liquidated") with some headroom. A synthetic set that over-uses a cue teaches the cue.
FEATURES = {
    "dollar": (re.compile(r"\$\s?\d"), 0.20), "percent": (re.compile(r"\d\s?(percent|%)", re.I), 0.20),
    "per_period": (re.compile(r"per (day|week|month)|for each (day|week|month)", re.I), 0.08), "pre_estimate": (PHRASE, 0.10),
}


def select_matched(rows: list[dict], seed: int = 1, warmup: int = 15) -> tuple[list[dict], dict]:
    """
    Greedy, seeded: add a positive only if no cue's share of the kept positives would exceed its ceiling
    (the first `warmup` positives are exempt so shares are not judged on tiny counts). Negatives are kept as they are.
    Returns (rows, report).
    """
    pos = [r for r in rows if r["answers"]]
    neg = [r for r in rows if not r["answers"]]
    random.Random(seed).shuffle(pos)
    kept, counts = [], {k: 0 for k in FEATURES}
    for r in pos:
        quote = r["answers"][0]["text"]
        hit = {k: bool(rx.search(quote if k != "pre_estimate" else r["text"])) for k, (rx, _) in FEATURES.items()}
        n = len(kept) + 1
        if n > warmup and any((counts[k] + hit[k]) / n > FEATURES[k][1] for k in FEATURES if hit[k]):
            continue
        kept.append(r)
        for k in FEATURES:
            counts[k] += hit[k]
    report = {"positives_in": len(pos), "positives_kept": len(kept), "negatives": len(neg),
              "shares": {k: round(counts[k] / max(1, len(kept)), 3) for k in FEATURES}}
    return sorted(kept + neg, key=lambda r: r["query_id"]), report


def judge_messages(positive: bool, text: str, quote: str | None, trigger: str) -> list[dict]:
    """A yes/no coherence check put to the same model. Positives must be a coherent, complete LD provision for the intended
    trigger; negatives must contain nothing that fixes an amount in advance (this also removes service-credit-like text)."""
    if positive:
        q = (f"Provision: {quote}\n\nQuestion: Does this provision clearly fix, in advance, the compensation payable by one party if there is "
             f"{trigger}, and is it coherent and complete as drafted? Answer Yes or No.")
    else:
        q = (f"Contract excerpt:\n{text}\n\nQuestion: Does any provision in this excerpt fix in advance an amount, rate, penalty or service credit "
             f"payable for breach or delay? Answer Yes or No.")
    return [{"role": "system", "content": "You are a careful contract lawyer. Answer with a single word: Yes or No."}, {"role": "user", "content": q}]


def yes_no_margin(model, tok, messages, yes_id: int, no_id: int) -> float:
    import torch
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    with torch.no_grad():
        logits = model(**ids, logits_to_keep=1).logits[0, -1]
    return float(logits[yes_id] - logits[no_id])


PLACEHOLDER = re.compile(r"\[[A-Z][A-Za-z /_-]{2,30}\]")


def surface_bin(r: dict) -> tuple:
    """Format features that must not predict the label: has a [PLACEHOLDER], starts with a numbered heading, length band."""
    n = len(r["text"])
    return (bool(PLACEHOLDER.search(r["text"])), r["text"].lstrip().startswith("1."), 0 if n < 1100 else 1 if n < 1400 else 2)


FILLERS = {
    "DATE": ["March 14, 2022", "June 1, 2023", "September 30, 2021", "January 12, 2024", "November 5, 2020"],
    "PARTY A": ["Northwind Ltd."], "PARTY B": ["Harbor Systems Inc."], "LICENSOR": ["Meridian Software LLC"], "LICENSEE": ["Brightline Corp."],
    "STATE": ["Delaware", "New York", "California", "Texas"], "COUNTRY": ["United States"], "ADDRESS": ["100 Market Street"],
    "COMPANY": ["Apex Industries"], "CUSTOMER": ["Brightline Corp."], "SUPPLIER": ["Meridian Software LLC"],
}


def fill_placeholders(rows: list[dict]) -> tuple[list[dict], int]:
    """Replace [DATE]-style template markers with realistic names/dates in text AND quote (real chunks have none; the markers
    were also much more common in negatives than positives). Rows whose quote cannot be re-located are dropped."""
    out, dropped = [], 0
    for r in rows:
        def sub(m, qid=r["query_id"]):
            key = m.group(0)[1:-1].upper().strip()
            opts = FILLERS.get(key, ["the Company"])
            # same placeholder -> same filler within a row, so the quote still matches the text
            return opts[0] if len(opts) == 1 else random.Random(f"{qid}-{key}").choice(opts)
        text = PLACEHOLDER.sub(sub, r["text"])
        if r["answers"]:
            quote = PLACEHOLDER.sub(sub, r["answers"][0]["text"])
            start = text.find(quote)
            if start < 0:
                dropped += 1
                continue
            r = {**r, "text": text, "answers": [{"start": start, "end": start + len(quote), "text": quote, "clipped": False}],
                 "explanation": f"Found: {quote}"}
        else:
            r = {**r, "text": text}
        out.append(r)
    return out, dropped


def balance_surface(rows: list[dict], seed: int = 1, ratio: float = 1.0) -> tuple[list[dict], dict]:
    """
    Keep all positives; per (numbered-start x length-band) bin keep at most `ratio` x as many negatives as there are positives,
    so formatting alone says little about the label. A bin with too few negatives keeps all it has (reported, not hidden).
    """
    pos = [r for r in rows if r["answers"]]
    neg = [r for r in rows if not r["answers"]]
    key = lambda r: surface_bin(r)[1:]  # noqa: E731  (numbered start, length band)
    pb, nb = {}, {}
    for r in pos:
        pb.setdefault(key(r), []).append(r)
    for r in neg:
        nb.setdefault(key(r), []).append(r)
    rng = random.Random(seed)
    kept = []
    short = {}
    for b, v in pb.items():
        want = int(round(ratio * len(v)))
        have = nb.get(b, [])
        kept += rng.sample(have, min(want, len(have)))
        if len(have) < want:
            short[str(b)] = {"positives": len(v), "negatives_available": len(have)}
    rate = lambda xs, f: round(sum(f(r) for r in xs) / max(1, len(xs)), 3)  # noqa: E731
    report = {"positives": len(pos), "negatives_in": len(neg), "negatives_kept": len(kept), "bins_short_of_negatives": short,
              "numbered_start_rate": {"pos": rate(pos, lambda r: surface_bin(r)[1]), "neg": rate(kept, lambda r: surface_bin(r)[1])},
              "length_band_rates": {f"band{b}": {"pos": rate(pos, lambda r, b=b: surface_bin(r)[2] == b), "neg": rate(kept, lambda r, b=b: surface_bin(r)[2] == b)} for b in range(3)}}
    return sorted(pos + kept, key=lambda r: r["query_id"]), report


def make_row(idx: int, text: str, answer: tuple | None, spec: dict) -> dict:
    answers = [] if answer is None else [{"start": answer[0], "end": answer[1], "text": text[answer[0]:answer[1]], "clipped": False}]
    qid = f"syn::ld::{idx:04d}"
    return {"query_id": qid, "chunk_id": qid, "contract_title": "synthetic", "clause_type": CLAUSE, "text": text,
            "answers": answers, "explanation": f"Found: {answers[0]['text']}" if answers else "None",
            "hard_negative": not answers, "synthetic": True, "gen_meta": {k: spec[k] for k in ("type", "style", "trigger", "position")}}


# ── generation (GPU) ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", metavar="IN_JSON", help="CPU only: keep a real-like subset of an existing generation (see select_matched) and exit")
    ap.add_argument("--n-pos", type=int, default=300)
    ap.add_argument("--n-neg", type=int, default=120)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--max-attempts-factor", type=float, default=3.0, help="stop after this many generations per wanted row")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "ld_v1.json"))
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--no-judge", action="store_true", help="skip the model's own yes/no coherence check on each accepted candidate")
    args = ap.parse_args()

    if args.select:
        rows, report = select_matched(json.load(open(args.select, encoding="utf-8")))
        rows, dropped = fill_placeholders(rows)
        rows, balance = balance_surface(rows)
        report["placeholders_filled_rows_dropped"] = dropped
        report["surface_balance"] = balance
        json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        json.dump(report, open(args.out.replace(".json", "_selection.json"), "w"), indent=2)
        print(f"wrote {args.out}: {report}")
        return

    import torch
    from dotenv import load_dotenv
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    load_dotenv()
    import prepare_data
    d = prepare_data.DATA_DIR
    real_chunks = [c for sp in ("train", "val", "test") for c in json.load(open(os.path.join(d, f"chunks_{sp}.json"), encoding="utf-8"))]
    real = [_shingles(c["text"]) for c in real_chunks]
    style_pool = [a["text"] for c in json.load(open(os.path.join(d, "chunks_train.json"), encoding="utf-8"))  # TRAIN split only
                  for a in c["labels"][CLAUSE]["answers"] if 100 <= len(a["text"]) <= 600]
    print(f"{len(real)} real chunks for the duplicate filter, {len(style_pool)} real LD clauses as style pool (train only)")

    rng = random.Random(f"{SEED}-synthetic-ld")
    token = os.environ["HF_TOKEN"]
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token=token)
    tok.pad_token = "<|finetune_right_pad_id|>"; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", token=token, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
    model.eval()
    yes_id = tok("Yes", add_special_tokens=False)["input_ids"][0]
    no_id = tok("No", add_special_tokens=False)["input_ids"][0]

    rows, seen, stats = [], [], Counter()
    partial = args.out.replace(".json", ".partial.json")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(partial):
        saved = json.load(open(partial, encoding="utf-8"))
        rows, stats = saved["rows"], Counter(saved["stats"])
        seen = [_shingles(r["text"]) for r in rows]
        rng = random.Random(f"{SEED}-synthetic-ld-{len(rows)}-{stats['attempts']}")  # fresh but reproducible stream on resume
        print(f"resuming with {len(rows)} accepted rows")
    want = {"pos": args.n_pos, "neg": args.n_neg}
    have = lambda kind: sum(1 for r in rows if bool(r["answers"]) == (kind == "pos"))  # noqa: E731
    max_attempts = int((args.n_pos + args.n_neg) * args.max_attempts_factor)
    t0 = time.time()
    while (have("pos") < want["pos"] or have("neg") < want["neg"]) and stats["attempts"] < max_attempts:
        specs = []
        for _ in range(args.batch):
            kind_pos = have("pos") + sum(1 for s in specs if s["positive"]) < want["pos"] and (rng.random() < 0.7 or have("neg") >= want["neg"])
            specs.append(pick_spec(rng, kind_pos))
        prompts = [tok.apply_chat_template(build_messages(s, rng.sample(style_pool, 2)), tokenize=False, add_generation_prompt=True) for s in specs]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=520, do_sample=True, temperature=args.temperature, top_p=0.95, pad_token_id=tok.pad_token_id)
        for spec, ids in zip(specs, out):
            stats["attempts"] += 1
            raw = tok.decode(ids[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if spec["positive"]:
                parsed = parse_marked(raw)
                text, ans = (parsed[0], (parsed[1], parsed[2])) if parsed else (raw, None)
            else:
                text, ans = raw, None
            keep, reason = filter_candidate(text, ans, spec["positive"], seen, real)
            if keep and not args.no_judge:
                quote = text[ans[0]:ans[1]] if ans else None
                m = yes_no_margin(model, tok, judge_messages(spec["positive"], text, quote, spec["trigger"]), yes_id, no_id)
                if (spec["positive"] and m <= 0) or (not spec["positive"] and m >= 0):
                    keep, reason = False, "judge_no" if spec["positive"] else "judge_yes"
            stats[("pos_" if spec["positive"] else "neg_") + reason] += 1
            if keep and have("pos" if spec["positive"] else "neg") < want["pos" if spec["positive"] else "neg"]:
                rows.append(make_row(len(rows) + 1, text, ans, spec))
                seen.append(_shingles(text))
        json.dump({"rows": rows, "stats": dict(stats)}, open(partial, "w", encoding="utf-8"))
        print(f"[{time.time() - t0:5.0f}s] attempts {stats['attempts']} | accepted pos {have('pos')}/{want['pos']} neg {have('neg')}/{want['neg']}", flush=True)

    # final checks use the same loader the training mix uses
    from train_mix import load_synthetic
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    before = len(rows)
    rows = cap_phrase_share(rows)
    print(f"phrase-share cap: kept {len(rows)} of {before} rows ({sum(1 for r in rows if r['answers'] and PHRASE.search(r['text']))} positives still contain 'pre-estimate')")
    json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    load_synthetic(args.out, {c["text"] for c in real_chunks})  # raises if any row breaks the rules
    json.dump({"rows": len(rows), "positives": have("pos"), "negatives": have("neg"), "stats": dict(stats),
               "seconds": round(time.time() - t0), "temperature": args.temperature}, open(args.out.replace(".json", "_stats.json"), "w"), indent=2)
    review = [f"# Synthetic LD review sample (read before training)\n", f"{have('pos')} positives, {have('neg')} negatives. 10 random of each:\n"]
    for kind, label in (("pos", "POSITIVE"), ("neg", "NEGATIVE")):
        pool = [r for r in rows if bool(r["answers"]) == (kind == "pos")]
        for r in random.Random(1).sample(pool, min(10, len(pool))):
            q = r["answers"][0]["text"] if r["answers"] else "(none)"
            review.append(f"## {label} {r['query_id']} ({r['gen_meta']['type']})\n\n{r['text']}\n\n**quote:** {q}\n")
    open(args.out.replace(".json", "_review.md"), "w", encoding="utf-8").write("\n".join(review))
    print(f"wrote {args.out}: {len(rows)} rows; stats {dict(stats)}")


if __name__ == "__main__":
    main()
