"""
Prompt format and training-example invariants (alignment_plan C1, C2, C3, C4, C5, B5).
Uses a tiny fake tokenizer that mimics the Llama-3.1 chat template — no model download.
"""

import glob
import os
import re

import pytest

from prompting import END_OF_TURN, build_messages, build_prompt, encode_prompt, format_target, split_prompt
from training_data import (
    IGNORE_INDEX,
    build_example,
    decision_metrics,
    select_val_subset,
    tokenize_split,
)

SPECIAL = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
BOS_ID = 1000


class FakeTokenizer:
    """Characters -> ord(c); special tokens -> 1000+. Template mirrors Llama-3.1's structure."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, date_string=None):
        assert tokenize is False
        out = "<|begin_of_text|>"
        for m in messages:
            out += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content'].strip()}<|eot_id|>"
        if add_generation_prompt:
            out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return out

    def __call__(self, text, add_special_tokens=True):
        ids = [BOS_ID] if add_special_tokens else []
        for part in re.split("(" + "|".join(map(re.escape, SPECIAL)) + ")", text):
            if part in SPECIAL:
                ids.append(1000 + SPECIAL.index(part))
            else:
                ids.extend(ord(c) for c in part)
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(SPECIAL[i - 1000] if i >= 1000 else chr(i) for i in ids)


@pytest.fixture
def tok():
    return FakeTokenizer()


def _sample(text="The laws of Delaware govern this Agreement.", answers=None, clause="Governing Law"):
    return {"clause_type": clause, "text": text, "answers": answers or [], "query_id": "q"}


# ── prompting (C2 / C4) ────────────────────────────────────────────────────────
def test_prompt_has_exactly_one_bos(tok):
    ids = encode_prompt(tok, "Governing Law", "some contract text")
    assert ids.count(BOS_ID) == 1 and ids[0] == BOS_ID


def test_prompt_contains_whole_chunk_unsliced():
    text = "x" * 5000 + " END-OF-CHUNK"
    user = build_messages("Non-Compete", text)[1]["content"]
    assert text in user


def test_split_prompt_prefix_is_shared_and_lossless(tok):
    chunk = "Some excerpt. Question: Does this excerpt mention a trick? More text."
    parts = [split_prompt(tok, ct, chunk) for ct in ("Governing Law", "Non-Compete", "Auto-Renewal")]
    assert len({p for p, _ in parts}) == 1                      # prefix identical across clause types
    for ct, (prefix, suffix) in zip(("Governing Law", "Non-Compete", "Auto-Renewal"), parts):
        assert prefix + suffix == build_prompt(tok, ct, chunk)  # nothing lost or reordered
        assert ct in suffix and ct not in prefix
        assert suffix.startswith("Question: Does this excerpt contain")  # split at the real question, not the quoted one


def test_no_hand_written_templates_left_in_src():
    """Every src/ script must build prompts through prompting.py (single source of truth)."""
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    offenders = [p for p in glob.glob(os.path.join(src, "*.py"))
                 if "<|begin_of_text|>" in open(p, encoding="utf-8").read()]
    assert offenders == []


# ── targets (B5 / C1) ──────────────────────────────────────────────────────────
def test_negative_target_is_none():
    assert format_target([], 600) == "None"


def test_long_answer_trimmed_to_verbatim_word_prefix():
    quote = "Neither party shall " + "compete with the other " * 60
    target = format_target([{"text": quote}], max_answer_chars=100)
    body = target.removeprefix("Found: ")
    assert len(body) <= 100 and quote.startswith(body) and not body.endswith(" ")


def test_multi_span_modes():
    answers = [{"text": "first span"}, {"text": "second span"}]
    assert format_target(answers, 600, "first") == "Found: first span"
    assert format_target(answers, 600, "join") == "Found: first span ... second span"
    with pytest.raises(ValueError):
        format_target(answers, 600, "bogus")


# ── examples (C1 / C3) ─────────────────────────────────────────────────────────
def test_loss_only_on_answer_tokens(tok):
    s = _sample(answers=[{"text": "The laws of Delaware govern this Agreement."}])
    ex = build_example(tok, s, max_seq_length=10_000, max_answer_chars=600)
    prompt_len = len(encode_prompt(tok, s["clause_type"], s["text"]))

    assert ex["labels"][:prompt_len] == [IGNORE_INDEX] * prompt_len
    learned = [t for t in ex["labels"] if t != IGNORE_INDEX]
    assert tok.decode(learned) == "Found: The laws of Delaware govern this Agreement." + END_OF_TURN


def test_negative_example_learns_none_and_stop_token(tok):
    ex = build_example(tok, _sample(), max_seq_length=10_000, max_answer_chars=600)
    learned = [t for t in ex["labels"] if t != IGNORE_INDEX]
    assert tok.decode(learned) == "None" + END_OF_TURN


def test_too_long_example_is_dropped_not_truncated(tok):
    s = _sample(text="word " * 500)
    assert build_example(tok, s, max_seq_length=200, max_answer_chars=600) is None


def test_tokenize_split_reports_drops(tok):
    samples = [_sample(), _sample(text="word " * 500, answers=[{"text": "word"}])]
    examples, report = tokenize_split(tok, samples, max_seq_length=1500, max_answer_chars=600)
    assert len(examples) == 1
    assert report["dropped_too_long"] == 1 and report["dropped_positive"] == 1
    assert report["dropped_fraction"] == 0.5


# ── validation (C5) ────────────────────────────────────────────────────────────
def test_val_subset_is_deterministic_and_order_independent():
    samples = [{"query_id": f"q{i:03d}"} for i in range(100)]
    a = select_val_subset(samples, 20, seed=42)
    b = select_val_subset(list(reversed(samples)), 20, seed=42)
    assert [s["query_id"] for s in a] == [s["query_id"] for s in b]
    assert len(a) == 20


def test_decision_metrics_macro_f1():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 1, 0]
    clauses = ["A", "A", "A", "A", "B", "B"]
    m = decision_metrics(y_true, y_pred, clauses)
    assert m["decision_f1_A"] == pytest.approx(0.5)   # tp=1 fp=1 fn=1
    assert m["decision_f1_B"] == pytest.approx(1.0)
    assert m["decision_macro_f1"] == pytest.approx(0.75)


def test_decision_metrics_skip_clauses_without_positives():
    m = decision_metrics([1, 0, 0, 0], [1, 0, 0, 0], ["A", "A", "B", "B"])
    assert "decision_f1_B" not in m
    assert m["decision_macro_f1"] == pytest.approx(1.0)
