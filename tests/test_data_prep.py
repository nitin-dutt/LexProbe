"""
Invariants for the dataset build (alignment_plan A3, A8, B1, B2, B7).
Runs on small synthetic contracts — no CUAD download or GPU needed.
"""

import random
from collections import Counter

import pytest

import prepare_data as pd_
from chunking import chunk_text, normalize_text


# ── normalize_text ─────────────────────────────────────────────────────────────
def test_normalize_collapses_whitespace_and_keeps_paragraphs():
    text, _ = normalize_text("  EXHIBIT   10.6\n\n\n\n   THIS    AGREEMENT \n  is made.")
    assert text == "EXHIBIT 10.6\n\nTHIS AGREEMENT\nis made."


def test_normalize_offset_map_preserves_spans():
    raw = "Section 1.   This   Agreement shall be governed by\n   the laws of   Delaware.  "
    answer = "This   Agreement shall be governed by\n   the laws of   Delaware."
    start = raw.index(answer)
    text, m = normalize_text(raw)
    s, e = m[start], m[start + len(answer)]
    assert " ".join(text[s:e].split()) == " ".join(answer.split())


# ── chunk_text ─────────────────────────────────────────────────────────────────
def _paragraphs(n, length=230):
    return "\n\n".join(f"Paragraph {i}. " + "word " * (length // 5) for i in range(n))


def test_chunks_respect_max_size_and_cover_text():
    text = _paragraphs(40)
    windows = chunk_text(text, max_chars=1000, overlap_chars=200)
    assert all(e - s <= 1000 for s, e in windows)
    assert windows[0][0] == 0 and windows[-1][1] == len(text)
    # No gaps: each window starts at or before the previous one's end
    assert all(b[0] <= a[1] for a, b in zip(windows, windows[1:]))


def test_chunks_overlap_and_progress():
    text = _paragraphs(40)
    windows = chunk_text(text, max_chars=1000, overlap_chars=200)
    assert all(b[0] > a[0] for a, b in zip(windows, windows[1:]))
    assert any(b[0] < a[1] for a, b in zip(windows, windows[1:])), "expected some overlap"


def test_chunk_rejects_overlap_not_smaller_than_window():
    with pytest.raises(ValueError):
        chunk_text("abc", max_chars=100, overlap_chars=100)


# ── labelling (B1 / B7) ────────────────────────────────────────────────────────
@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(pd_, "CHUNK_CHARS", 1000)
    monkeypatch.setattr(pd_, "OVERLAP_CHARS", 200)


def _contract(text, spans, title="C1", group_id=0):
    return {"title": title, "group_id": group_id, "text": text, "spans": spans}


def test_positive_answers_are_inside_chunk_text(small_chunks):
    text = _paragraphs(30)
    clause = "This Agreement is governed by the laws of the State of New York."
    pos = text.index("Paragraph 12.")
    text = text[:pos] + clause + "\n\n" + text[pos:]
    s = text.index(clause)
    chunks = pd_.build_chunks(_contract(text, {"Governing Law": [(s, s + len(clause))]}), Counter())

    positives = [c for c in chunks if c["labels"]["Governing Law"]["status"] == "positive"]
    assert positives, "clause must be positive in at least one chunk"
    for c in positives:
        for a in c["labels"]["Governing Law"]["answers"]:
            assert c["text"][a["start"]:a["end"]] == a["text"] == clause


def test_span_longer_than_chunk_is_clipped_not_lost(small_chunks):
    text = _paragraphs(30)
    s, e = 2000, 3400  # longer than CHUNK_CHARS
    stats = Counter()
    chunks = pd_.build_chunks(_contract(text, {"Non-Compete": [(s, e)]}), stats)

    positives = [c for c in chunks if c["labels"]["Non-Compete"]["status"] == "positive"]
    assert len(positives) == 1
    assert positives[0]["labels"]["Non-Compete"]["answers"][0]["clipped"] is True
    assert stats["clipped_spans::Non-Compete"] == 1


def test_chunk_cutting_a_span_is_partial_not_negative(small_chunks):
    text = _paragraphs(30)
    s = 950  # straddles the first window edge
    chunks = pd_.build_chunks(_contract(text, {"Governing Law": [(s, s + 120)]}), Counter())
    statuses = {c["labels"]["Governing Law"]["status"] for c in chunks if c["start"] < s + 120 and c["end"] > s}
    assert "positive" in statuses
    assert all(st in ("positive", "partial") for st in statuses)


# ── sampling independence (A3) ─────────────────────────────────────────────────
def _labelled_chunks(n=60):
    chunks = []
    for i in range(n):
        labels = {ct: {"status": "negative", "answers": []} for ct in pd_.CLAUSE_TYPES}
        if i % 7 == 0:
            labels["Governing Law"] = {"status": "positive",
                                       "answers": [{"start": 0, "end": 4, "text": "text", "clipped": False}]}
        chunks.append({"chunk_id": f"c::{i:04d}", "contract_title": f"C{i // 10}", "text": "text " * 10,
                       "labels": labels})
    return chunks


def test_eval_queries_do_not_depend_on_train_sampling():
    train_chunks, test_chunks = _labelled_chunks(), _labelled_chunks()

    pd_.sample_queries(train_chunks, random.Random("42-train-queries"), neg_per_pos=1.5)
    test_a = pd_.sample_queries(test_chunks, random.Random("42-test-queries"), neg_per_pos=2.0)

    pd_.sample_queries(train_chunks, random.Random("42-train-queries"), neg_per_pos=4.0)
    test_b = pd_.sample_queries(test_chunks, random.Random("42-test-queries"), neg_per_pos=2.0)

    assert [q["query_id"] for q in test_a] == [q["query_id"] for q in test_b]


def test_all_positives_kept_and_negative_ratio_applied():
    chunks = _labelled_chunks()
    queries = pd_.sample_queries(chunks, random.Random(0), neg_per_pos=2.0)
    gl = [q for q in queries if q["clause_type"] == "Governing Law"]
    n_pos = sum(1 for q in gl if q["answers"])
    assert n_pos == sum(1 for c in chunks if c["labels"]["Governing Law"]["status"] == "positive")
    assert len(gl) - n_pos == 2 * n_pos


# ── group split (A8) ───────────────────────────────────────────────────────────
def test_near_duplicates_land_in_same_split():
    base = " ".join(f"clause{i} term{i} party{i}" for i in range(400))
    contracts = [{"title": f"K{i}", "text": f"unique{i} " * 50 + " ".join(f"x{i}y{j}" for j in range(300)),
                  "spans": {}} for i in range(30)]
    contracts.append({"title": "DUP-A", "text": base, "spans": {}})
    contracts.append({"title": "DUP-B", "text": base + " amendment", "spans": {}})

    group_of, pairs = pd_.group_near_duplicates(contracts)
    assert any({p["a"], p["b"]} == {"DUP-A", "DUP-B"} for p in pairs)

    splits = pd_.split_by_group(contracts, group_of)
    where = {c["title"]: name for name, cs in splits.items() for c in cs}
    assert where["DUP-A"] == where["DUP-B"]


# ── label map (B2 / B3) ────────────────────────────────────────────────────────
def test_label_map_uses_exact_categories_and_excludes_opposites():
    assert "Uncapped Liability" not in pd_.CLAUSE_MAP
    assert "Competitive Restriction Exception" not in pd_.CLAUSE_MAP
    assert "Notice Period To Terminate Renewal" not in pd_.CLAUSE_MAP
    assert pd_.CLAUSE_MAP["Liquidated Damages"] == "Liquidated Damages"
    assert len(pd_.CLAUSE_TYPES) == 6
