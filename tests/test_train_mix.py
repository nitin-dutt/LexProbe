import json
import os
import random

import pytest

from prepare_data import CLAUSE_TYPES
from train_mix import sample_mix, is_near_miss, sha256


def make_chunk(i, text="plain boilerplate", status=None):
    status = status or {}
    labels = {ct: {"status": status.get(ct, "negative"), "answers": []} for ct in CLAUSE_TYPES}
    for ct, st in status.items():
        if st == "positive":
            labels[ct]["answers"] = [{"start": 0, "end": 5, "text": text[:5], "clipped": False}]
    return {"chunk_id": f"c::{i:04d}", "contract_title": "c", "group_id": 0, "chunk_index": i,
            "start": 0, "end": len(text), "text": text, "labels": labels}


@pytest.fixture
def chunks():
    cs, i = [], 0
    for _ in range(10):  # 10 Liquidated Damages positives
        cs.append(make_chunk(i, "liquidated damages of $5 per day", {"Liquidated Damages": "positive"})); i += 1
    for _ in range(5):   # near-misses: own keywords but labelled negative
        cs.append(make_chunk(i, "a late fee applies to overdue invoices")); i += 1
    for _ in range(5):   # neighbour-clause positives (hard by neighbour)
        cs.append(make_chunk(i, "governed by the laws of Ohio", {"Governing Law": "positive"})); i += 1
    for _ in range(100):  # easy negatives
        cs.append(make_chunk(i, "the parties sign here")); i += 1
    cs.append(make_chunk(i, "partial text", {"Liquidated Damages": "partial"}))
    return cs


def ld(rows):
    return [r for r in rows if r["clause_type"] == "Liquidated Damages"]


def test_all_positives_kept_and_oversampled(chunks):
    rows = ld(sample_mix(chunks, random.Random(0), 4.0, 0.7, {"Liquidated Damages": 2}))
    assert sum(1 for r in rows if r["answers"]) == 20  # 10 positives x2


def test_negatives_scale_with_unique_positives_not_oversampling(chunks):
    rows = ld(sample_mix(chunks, random.Random(0), 4.0, 0.7, {"Liquidated Damages": 3}))
    assert sum(1 for r in rows if not r["answers"]) == 40


def test_hard_fraction_and_definition(chunks):
    rows = ld(sample_mix(chunks, random.Random(0), 4.0, 0.7, {}))
    hard = [r for r in rows if r["hard_negative"]]
    assert len(hard) == 10  # only 5 keyword near-misses + 5 neighbour positives exist
    assert all(not r["answers"] for r in hard)
    assert all(is_near_miss(next(c for c in chunks if c["chunk_id"] == r["chunk_id"]), "Liquidated Damages") for r in hard)


def test_partial_chunks_never_used_as_negatives(chunks):
    partial = chunks[-1]["chunk_id"]
    rows = sample_mix(chunks, random.Random(1), 50.0, 0.7, {})
    assert not any(r["chunk_id"] == partial and r["clause_type"] == "Liquidated Damages" for r in rows)


def test_deterministic_and_does_not_mutate_input(chunks):
    before = json.dumps(chunks, sort_keys=True)
    a = sample_mix(chunks, random.Random(7))
    b = sample_mix(chunks, random.Random(7))
    assert [r["query_id"] for r in a] == [r["query_id"] for r in b]
    assert json.dumps(chunks, sort_keys=True) == before


def test_query_format_matches_existing_training_rows(chunks):
    row = ld(sample_mix(chunks, random.Random(0)))[0]
    assert set(row) == {"query_id", "chunk_id", "contract_title", "clause_type", "text",
                        "answers", "explanation", "hard_negative"}
    assert row["explanation"] == ("None" if not row["answers"] else f"Found: {row['answers'][0]['text']}")


MIX = "data/train_mixes/v5"


@pytest.mark.skipif(not os.path.exists(f"{MIX}/manifest.json"), reason="v5 mix not built")
def test_built_mix_leaves_frozen_files_untouched_and_uses_train_chunks_only():
    m = json.load(open(f"{MIX}/manifest.json", encoding="utf-8"))
    for name, digest in m["frozen_files_sha256"].items():
        assert sha256(f"data/processed/{name}") == digest, name
    ds = json.load(open("data/processed/manifest.json", encoding="utf-8"))
    for name in ("val.json", "test.json", "chunks_val.json", "chunks_test.json", "chunks_train.json"):
        assert m["frozen_files_sha256"][name] == ds["files"][name], name
    train_ids = {c["chunk_id"] for c in json.load(open("data/processed/chunks_train.json", encoding="utf-8"))}
    rows = json.load(open(f"{MIX}/train.json", encoding="utf-8"))
    assert {r["chunk_id"] for r in rows} <= train_ids
    assert sha256(f"{MIX}/train.json") == m["train_sha256"]


# ── synthetic rows (training only) ────────────────────────────────────────────
from train_mix import load_synthetic  # noqa: E402


def _syn(text="Liquidated damages of $100 per day shall apply.", answer="Liquidated damages of $100 per day", **kw):
    row = {"query_id": "syn::1", "chunk_id": "syn::1", "contract_title": "synthetic", "clause_type": "Liquidated Damages",
           "text": text, "answers": [{"start": 0, "end": len(answer), "text": answer, "clipped": False}],
           "explanation": f"Found: {answer}", "hard_negative": False, "synthetic": True}
    row.update(kw)
    return row


def _write(tmp_path, rows):
    p = tmp_path / "syn.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return str(p)


def test_valid_synthetic_rows_load(tmp_path):
    assert len(load_synthetic(_write(tmp_path, [_syn()]), set())) == 1


def test_synthetic_answer_must_be_verbatim(tmp_path):
    bad = _syn(answer="Liquidated damages of $100 per day")
    bad["answers"][0]["text"] = "something invented"
    with pytest.raises(ValueError, match="verbatim"):
        load_synthetic(_write(tmp_path, [bad]), set())


def test_synthetic_rows_must_be_flagged_and_known(tmp_path):
    with pytest.raises(ValueError, match="synthetic=true"):
        load_synthetic(_write(tmp_path, [_syn(synthetic=False)]), set())
    with pytest.raises(ValueError, match="unknown clause"):
        load_synthetic(_write(tmp_path, [_syn(clause_type="Made Up")]), set())


def test_synthetic_text_may_not_copy_a_real_chunk(tmp_path):
    row = _syn()
    with pytest.raises(ValueError, match="duplicates a real chunk"):
        load_synthetic(_write(tmp_path, [row]), {row["text"]})
