import os

import pytest
import yaml

from prepare_data import CLAUSE_TYPES
from test_train_mix import make_chunk
from training_data import (realistic_ap_metrics, select_realistic_val, verify_train_mix,
                           weighted_average_precision)


def pool():
    cs, i = [], 0
    for _ in range(20):
        cs.append(make_chunk(i, "liquidated damages of $5 per day", {"Liquidated Damages": "positive"})); i += 1
    for _ in range(40):
        cs.append(make_chunk(i, "a late fee applies")); i += 1          # near-miss negatives
    for _ in range(400):
        cs.append(make_chunk(i, "the parties sign here")); i += 1       # easy negatives
    return cs


def test_weights_restore_the_real_mix():
    items = select_realistic_val(pool(), CLAUSE_TYPES, 5, 8, 10, seed=1)
    ld = [it for it in items if it["clause_type"] == "Liquidated Damages"]
    assert sum(it["weight"] for it in ld if it["label"] == 1) == pytest.approx(20)
    assert sum(it["weight"] for it in ld if it["label"] == 0) == pytest.approx(440)
    assert len(ld) == 5 + 8 + 10


def test_selection_is_deterministic():
    a = [it["chunk_id"] for it in select_realistic_val(pool(), CLAUSE_TYPES, 5, 8, 10, 1)]
    b = [it["chunk_id"] for it in select_realistic_val(pool(), CLAUSE_TYPES, 5, 8, 10, 1)]
    assert a == b


def test_weighted_ap_matches_hand_values():
    assert weighted_average_precision([1, 1, 0, 0], [4, 3, 2, 1], [1, 1, 1, 1]) == 1.0
    # ranking pos, neg, pos -> precision 1 at first positive, 2/3 at second -> (1 + 2/3) / 2
    assert weighted_average_precision([1, 0, 1], [3, 2, 1], [1, 1, 1]) == pytest.approx(5 / 6)
    # a heavy negative ranked above the positive hurts more than a light one
    heavy = weighted_average_precision([0, 1], [2, 1], [100, 1])
    light = weighted_average_precision([0, 1], [2, 1], [1, 1])
    assert heavy < light


def test_ap_is_worse_for_random_scores_than_perfect_ones():
    items = select_realistic_val(pool(), CLAUSE_TYPES, 5, 8, 10, 1)
    perfect = realistic_ap_metrics(items, [10.0 if it["label"] else -10.0 for it in items])
    flat = realistic_ap_metrics(items, [-10.0 if it["label"] else 10.0 for it in items])  # worst ranking
    assert perfect["realistic_ap"] == 1.0 and flat["realistic_ap"] < 0.5
    assert "realistic_ap_Liquidated_Damages" in perfect


def test_clauses_without_positives_are_skipped_not_counted_as_zero():
    items = select_realistic_val(pool(), CLAUSE_TYPES, 5, 8, 10, 1)
    m = realistic_ap_metrics(items, [10.0 if it["label"] else -10.0 for it in items])
    assert m["realistic_ap"] == 1.0  # only Liquidated Damages has positives here


def test_train_mix_must_match_the_dataset_in_use():
    names = ("val.json", "test.json", "chunks_val.json", "chunks_test.json", "chunks_train.json")
    ds = {"files": {n: f"h-{n}" for n in names}}
    mix = {"frozen_files_sha256": {n: f"h-{n}" for n in names}}
    verify_train_mix(mix, ds)
    mix["frozen_files_sha256"]["val.json"] = "other"
    with pytest.raises(ValueError):
        verify_train_mix(mix, ds)


@pytest.mark.parametrize("path", ["configs/qlora_v5.yaml", "configs/smoke_v5.yaml"])
def test_v5_configs_are_consistent(path):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    assert cfg["training"]["optimizer"] == "adamw_8bit"           # never a paged optimizer (BSOD)
    assert cfg["validation"]["metric_for_best_model"] == "realistic_ap"
    assert cfg["training"]["save_steps"] % cfg["training"]["eval_steps"] == 0
    assert cfg["lora"]["r"] == 16                                  # matches the v4 adapter
    assert cfg["data"]["train_mix_dir"] == "data/train_mixes/v5"
    assert cfg["output_dir"] != "models/llama-3.1-8b-cuad-lora-v4"  # never overwrite the model of record
    assert os.path.isdir(os.path.dirname(cfg["init_adapter"]))
