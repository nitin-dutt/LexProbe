import json
import os

import pytest

import prepare_data as pdm


@pytest.fixture
def restore_profile():
    saved = (dict(pdm.CLAUSE_MAP), list(pdm.CLAUSE_TYPES), pdm.DATASET_VERSION)
    yield
    pdm.CLAUSE_MAP.clear(); pdm.CLAUSE_MAP.update(saved[0])
    pdm.CLAUSE_TYPES[:] = saved[1]
    pdm.DATASET_VERSION = saved[2]


def test_default_is_the_six_reported_clauses():
    assert len(pdm.CLAUSE_TYPES) == 6 and "Anti-Assignment" not in pdm.CLAUSE_TYPES


def test_v4_profile_adds_exactly_one_clause_and_changes_nothing_else(restore_profile):
    before = list(pdm.CLAUSE_TYPES)
    out = pdm.apply_profile("v4")
    assert out == "data/processed_v4" and pdm.DATASET_VERSION == "v4"
    assert pdm.CLAUSE_TYPES == sorted(before + ["Anti-Assignment"])
    assert pdm.CLAUSE_MAP["Anti-Assignment"] == "Anti-Assignment"


def test_v3_profile_keeps_the_reported_output_dir(restore_profile):
    assert pdm.apply_profile("v3") == "data/processed"


RAW = "data/raw/CUAD_v1.json"


@pytest.mark.skipif(not os.path.exists(RAW), reason="raw CUAD not present")
def test_v4_uses_the_same_contracts_per_split_as_v3(restore_profile):
    pdm.apply_profile("v4")
    contracts, _, _ = pdm.load_contracts(RAW)
    groups, _ = pdm.group_near_duplicates(contracts)
    splits = pdm.split_by_group(contracts, groups)
    v3 = json.load(open("data/processed/manifest.json", encoding="utf-8"))["splits"]
    assert {k: sorted(c["title"] for c in v) for k, v in splits.items()} == v3
