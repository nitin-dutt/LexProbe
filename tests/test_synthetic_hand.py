import json
import os

import pytest

import synthetic_hand as H
from train_mix import load_synthetic

BLOCK = """=== POS | termination_fee | licence
Intro sentence number one about the term of the licence and renewal. Another sentence about notices and payment timing for the parties. More text so that the excerpt is long enough to trim safely, covering reporting, records and audit mechanics in detail.

[[[If Licensee terminates this Agreement for convenience before the third anniversary, Licensee shall pay Licensor a termination fee equal to the Minimum Annual Royalty for the Contract Year in which termination takes effect.]]]

Closing sentences about survival, return of materials and the effect of termination on accrued rights, written at some length so the excerpt stays above the minimum size after trimming.

=== NEG | notices | licence
Notices must be in writing and delivered by hand or courier to the addresses on the signature page. A notice is deemed received on the second business day after dispatch. Any change of address must be notified in writing. This section also covers the language of notices, the use of email for operational messages only, and the persons authorized to give notices on behalf of each party, so the text is long enough to be a realistic excerpt of a contract.
"""


def test_parse_and_check_blocks():
    blocks = H.parse_blocks(BLOCK, "t.txt")
    assert [b["label"] for b in blocks] == ["POS", "NEG"]
    ok, reason, span, text = H.check_block(blocks[0], [], [])
    assert ok and text[span[0]:span[1]].startswith("If Licensee terminates")


@pytest.mark.parametrize("bad,reason", [
    ("pay a fee of ____ dollars", "template_artifact"),
    ("pay [PARTY B] the REDACTED sum", "template_artifact"),
])
def test_artifacts_are_rejected(bad, reason):
    b = H.parse_blocks(BLOCK, "t.txt")[0]
    b["text"] = b["text"].replace("pay Licensor a termination fee", bad)
    assert H.check_block(b, [], [])[1] == reason


def test_negative_with_ld_wording_is_rejected():
    n = H.parse_blocks(BLOCK, "t.txt")[1]
    n["text"] += " The parties agree these are liquidated damages."
    assert H.check_block(n, [], [])[1] == "negative_has_ld_wording"


def test_roughen_keeps_quotes_verbatim_and_hides_the_clean_start_cue():
    rows = []
    for i in range(40):
        label = "POS" if i % 2 == 0 else "NEG"
        blocks = H.parse_blocks(BLOCK, "t.txt")
        b = blocks[0] if label == "POS" else blocks[1]
        ok, _, span, text = H.check_block(b, [], [])
        rows.append({"query_id": f"q{i}", "text": text, "answers": [{"start": span[0], "end": span[1], "text": text[span[0]:span[1]], "clipped": False}] if span else []})
    out = H.roughen(rows, min_chars=300)
    for r in out:
        for a in r["answers"]:
            assert r["text"][a["start"]:a["end"]] == a["text"]
    clean = lambda r: r["text"][:1].isupper()  # noqa: E731
    pos_clean = sum(clean(r) for r in out if r["answers"]) / 20
    neg_clean = sum(clean(r) for r in out if not r["answers"]) / 20
    assert pos_clean < 0.9 and neg_clean < 0.9  # some excerpts now start mid-sentence in BOTH classes


@pytest.mark.skipif(not os.path.exists("data/synthetic/hand_v1.json"), reason="hand-written set not built")
def test_built_set_loads_with_the_training_mix_loader():
    rows = load_synthetic("data/synthetic/hand_v1.json", set())
    assert any(r["answers"] for r in rows) and any(not r["answers"] for r in rows)
