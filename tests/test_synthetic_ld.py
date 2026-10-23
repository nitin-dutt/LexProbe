import json
import random

import pytest

import synthetic_ld as S
from prepare_data import _shingles
from train_mix import load_synthetic

CLAUSE_TEXT = "If Supplier delays delivery, Supplier shall pay liquidated damages of $1,000 for each day of delay, up to 10 percent of the fees."
GOOD = ("1. Payment. Invoices are due within thirty days of receipt and unpaid sums are reviewed monthly by the finance teams of both parties. "
        "Each party keeps accurate records of all amounts invoiced, credited and paid under this agreement for audit purposes. " * 2
        + f"2. Delay. [[[{CLAUSE_TEXT}]]] " + "3. Notices. Notices must be in writing and delivered to the addresses stated in the order form signed by both parties. " * 3)


def test_parse_marked_gives_a_verbatim_slice_and_removes_markers():
    text, a, b = S.parse_marked("  " + GOOD)
    assert text[a:b] == CLAUSE_TEXT and "[[[" not in text and "]]]" not in text and not text.startswith(" ")


@pytest.mark.parametrize("raw", ["no markers here", "[[[one]]] and [[[two]]]", "]]]backwards[[[", "[[[   ]]]"])
def test_parse_marked_rejects_bad_markers(raw):
    assert S.parse_marked(raw) is None


def test_good_positive_passes_and_bad_ones_are_named():
    text, a, b = S.parse_marked(GOOD)
    assert S.filter_candidate(text, (a, b), True, [], []) == (True, "ok")
    assert S.filter_candidate(text[:100], (0, 5), True, [], [])[1] == "length_short"
    assert S.filter_candidate(text * 3, (a, b), True, [], [])[1] == "length_long"
    assert S.filter_candidate(text, None, True, [], [])[1] == "no_markers"
    sent = "Section one covers warranty remedies for defective goods and the repair duties of the seller in detail. "
    plain = sent * 10
    assert S.filter_candidate(plain, (0, len(sent) - 1), True, [], [])[1] == "not_ld_like"
    frag = "[ The agreed fixed charge shall be paid by Licensee in addition to any other obligations under this Agreement"
    fragtext = "Intro text here. " * 30 + frag + " end. " * 30
    start = fragtext.index(frag)
    assert S.filter_candidate(fragtext, (start, start + len(frag)), True, [], [])[1] == "quote_fragment"
    no_trigger = "The agreed damages set out in the schedule of charges shall be paid by the Customer to the Supplier in full."
    nt = "Intro text here. " * 30 + no_trigger + " end. " * 30
    s0 = nt.index(no_trigger)
    assert S.filter_candidate(nt, (s0, s0 + len(no_trigger)), True, [], [])[1] == "quote_without_trigger"
    sc = "Section. " * 90 + " The client is entitled to a service credit on a pro rata basis. "
    assert S.filter_candidate(sc, None, False, [], [])[1] == "negative_service_credit"


def test_negatives_may_not_mention_ld_or_look_like_it():
    neg = "Section one covers warranty remedies in detail with repair or replace obligations for defective goods. " * 8
    assert S.filter_candidate(neg, None, False, [], []) == (True, "ok")
    assert S.filter_candidate(neg + " liquidated damages apply.", None, False, [], [])[1] == "negative_mentions_ld"
    assert S.filter_candidate(neg + " A penalty of $500 per day applies.", None, False, [], [])[1] == "negative_looks_like_ld"


def test_duplicates_are_rejected_against_synthetic_and_real_text():
    text, a, b = S.parse_marked(GOOD)
    sh = _shingles(text)
    assert S.filter_candidate(text, (a, b), True, [sh], [])[1] == "duplicate_synthetic"
    assert S.filter_candidate(text, (a, b), True, [], [sh])[1] == "duplicate_real"


def test_specs_are_reproducible_and_prompts_carry_the_markers_rule():
    s1, s2 = S.pick_spec(random.Random(3), True), S.pick_spec(random.Random(3), True)
    assert s1 == s2
    msgs = S.build_messages(s1, ["clause one", "clause two"])
    assert "[[[" in msgs[1]["content"] and "never copy" in msgs[1]["content"]
    neg = S.build_messages(S.pick_spec(random.Random(3), False), ["a", "b"])
    assert "Do not use any [[[" in neg[1]["content"] and "must not appear" in neg[1]["content"]


def test_rows_pass_the_training_mix_loader(tmp_path):
    text, a, b = S.parse_marked(GOOD)
    pos = S.make_row(1, text, (a, b), S.pick_spec(random.Random(1), True))
    neg = S.make_row(2, "plain neighbouring text " * 40, None, S.pick_spec(random.Random(2), False))
    path = tmp_path / "syn.json"
    path.write_text(json.dumps([pos, neg]), encoding="utf-8")
    rows = load_synthetic(str(path), set())
    assert rows[0]["explanation"].startswith("Found: ") and rows[1]["explanation"] == "None" and rows[1]["hard_negative"]


def test_phrase_share_is_capped_and_negatives_are_untouched():
    mk = lambda i, phrase, pos: {"query_id": f"q{i:03d}", "text": ("a genuine pre-estimate of loss" if phrase else "plain") + " x" * 30,  # noqa: E731
                                 "answers": [{"start": 0, "end": 1, "text": "a"}] if pos else []}
    rows = [mk(i, True, True) for i in range(60)] + [mk(100 + i, False, True) for i in range(20)] + [mk(200 + i, False, False) for i in range(15)]
    out = S.cap_phrase_share(rows, max_share=0.35)
    pos = [r for r in out if r["answers"]]
    share = sum(1 for r in pos if S.PHRASE.search(r["text"])) / len(pos)
    assert share <= 0.36 and sum(1 for r in out if not r["answers"]) == 15 and len(pos) == 20 + 10


def test_most_prompts_no_longer_force_the_shortcut_phrase():
    specs = [S.pick_spec(random.Random(i), True) for i in range(200)]
    share = sum(s["preestimate"] for s in specs) / 200
    assert 0.15 < share < 0.45
    plain = next(s for s in specs if not s["preestimate"])
    assert "do not use the words 'pre-estimate'" in S.build_messages(plain, ["a", "b"])[1]["content"]


def test_select_matched_enforces_cue_ceilings_and_keeps_negatives():
    def mk(i, quote, pos=True):
        text = "intro " * 20 + quote + " outro" * 20
        return {"query_id": f"q{i:03d}", "text": text, "answers": [{"start": 0, "end": 1, "text": quote}] if pos else []}
    rows = [mk(i, "Supplier shall pay $5,000 per day as liquidated damages") for i in range(60)]
    rows += [mk(100 + i, f"The agreed damages stated in schedule {i} apply on delay") for i in range(60)]
    rows += [mk(300 + i, "none", pos=False) for i in range(7)]
    out, rep = S.select_matched(rows, warmup=10)
    assert rep["positives_kept"] > 0
    assert rep["shares"]["dollar"] <= 0.25            # ceiling 0.20 plus the exempt warm-up
    assert sum(1 for r in out if not r["answers"]) == 7
    assert out == sorted(out, key=lambda r: r["query_id"])


def test_triggers_and_amounts_are_always_compatible():
    for i in range(400):
        sp = S.pick_spec(random.Random(i), True)
        delay_amount = sp["amount"] in S.AMOUNTS_PLAIN["delay"] or any(sp["amount"] == t.format(n=0, amt="", p=0) for t in S.AMOUNTS_NUMERIC["delay"]) or "delay" in sp["amount"] or "continues" in sp["amount"]
        if sp["kind"] == "oneoff":
            assert sp["trigger"] in S.ONEOFF_TRIGGERS and "delay" not in sp["amount"] and "continues" not in sp["amount"] and sp["cap"] == ""
        else:
            assert sp["trigger"] in S.DELAY_TRIGGERS
    assert not any("service credit" in t for t in S.NEGATIVE_THEMES)


def test_judge_prompts_ask_the_right_question():
    pos = S.judge_messages(True, "full text", "The quote.", "late delivery")[1]["content"]
    assert "Provision: The quote." in pos and "late delivery" in pos and "coherent and complete" in pos
    neg = S.judge_messages(False, "the excerpt text", None, "x")[1]["content"]
    assert "the excerpt text" in neg and "service credit" in neg and "Yes or No" in neg


def test_placeholders_are_filled_in_text_and_quote_and_offsets_stay_valid():
    quote = "Supplier shall pay [PARTY B] agreed damages as set out in the schedule dated [DATE] for late delivery."
    text = "Intro. " * 30 + quote + " End. " * 30
    start = text.index(quote)
    row = {"query_id": "q1", "text": text, "answers": [{"start": start, "end": start + len(quote), "text": quote, "clipped": False}], "explanation": "x"}
    neg = {"query_id": "q2", "text": "Notice to [ADDRESS] on [DATE]. " * 20, "answers": []}
    out, dropped = S.fill_placeholders([row, neg])
    assert dropped == 0 and not any(S.PLACEHOLDER.search(r["text"]) for r in out)
    a = out[0]["answers"][0]
    assert out[0]["text"][a["start"]:a["end"]] == a["text"] and "[" not in a["text"]


def test_surface_balance_keeps_all_positives_and_matches_negatives_per_bin():
    def mk(i, pos, numbered, n=1200):
        body = ("1. " if numbered else "") + "word " * (n // 5)
        return {"query_id": f"q{i:03d}", "text": body, "answers": [{"start": 0, "end": 1, "text": "w"}] if pos else []}
    rows = [mk(i, True, True) for i in range(30)] + [mk(100 + i, True, False) for i in range(10)]
    rows += [mk(200 + i, False, False) for i in range(60)] + [mk(300 + i, False, True) for i in range(20)]
    out, rep = S.balance_surface(rows)
    pos = [r for r in out if r["answers"]]
    neg = [r for r in out if not r["answers"]]
    assert len(pos) == 40
    assert sum(S.surface_bin(r)[1] for r in neg) == 20            # only 20 numbered negatives exist (30 wanted): all kept, reported
    assert sum(1 for r in neg if not S.surface_bin(r)[1]) == 10   # 10 un-numbered wanted, 10 kept
    assert rep["bins_short_of_negatives"]
