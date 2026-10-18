"""Full-document analysis with a fake model (alignment_plan E1, E2)."""

from inference import analyze_document

CLAUSE = "This Agreement shall be governed by the laws of the State of Delaware."
FILLER = "The parties agree to cooperate in good faith on all matters. "


def fake_decide(clause_type, chunk):
    return 3.0 if clause_type == "Governing Law" and "governed by the laws" in chunk else -3.0


def quote_verbatim(clause_type, chunk):
    return " " + CLAUSE


def quote_invented(clause_type, chunk):
    return " The parties submit to arbitration in Paris."


def _doc_with_clause_at(pos):
    filler = (FILLER * (pos // len(FILLER) + 2))[:pos]
    return filler + "\n\n" + CLAUSE + "\n\n" + FILLER * 20


def test_clause_deep_in_document_is_found():
    """E1 acceptance: a clause at ~char 10,000 must be analysed (old API saw only 3,000)."""
    doc = _doc_with_clause_at(10_000)
    out = analyze_document(doc, fake_decide, quote_verbatim)
    gl = next(c for c in out["clauses"] if c["clause_type"] == "Governing Law")
    assert gl["detected"] and gl["chunk_index"] > 0
    assert out["chunks_analyzed"] > 5


def test_quote_position_points_into_returned_text():
    out = analyze_document(_doc_with_clause_at(4_000), fake_decide, quote_verbatim)
    gl = next(c for c in out["clauses"] if c["clause_type"] == "Governing Law")
    assert gl["grounded"] is True
    assert out["contract_text"][gl["start"]:gl["end"]] == gl["extracted_text"]
    assert CLAUSE.startswith(gl["extracted_text"])  # clean_quote drops the trailing full stop


def test_invented_quote_is_flagged_not_grounded():
    out = analyze_document(_doc_with_clause_at(2_000), fake_decide, quote_invented)
    gl = next(c for c in out["clauses"] if c["clause_type"] == "Governing Law")
    assert gl["detected"] and gl["grounded"] is False and gl["start"] is None


def test_undetected_clauses_have_no_evidence():
    out = analyze_document(_doc_with_clause_at(2_000), fake_decide, quote_verbatim)
    others = [c for c in out["clauses"] if c["clause_type"] != "Governing Law"]
    assert len(others) == 5
    assert all(not c["detected"] and c["extracted_text"] is None for c in others)


def test_whole_document_is_chunked_with_overlap_cover():
    doc = FILLER * 400  # ~24k chars
    seen = []
    analyze_document(doc, lambda ct, ch: seen.append(ch) or -1.0, quote_verbatim, clause_types=["Governing Law"])
    joined_len = sum(len(c) for c in seen)
    assert joined_len >= len(" ".join(doc.split()))  # every character reached the model at least once


def test_higher_threshold_flags_fewer_clauses():
    doc = _doc_with_clause_at(3_000)
    decide = lambda ct, ch: 2.0 if ct == "Governing Law" and "governed by the laws" in ch else -3.0
    assert next(c for c in analyze_document(doc, decide, quote_verbatim, threshold=0.0)["clauses"] if c["clause_type"] == "Governing Law")["detected"]
    assert not next(c for c in analyze_document(doc, decide, quote_verbatim, threshold=3.0)["clauses"] if c["clause_type"] == "Governing Law")["detected"]
