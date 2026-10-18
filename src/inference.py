"""
Full-document clause analysis (alignment_plan E1, E2, C2, D7).

Pure orchestration with no torch imports: the model is passed in as callables, so the
logic is unit-tested without a GPU and the API (api/model.py) supplies the real model.

  score_chunk(chunk_text)         -> {clause: score}   all clause scores for one chunk at once
                                                       (v4 margin, or the blended score)
  quote(clause_type, chunk_text)  -> str               generated evidence after a forced "Found:" prefix

`analyze_document` is the older per-clause form: decide(clause_type, chunk_text) -> margin.

Steps (same text handling as training/evaluation):
  1. normalise whitespace and chunk the WHOLE document with the training chunker
  2. score every (chunk, clause) pair
  3. for each clause flagged anywhere, generate one quote from its highest-scoring chunk
  4. check the quote is really in that chunk (grounding) and return its position
"""

from chunking import chunk_text, normalize_text
from eval_metrics import clean_quote, grounding
from prepare_data import CHUNK_CHARS, CLAUSE_TYPES, OVERLAP_CHARS


def _locate(quote: str, text: str, start: int, end: int) -> tuple[int, int] | tuple[None, None]:
    """Character span of the quote inside text[start:end], whitespace/case-insensitive."""
    q = " ".join(quote.split()).lower()
    if not q:
        return None, None
    idx = text.lower().find(q, start, end)
    return (idx, idx + len(q)) if idx != -1 else (None, None)


def analyze_scored(text: str, score_chunk, quote, clause_types=CLAUSE_TYPES, threshold: float = 0.0) -> dict:
    """Analyse a whole document. A clause is flagged if ANY chunk's score exceeds `threshold`."""
    norm, _ = normalize_text(text)
    windows = chunk_text(norm, CHUNK_CHARS, OVERLAP_CHARS)
    chunk_scores = [score_chunk(norm[s:e]) for s, e in windows]

    clauses = []
    for ct in clause_types:
        hits = [(chunk_scores[i][ct], i, s, e) for i, (s, e) in enumerate(windows) if chunk_scores[i][ct] > threshold]
        result = {
            "clause_type": ct,
            "detected": bool(hits),
            "chunks_flagged": len(hits),
            "confidence_margin": None,
            "extracted_text": None,
            "grounded": None,
            "start": None,
            "end": None,
            "chunk_index": None,
        }
        if hits:
            score, i, s, e = max(hits)
            q = clean_quote(quote(ct, norm[s:e]))
            start, end = _locate(q, norm, s, e)
            result.update({
                "confidence_margin": round(float(score), 3),
                "extracted_text": q or None,
                "grounded": bool(q) and grounding(q, norm[s:e])["grounded"],
                "start": start,
                "end": end,
                "chunk_index": i,
            })
        clauses.append(result)

    return {"contract_text": norm, "chunks_analyzed": len(windows), "clauses": clauses}


def analyze_document(text: str, decide, quote, clause_types=CLAUSE_TYPES, threshold: float = 0.0) -> dict:
    """Per-clause form: decide(clause_type, chunk_text) -> margin."""
    return analyze_scored(text, lambda chunk: {ct: decide(ct, chunk) for ct in clause_types}, quote, clause_types, threshold)
