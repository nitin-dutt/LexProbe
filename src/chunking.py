"""
Text normalisation and chunking shared by data preparation, evaluation and serving.

The same functions must be used everywhere a contract is turned into model input,
so the text the model sees at serving time matches what it was trained and evaluated on.
"""

import re

_INLINE_WS = " \t\r\f\v "


def normalize_text(text: str) -> tuple[str, list[int]]:
    """
    Collapse whitespace while keeping paragraph structure:
      - runs of spaces/tabs        -> one space
      - spaces around a newline    -> removed
      - 3+ consecutive newlines    -> two newlines

    Returns (normalized_text, offset_map) where offset_map[i] is the index in the
    normalized text corresponding to original index i. The map has len(text) + 1
    entries so exclusive span ends can be mapped too.
    """
    out: list[str] = []
    offset_map: list[int] = []

    for ch in text:
        offset_map.append(len(out))
        if ch in _INLINE_WS:
            if out and out[-1] not in (" ", "\n"):
                out.append(" ")
        elif ch == "\n":
            if out and out[-1] == " ":
                out.pop()
            if len(out) >= 2 and out[-1] == "\n" and out[-2] == "\n":
                continue
            if out:
                out.append("\n")
        else:
            out.append(ch)

    offset_map.append(len(out))
    # A popped trailing space can leave map entries pointing one past the end
    n = len(out)
    offset_map = [min(i, n) for i in offset_map]
    return "".join(out), offset_map


def _find_break(text: str, lo: int, hi: int) -> int:
    """Best split position in text[lo:hi]: paragraph > line > sentence > word. Returns hi if none."""
    for sep in ("\n\n", "\n", ". ", "; ", " "):
        idx = text.rfind(sep, lo, hi)
        if idx != -1:
            return idx + len(sep)
    return hi


def _find_next_break(text: str, lo: int, hi: int) -> int:
    """Earliest line > sentence > word boundary in text[lo:hi] (maximises overlap). Returns lo if none."""
    for sep in ("\n", ". ", "; ", " "):
        idx = text.find(sep, lo, hi)
        if idx != -1 and idx + len(sep) < hi:
            return idx + len(sep)
    return lo


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    """
    Split text into windows of at most `max_chars`, preferring natural boundaries.
    Consecutive windows overlap by up to `overlap_chars` (snapped to a boundary) so a
    clause cut at one window's edge appears whole in the next.

    Returns a list of (start, end) offsets; each chunk is text[start:end].
    """
    if max_chars <= overlap_chars:
        raise ValueError("max_chars must be larger than overlap_chars")

    spans = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            end = _find_break(text, start + max_chars // 2, end)
        if text[start:end].strip():
            spans.append((start, end))
        if end >= n:
            break
        # Next window starts up to `overlap_chars` back, snapped forward to a boundary
        lo = max(end - overlap_chars, start + 1)
        next_start = _find_next_break(text, lo, end) if overlap_chars > 0 else end
        start = max(next_start, start + 1)
    return spans
