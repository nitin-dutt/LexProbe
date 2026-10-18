"""
Single source of truth for the clause-detection prompt and its target format
(alignment_plan C2, C4).

Training, evaluation and (from Phase 4) serving must all build model input through
this module so the model sees exactly the same format everywhere:
  - the official Llama-3.1 chat template via tokenizer.apply_chat_template
  - exactly one BOS token (the template adds it; tokenization adds none)
  - the whole chunk as context — chunks are already sized to the budget by
    prepare_data.py, so nothing is sliced here
"""

SYSTEM_PROMPT = (
    "You are a legal contract analyst. Your job is to determine whether a specific "
    "clause type is present in the provided contract excerpt."
)

# Pinned so the template's "Today Date" line never changes the prompt between runs
TEMPLATE_DATE = "26 Jul 2024"

POSITIVE_PREFIX = "Found: "
NEGATIVE_ANSWER = "None"
END_OF_TURN     = "<|eot_id|>"  # Llama-3.1 assistant turn terminator (learned as the stop token)


def build_messages(clause_type: str, text: str) -> list[dict]:
    user = (
        f"Contract excerpt:\n{text}\n\n"
        f'Question: Does this excerpt contain a "{clause_type}" clause?\n'
        f'- If YES: respond with "{POSITIVE_PREFIX.strip()} " followed by the exact relevant text.\n'
        f'- If NO: respond with exactly "{NEGATIVE_ANSWER}".'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_prompt(tokenizer, clause_type: str, text: str) -> str:
    """Prompt string ending with the assistant header, ready for generation."""
    return tokenizer.apply_chat_template(
        build_messages(clause_type, text),
        tokenize=False,
        add_generation_prompt=True,
        date_string=TEMPLATE_DATE,
    )


QUESTION_MARKER = "Question: Does this excerpt"


def split_prompt(tokenizer, clause_type: str, text: str) -> tuple[str, str]:
    """
    (prefix, suffix) of the prompt, split where the clause-specific question begins.
    The prefix (system message + excerpt) is identical for every clause type, so its
    computation can be shared across the 6 clause questions (KV-cache reuse in
    src/evaluate_documents.py). prefix + suffix == build_prompt(...) exactly.
    """
    full = build_prompt(tokenizer, clause_type, text)
    i = full.rfind(QUESTION_MARKER)  # rfind: the real question is last, even if the excerpt quotes it
    if i == -1:
        raise ValueError("question marker not found in prompt")
    return full[:i], full[i:]


def encode_prompt(tokenizer, clause_type: str, text: str) -> list[int]:
    """Token ids of the prompt. The chat template already inserts the BOS token, so no special tokens are added."""
    return tokenizer(build_prompt(tokenizer, clause_type, text), add_special_tokens=False)["input_ids"]


def _trim_to_words(text: str, max_chars: int) -> str:
    """Verbatim prefix of `text`, at most max_chars, cut at a word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars + 1)
    return text[: cut if cut > 0 else max_chars]


def format_target(answers: list[dict], max_answer_chars: int, multi_span: str = "first") -> str:
    """
    Assistant response for a training sample (alignment_plan B5, C1).

    answers          — gold spans in the chunk (from prepare_data.py), sorted by position
    max_answer_chars — the quote is trimmed to a verbatim prefix of this length, so long
                       clauses never push the sequence past the token budget
    multi_span       — "first": quote the first span; "join": all spans joined with " ... "
    """
    if not answers:
        return NEGATIVE_ANSWER
    if multi_span == "first":
        quote = answers[0]["text"]
    elif multi_span == "join":
        quote = " ... ".join(a["text"] for a in answers)
    else:
        raise ValueError(f"Unknown multi_span mode: {multi_span}")
    return POSITIVE_PREFIX + _trim_to_words(quote, max_answer_chars)
