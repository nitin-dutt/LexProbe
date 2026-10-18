"""
GPU scoring of one chunk for all clause types, shared by the API (alignment_plan D7).

The system prompt and the excerpt are identical for the 6 clause questions, so the excerpt is run once and its
KV cache reused for the 6 short questions (about 3x faster than 6 full passes; the same method and the same
tokenisation check as src/evaluate_documents.py, which verified margins within bf16 rounding, 0 decision flips).

Torch is imported lazily so importing this module does not initialise CUDA.
"""

from prepare_data import CLAUSE_TYPES
from prompting import split_prompt


def chunk_margins(model, tokenizer, found_id: int, none_id: int, text: str, clause_types=CLAUSE_TYPES) -> dict:
    """{clause: logit(Found) - logit(None)} for one chunk, with the model's CURRENT adapter state."""
    import torch

    prefix, _ = split_prompt(tokenizer, clause_types[0], text)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prefix_ids], device="cuda"), use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        res = {}
        for ct in clause_types:
            _, suffix = split_prompt(tokenizer, ct, text)
            sids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
            o = model(input_ids=torch.tensor([sids], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=1)
            logits = o.logits[0, -1]
            res[ct] = float(logits[found_id] - logits[none_id])
            cache.crop(-len(sids))  # drop the question tokens, keep the shared excerpt
    return res


def blend_chunk_margins(model, tokenizer, found_id: int, none_id: int, text: str, clause_types=CLAUSE_TYPES):
    """(v4 margins, zero-shot margins): adapter on, then adapter off. The adapter is always re-enabled."""
    v4 = chunk_margins(model, tokenizer, found_id, none_id, text, clause_types)
    with model.disable_adapter():
        base = chunk_margins(model, tokenizer, found_id, none_id, text, clause_types)
    return v4, base
