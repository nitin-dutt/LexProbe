"""
Memory-lean training loss: logits only at the answer positions.

Why: on the 8 GB card the full (sequence x 128,256 vocab) fp32 logits cost ~0.4-1.2 GB per step. Only the
answer tokens carry loss (everything else is -100), so we ask the model for logits at just those
positions (`logits_to_keep=<indices>`). Mathematically identical to the default loss: the same tokens,
summed, then divided by the same count (`num_items_in_batch`, as Hugging Face does).
Requires micro-batch size 1 (no padding), which every config here uses.
"""

import torch
import torch.nn.functional as F


def answer_only_loss(model, inputs: dict, num_items_in_batch=None, return_outputs: bool = False):
    labels = inputs["labels"]
    if labels.shape[0] != 1:
        raise ValueError("answer_only_loss needs micro-batch size 1")
    keep = (labels[0, 1:] != -100).nonzero(as_tuple=True)[0]  # positions t whose logits predict a labelled token t+1
    forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
    out = model(**forward_inputs, logits_to_keep=keep)
    logits = out.logits[0].float()
    total = F.cross_entropy(logits, labels[0, 1:][keep], reduction="sum")
    loss = total / num_items_in_batch if num_items_in_batch is not None else total / max(1, keep.numel())
    return (loss, out) if return_outputs else loss
