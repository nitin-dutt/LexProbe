import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from answer_loss import answer_only_loss  # noqa: E402


class TinyLM(torch.nn.Module):
    """Embedding -> linear head. Accepts logits_to_keep (index tensor) like Hugging Face causal LMs."""

    def __init__(self, vocab=11, dim=8):
        super().__init__()
        torch.manual_seed(0)
        self.emb, self.head = torch.nn.Embedding(vocab, dim), torch.nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None, logits_to_keep=None):
        logits = self.head(self.emb(input_ids))
        if logits_to_keep is not None:
            logits = logits[:, logits_to_keep, :]
        return type("Out", (), {"logits": logits})()


def default_loss(model, ids, labels, num_items=None):
    logits = model(ids).logits
    total = F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100, reduction="sum")
    return total / (num_items if num_items is not None else (labels[0, 1:] != -100).sum())


def example():
    ids = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]])
    labels = torch.tensor([[-100, -100, -100, -100, 5, 9, 2, 6]])  # only the answer carries loss
    return ids, labels


def test_loss_and_gradients_match_the_default_loss():
    ids, labels = example()
    a, b = TinyLM(), TinyLM()
    la = default_loss(a, ids, labels); la.backward()
    lb = answer_only_loss(b, {"input_ids": ids, "labels": labels}); lb.backward()
    assert lb.item() == pytest.approx(la.item(), rel=1e-6)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6)


def test_uses_num_items_in_batch_like_huggingface():
    ids, labels = example()
    m = TinyLM()
    assert answer_only_loss(m, {"input_ids": ids, "labels": labels}, num_items_in_batch=16).item() == \
        pytest.approx(default_loss(m, ids, labels, 16).item(), rel=1e-6)


def test_only_answer_positions_are_materialised():
    ids, labels = example()
    _, out = answer_only_loss(TinyLM(), {"input_ids": ids, "labels": labels}, return_outputs=True)
    assert out.logits.shape[1] == 4  # 4 answer tokens, not 8 positions


def test_rejects_batches_larger_than_one():
    ids, labels = example()
    with pytest.raises(ValueError):
        answer_only_loss(TinyLM(), {"input_ids": ids.repeat(2, 1), "labels": labels.repeat(2, 1)})
