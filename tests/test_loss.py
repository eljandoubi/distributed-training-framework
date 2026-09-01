"""Unit tests for `src/components/loss.py`: token-normalized cross-entropy loss."""

import torch

from src.components.loss import IGNORE_INDEX, cross_entropy_loss


def test_cross_entropy_loss_matches_manual_sum_reduction():
    """`cross_entropy_loss` should equal `F.cross_entropy` with `reduction='sum'` and the same ignore index."""
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 4, 6
    pred = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = cross_entropy_loss(pred, labels)
    expected = torch.nn.functional.cross_entropy(
        pred.flatten(0, 1), labels.flatten(0, 1), reduction="sum"
    )
    assert torch.allclose(loss, expected)


def test_cross_entropy_loss_ignores_masked_labels():
    """Positions labeled `IGNORE_INDEX` must not contribute to the loss."""
    torch.manual_seed(0)
    vocab_size = 5
    pred = torch.randn(1, 3, vocab_size)
    labels_with_ignore = torch.tensor([[0, IGNORE_INDEX, 2]])
    labels_all_valid = torch.tensor([[0, 1, 2]])

    loss_ignored = cross_entropy_loss(pred, labels_with_ignore)
    # Manually compute the sum loss over only the two valid (non-ignored) positions.
    per_token_loss = torch.nn.functional.cross_entropy(
        pred.flatten(0, 1), labels_all_valid.flatten(0, 1), reduction="none"
    )
    expected = per_token_loss[0] + per_token_loss[2]

    assert torch.allclose(loss_ignored, expected)
