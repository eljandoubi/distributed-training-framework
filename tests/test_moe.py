"""Unit tests for `src/model/moe/moe.py`: SwiGLU feed-forward and the MoE layer's routing/combine logic."""

import torch

from src.model.moe.moe import FeedForward, MoE, MoEArgs, TokenChoiceTopKRouter


def test_feed_forward_shape():
    """`FeedForward` should preserve the input's leading dims and project back to `dim`."""
    ff = FeedForward(dim=8, hidden_dim=16)
    x = torch.randn(2, 5, 8)
    out = ff(x)
    assert out.shape == x.shape


def test_token_choice_router_selects_top_k_and_counts_tokens():
    """The router should select exactly `top_k` experts per token and produce a matching token-count histogram."""
    torch.manual_seed(0)
    num_experts, top_k, dim = 6, 2, 8
    router = TokenChoiceTopKRouter(
        dim=dim, num_experts=num_experts, top_k=top_k, score_func="softmax",
        route_norm=False, route_scale=1.0,
    )
    n_tokens = 10
    x = torch.randn(n_tokens, dim)

    top_scores, selected_experts, num_tokens_per_expert = router(x)

    assert top_scores.shape == (n_tokens, top_k)
    assert selected_experts.shape == (n_tokens, top_k)
    assert num_tokens_per_expert.shape == (num_experts,)
    # Total assignments across all experts must equal n_tokens * top_k.
    assert num_tokens_per_expert.sum().item() == n_tokens * top_k
    # No token should route to the same expert twice (topk selects distinct indices).
    for row in selected_experts:
        assert len(set(row.tolist())) == top_k


def test_moe_forward_preserves_shape_and_updates_tokens_per_expert():
    """`MoE.forward` should preserve (batch, seq, dim) shape and accumulate `tokens_per_expert` for load balancing."""
    torch.manual_seed(0)
    moe_args = MoEArgs(
        num_experts=4, num_shared_experts=1, top_k=2,
        score_func="softmax", route_norm=False, score_before_experts=False,
    )
    moe = MoE(moe_args, dim=8, hidden_dim=16)
    moe.init_weights(init_std=0.02, buffer_device=torch.device("cpu"))

    batch_size, seq_len, dim = 2, 3, 8
    x = torch.randn(batch_size, seq_len, dim)

    assert torch.all(moe.tokens_per_expert == 0)
    out = moe(x)

    assert out.shape == (batch_size, seq_len, dim)
    # Every routed token should have incremented some expert's count; total equals n_tokens * top_k.
    assert moe.tokens_per_expert.sum().item() == batch_size * seq_len * moe_args.top_k
