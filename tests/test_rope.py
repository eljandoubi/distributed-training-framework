"""Unit tests for `src/model/rope.py`: RoPE frequency precomputation and application."""

import torch

from src.model.args import DeepSeekV3ModelArgs
from src.model.rope import apply_rotary_emb, precompute_freqs_cis


def _args_no_yarn() -> DeepSeekV3ModelArgs:
    # max_seq_len == original_seq_len disables the YaRN scaling branch, giving plain RoPE.
    return DeepSeekV3ModelArgs(max_seq_len=8, original_seq_len=8, qk_rope_head_dim=4)


def test_precompute_freqs_cis_shape_and_unit_magnitude():
    """`freqs_cis` should have shape (max_seq_len, rope_dim/2) and unit-magnitude complex entries."""
    args = _args_no_yarn()
    freqs_cis = precompute_freqs_cis(args)

    assert freqs_cis.shape == (args.max_seq_len, args.qk_rope_head_dim // 2)
    assert freqs_cis.is_complex()
    # Every entry is e^{i*theta}, so its magnitude must be (approximately) 1.
    assert torch.allclose(freqs_cis.abs(), torch.ones_like(freqs_cis.abs()), atol=1e-6)


def test_apply_rotary_emb_preserves_shape_and_dtype():
    """Rotary embeddings should not change the input tensor's shape or dtype."""
    args = _args_no_yarn()
    freqs_cis = precompute_freqs_cis(args)

    batch_size, n_heads = 2, 3
    x = torch.randn(
        batch_size,
        args.max_seq_len,
        n_heads,
        args.qk_rope_head_dim,
        dtype=torch.float32,
    )
    out = apply_rotary_emb(x, freqs_cis)

    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_apply_rotary_emb_preserves_vector_norm():
    """Rotation is norm-preserving: each (pos, head) vector's L2 norm should be unchanged by RoPE."""
    args = _args_no_yarn()
    freqs_cis = precompute_freqs_cis(args)

    x = torch.randn(1, args.max_seq_len, 1, args.qk_rope_head_dim)
    out = apply_rotary_emb(x, freqs_cis)

    norm_before = x.norm(dim=-1)
    norm_after = out.norm(dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-5, rtol=1e-5)


def test_apply_rotary_emb_position_zero_is_identity():
    """At position 0, all RoPE angles are zero, so the embedding should leave the input unchanged."""
    args = _args_no_yarn()
    freqs_cis = precompute_freqs_cis(args)

    x = torch.randn(1, 1, 1, args.qk_rope_head_dim)
    out = apply_rotary_emb(x, freqs_cis[:1])

    assert torch.allclose(out, x, atol=1e-5, rtol=1e-5)
