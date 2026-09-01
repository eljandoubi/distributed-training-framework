"""Single-process correctness tests for `Attention`'s MLA KV cache: incremental decoding must match full-sequence
forward, weight absorption must be numerically equivalent, and the cache must be rejected during training."""

import pytest
import torch

from src.model.args import DeepSeekV3ModelArgs
from src.model.kv_cache import MLAKVCache
from src.model.model import Attention
from src.model.rope import precompute_freqs_cis


def _small_model_args(q_lora_rank: int) -> DeepSeekV3ModelArgs:
    return DeepSeekV3ModelArgs(
        max_seq_len=16,
        dim=32,
        n_heads=4,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        # Keep original_seq_len >= max_seq_len so the YaRN scaling branch is inactive,
        # matching the simplest / most common config path.
        original_seq_len=16,
    )


@pytest.mark.parametrize("q_lora_rank", [0, 8])
def test_kv_cache_incremental_decode_matches_full_forward(q_lora_rank: int):
    """Token-by-token decoding through an `MLAKVCache` must reproduce the equivalent full-sequence forward pass."""
    torch.manual_seed(0)
    model_args = _small_model_args(q_lora_rank)
    attn = Attention(layer_id=0, model_args=model_args)
    attn.eval()

    batch_size, seq_len = 2, 6
    x = torch.randn(batch_size, seq_len, model_args.dim)
    freqs_cis = precompute_freqs_cis(model_args)

    with torch.no_grad():
        out_full = attn(x, freqs_cis[:seq_len])

        cache = MLAKVCache(
            n_layers=1,
            max_batch_size=batch_size,
            max_seq_len=seq_len,
            kv_lora_rank=model_args.kv_lora_rank,
            qk_rope_head_dim=model_args.qk_rope_head_dim,
            dtype=x.dtype,
        )
        out_chunks = []
        for t in range(seq_len):
            out_t = attn(
                x[:, t : t + 1],
                freqs_cis[t : t + 1],
                kv_cache=cache,
                start_pos=t,
            )
            out_chunks.append(out_t)
        out_incremental = torch.cat(out_chunks, dim=1)

    assert torch.allclose(out_full, out_incremental, atol=1e-5, rtol=1e-5)


def test_forward_absorbed_matches_forward_with_cache():
    """After `absorb_mla_weights`, `forward_absorbed` with a cache must match the vanilla cached `forward`."""
    torch.manual_seed(0)
    # Weight absorption is only supported when q_lora_rank == 0 (see `Attention.absorb_mla_weights`).
    model_args = _small_model_args(q_lora_rank=0)
    attn = Attention(layer_id=0, model_args=model_args)
    attn.eval()

    batch_size, seq_len = 2, 5
    x = torch.randn(batch_size, seq_len, model_args.dim)
    freqs_cis = precompute_freqs_cis(model_args)

    with torch.no_grad():
        cache_vanilla = MLAKVCache(
            n_layers=1,
            max_batch_size=batch_size,
            max_seq_len=seq_len,
            kv_lora_rank=model_args.kv_lora_rank,
            qk_rope_head_dim=model_args.qk_rope_head_dim,
            dtype=x.dtype,
        )
        out_vanilla = attn(x, freqs_cis[:seq_len], kv_cache=cache_vanilla, start_pos=0)

        attn.absorb_mla_weights()
        assert attn.is_absorbed

        cache_absorbed = MLAKVCache(
            n_layers=1,
            max_batch_size=batch_size,
            max_seq_len=seq_len,
            kv_lora_rank=model_args.kv_lora_rank,
            qk_rope_head_dim=model_args.qk_rope_head_dim,
            dtype=x.dtype,
        )
        out_absorbed = attn.forward_absorbed(
            x, freqs_cis[:seq_len], kv_cache=cache_absorbed, start_pos=0
        )

    assert torch.allclose(out_vanilla, out_absorbed, atol=1e-4, rtol=1e-4)


def test_kv_cache_rejected_during_training():
    """Passing a `kv_cache` while the module is in training mode must raise, never silently proceed."""
    model_args = _small_model_args(q_lora_rank=0)
    attn = Attention(layer_id=0, model_args=model_args)
    attn.train()

    batch_size, seq_len = 1, 2
    x = torch.randn(batch_size, seq_len, model_args.dim)
    freqs_cis = precompute_freqs_cis(model_args)[:seq_len]
    cache = MLAKVCache(
        n_layers=1,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        kv_lora_rank=model_args.kv_lora_rank,
        qk_rope_head_dim=model_args.qk_rope_head_dim,
        dtype=x.dtype,
    )

    with pytest.raises(RuntimeError):
        attn(x, freqs_cis, kv_cache=cache, start_pos=0)


def test_absorb_mla_weights_rejects_q_lora():
    """`absorb_mla_weights` must reject models with `q_lora_rank > 0` (only the `q_lora_rank == 0` path is supported)."""
    model_args = _small_model_args(q_lora_rank=8)
    attn = Attention(layer_id=0, model_args=model_args)

    with pytest.raises(NotImplementedError):
        attn.absorb_mla_weights()
