"""Unit tests for `MLAKVCache` (pure tensor logic, no distributed process group needed)."""

import torch

from src.model.kv_cache import MLAKVCache


def test_kv_cache_update_prefill_then_decode():
    """Sequential `update()` calls (prefill, then single-token decode steps) should return a growing cached prefix."""
    n_layers, max_batch, max_seq, kv_lora_rank, rope_dim = 2, 3, 8, 4, 2
    cache = MLAKVCache(
        n_layers=n_layers,
        max_batch_size=max_batch,
        max_seq_len=max_seq,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope_dim,
        dtype=torch.float32,
    )

    batch_size = 2
    prefill_len = 3
    prefill_latent = torch.randn(batch_size, prefill_len, kv_lora_rank)
    prefill_rope = torch.randn(batch_size, prefill_len, rope_dim)

    cached_latent, cached_rope = cache.update(
        layer_id=0, start_pos=0, latent=prefill_latent, k_rope=prefill_rope
    )
    assert cached_latent.shape == (batch_size, prefill_len, kv_lora_rank)
    assert torch.allclose(cached_latent, prefill_latent)
    assert torch.allclose(cached_rope, prefill_rope)

    # Decode one more token at position `prefill_len`.
    new_latent = torch.randn(batch_size, 1, kv_lora_rank)
    new_rope = torch.randn(batch_size, 1, rope_dim)
    cached_latent, cached_rope = cache.update(
        layer_id=0, start_pos=prefill_len, latent=new_latent, k_rope=new_rope
    )
    assert cached_latent.shape == (batch_size, prefill_len + 1, kv_lora_rank)
    assert torch.allclose(cached_latent[:, :prefill_len], prefill_latent)
    assert torch.allclose(cached_latent[:, prefill_len:], new_latent)

    # Layer 1's cache must be untouched (each layer has its own slot).
    other_layer_latent, _ = cache.update(
        layer_id=1,
        start_pos=0,
        latent=torch.zeros(batch_size, 1, kv_lora_rank),
        k_rope=torch.zeros(batch_size, 1, rope_dim),
    )
    assert torch.allclose(other_layer_latent, torch.zeros_like(other_layer_latent))


def test_kv_cache_rejects_overflow():
    """Writing past `max_seq_len` should raise, rather than silently truncating or wrapping."""
    cache = MLAKVCache(
        n_layers=1,
        max_batch_size=1,
        max_seq_len=2,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        dtype=torch.float32,
    )
    latent = torch.randn(1, 3, 4)  # seq_len=3 > max_seq_len=2
    k_rope = torch.randn(1, 3, 2)
    try:
        cache.update(layer_id=0, start_pos=0, latent=latent, k_rope=k_rope)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected an assertion error when exceeding max_seq_len")


def test_kv_cache_reset_zeroes_buffers():
    """`reset()` should zero out both the latent and rope cache buffers."""
    cache = MLAKVCache(
        n_layers=1,
        max_batch_size=1,
        max_seq_len=4,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        dtype=torch.float32,
    )
    cache.update(
        layer_id=0,
        start_pos=0,
        latent=torch.randn(1, 2, 4),
        k_rope=torch.randn(1, 2, 2),
    )
    cache.reset()
    assert torch.all(cache.latent_cache == 0)
    assert torch.all(cache.rope_cache == 0)
