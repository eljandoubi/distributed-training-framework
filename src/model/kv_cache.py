"""KV cache for Multi-Head Latent Attention (MLA) inference.

Per the DeepSeek-V2/V3 papers, MLA compresses keys and values into a shared low-rank
latent vector `c^KV` (dimension `kv_lora_rank`) plus a small decoupled rotary key
`k^R` (dimension `qk_rope_head_dim`). Both are identical across all attention heads,
so caching only `[latent, k_rope]` -- as `MLAKVCache` does -- shrinks the KV-cache
footprint by roughly:

    n_heads * (qk_nope_head_dim + v_head_dim) / (kv_lora_rank + qk_rope_head_dim)

compared to a conventional per-head KV cache (DeepSeek-V2 reports a 93.3% reduction).

Two inference modes can read/write this same cache:

- `Attention.forward` (vanilla): re-projects the cached latent through `wkv_b` into
  per-head keys/values on every call. Simple, but redoes that projection over the
  whole cached prefix at every decoding step.
- `Attention.forward_absorbed` (after `absorb_mla_weights`): folds `wkv_b` into the
  query/output projections ("weight absorption"), so attention is computed directly
  against the compressed cache with no per-step re-projection of past tokens. This is
  the efficient decoding path described in the papers.

References:
    - DeepSeek-V2: https://arxiv.org/abs/2405.04434 (Section 2.1, Multi-Head Latent Attention)
    - DeepSeek-V3: https://arxiv.org/abs/2412.19437

Safety: this cache is strictly an inference-time optimization. `Attention.forward` /
`Attention.forward_absorbed` / `DeepSeekV3Model.forward` all raise `RuntimeError` if a
cache is passed while the module is in training mode, so it can never silently affect
gradients or interact with the distributed training loop (DP/FSDP/TP/PP/CP) in
`src/train.py`. The cache itself is a plain object built via `DeepSeekV3Model.build_kv_cache`
and passed explicitly as a `forward` argument -- it is never attached as a submodule/buffer
of the trained model, so it is never touched by FSDP/DDP wrapping, `torch.compile`, or
checkpointing of the training model.
"""

import torch
from torch import nn

__all__ = ["MLAKVCache"]


class MLAKVCache(nn.Module):
    """Holds the compressed latent + decoupled rotary-key cache for every MLA layer, for incremental decoding."""

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        """Allocate zero-initialized latent/rope cache buffers for `n_layers`, sized for `max_batch_size` x `max_seq_len`."""
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.register_buffer(
            "latent_cache",
            torch.zeros(
                n_layers,
                max_batch_size,
                max_seq_len,
                kv_lora_rank,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "rope_cache",
            torch.zeros(
                n_layers,
                max_batch_size,
                max_seq_len,
                qk_rope_head_dim,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )

    @torch.no_grad()
    def update(
        self,
        layer_id: int,
        start_pos: int,
        latent: torch.Tensor,
        k_rope: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write the new tokens' `latent`/`k_rope` into the cache at `start_pos` and return the full cached prefix.

        Args:
            layer_id: Index of the transformer layer this cache slot belongs to.
            start_pos: Position of the first new token within the full sequence.
            latent: New tokens' compressed latent, shape `(batch_size, seq_len, kv_lora_rank)`.
            k_rope: New tokens' decoupled rotary key, shape `(batch_size, seq_len, qk_rope_head_dim)`.

        Returns:
            `(cached_latent, cached_k_rope)`, each covering every token from position 0 up to
            `start_pos + seq_len` (i.e. the previously cached prefix plus the new tokens).
        """
        batch_size, seq_len, _ = latent.shape
        end_pos = start_pos + seq_len
        assert batch_size <= self.max_batch_size, (batch_size, self.max_batch_size)
        assert end_pos <= self.max_seq_len, (
            f"KV cache exceeded max_seq_len: {end_pos} > {self.max_seq_len}"
        )

        self.latent_cache[layer_id, :batch_size, start_pos:end_pos] = latent
        self.rope_cache[layer_id, :batch_size, start_pos:end_pos] = k_rope

        return (
            self.latent_cache[layer_id, :batch_size, :end_pos],
            self.rope_cache[layer_id, :batch_size, :end_pos],
        )

    def reset(self) -> None:
        """Zero out the cache, e.g. to reuse the same buffers for a new, independent generation."""
        self.latent_cache.zero_()
        self.rope_cache.zero_()
