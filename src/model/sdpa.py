import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

__all__ = [
    "ScaledDotProductAttentionWrapper",
]


class ScaledDotProductAttentionWrapper(torch.nn.Module):
    """Wrapper around `F.scaled_dot_product_attention` to make it CP compatible.

    This wrapper is needed because `F.scaled_dot_product_attention` is not a torch.nn.Module, and thus cannot be applied with _ContextParallel.
    We need to wrap it into a torch.nn.Module.

    Note:
        The forward function must have q, k, v as the first three arguments to be compatible with _ContextParallel.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        v_head_dim = v.shape[-1]
        q_head_dim = q.shape[-1]

        # When Tensor Parallel (TP) and Context Parallel (CP) are both active, q, k, v arrive as DTensors on the TP mesh.
        # We must convert them to local tensors so that CP's input hook can properly wrap them on the CP mesh for ring attention.
        # Without this, ring attention would incorrectly communicate on the TP process group instead of the CP process group.
        # This also avoids DTensor dispatch issues with F.pad (no registered sharding strategy for pad), which would change V's placements and cause the CP SDPA handler to fail with "inputs need to be redistributed".
        tp_spec = None
        if isinstance(q, DTensor):
            assert isinstance(k, DTensor) and isinstance(v, DTensor)
            tp_spec = (q.device_mesh, q.placements)
            q = q.to_local()
            k = k.to_local()
            v = v.to_local()

        if v_head_dim < q_head_dim:
            # Flash Attention requires Q, K, V to have the same head_dim, but MLA uses v_head_dim < qk_head_dim. Pad V with zeros so Flash Attention can be selected.
            # This is mathematically lossless:
            # softmax(QK^T/s) @ [V | 0] = [softmax(QK^T/s) @ V | 0], so trimming the output recovers the exact original result.
            v = F.pad(v, (0, q_head_dim - v_head_dim))

        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ],
            set_priority=True,
        ):
            q_len, kv_len = q.shape[-2], k.shape[-2]
            # MLA's weight-absorbed path (`Attention.forward_absorbed`) shares a single K/V
            # "head" across all `n_heads` query heads (MQA-style). The CUDA flash/efficient
            # kernels broadcast this implicitly, but the CPU math fallback requires
            # `enable_gqa=True` to broadcast differing head counts between q and k/v.
            enable_gqa = q.shape[-3] != k.shape[-3]
            if q_len == kv_len:
                # Prefill / training: q and k/v cover the same positions, so PyTorch SDPA's
                # `is_causal=True` (top-left aligned `tril`) is exactly the standard causal mask.
                out = F.scaled_dot_product_attention(
                    q, k, v, scale=scale, is_causal=True, enable_gqa=enable_gqa
                )
            else:
                # KV-cache decoding: q covers only the newest `q_len` positions while k/v
                # cover the full cached prefix of length `kv_len` (`kv_len > q_len`).
                # `is_causal=True` would apply a top-left aligned mask here, which is wrong:
                # it would forbid the new tokens from attending to most of the cached past.
                # We need the "bottom-right aligned" causal mask instead, where query row i
                # (i.e. absolute position `kv_len - q_len + i`) may attend to key positions
                # `[0, kv_len - q_len + i]`. See `src/model/kv_cache.py` for the cache this
                # supports.
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=scale,
                    attn_mask=causal_lower_right(q_len, kv_len),
                    enable_gqa=enable_gqa,
                )

        if out.shape[-1] != v_head_dim:
            out = out[..., :v_head_dim]

        # Re-wrap as DTensor on the TP mesh so that downstream layers (e.g., the output projection wo with RowwiseParallel) receive correctly sharded DTensors.
        if tp_spec is not None:
            out = DTensor.from_local(out, tp_spec[0], tp_spec[1], run_check=False)

        return out
