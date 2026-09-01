"""DeepSeek-V3 transformer model definition: Multi-Head Latent Attention (MLA), transformer blocks, and the top-level model class."""

import math

import torch
from torch import nn

from src.model.args import DeepSeekV3ModelArgs
from src.model.kv_cache import MLAKVCache
from src.model.moe import FeedForward, MoE
from src.model.rope import apply_rotary_emb, precompute_freqs_cis
from src.model.sdpa import ScaledDotProductAttentionWrapper


class Attention(nn.Module):
    """Multi-Head Latent Attention (MLA) with low-rank Q/KV projections and rotary position embeddings.

    Supports incremental decoding via an `MLAKVCache` that stores only the compressed
    latent representation (shared across heads), either through the vanilla `forward`
    (which re-projects the cache through `wkv_b` each call) or, after calling
    `absorb_mla_weights`, the cheaper `forward_absorbed` path.
    """

    def __init__(self, layer_id: int, model_args: DeepSeekV3ModelArgs):
        """Build the attention projections and softmax scaling from the given model config."""
        super().__init__()

        self.layer_id = layer_id
        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.v_head_dim = model_args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
            self.wq_b = nn.Linear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
            )
        self.wkv_a = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)

        self.softmax_scale = self.qk_head_dim**-0.5
        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.inner_attention = ScaledDotProductAttentionWrapper()

        # Set by `absorb_mla_weights()`; enables the cheaper `forward_absorbed` decoding path.
        self.wq_abs: nn.Linear | None = None
        self.wo_abs: nn.Linear | None = None

    @property
    def is_absorbed(self) -> bool:
        """Whether `absorb_mla_weights` has been called, enabling `forward_absorbed`."""
        return self.wq_abs is not None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: MLAKVCache | None = None,
        start_pos: int = 0,
    ):
        """Vanilla (non-absorbed) MLA forward pass, optionally reading/writing a compressed latent KV cache.

        When `kv_cache` is provided, only the compressed latent (`kv_lora_rank`) and the
        decoupled rotary key (`qk_rope_head_dim`) for the new tokens in `x` are cached; the
        full cached prefix is then up-projected through `wkv_b` into per-head keys/values on
        every call. This trades extra per-step compute for the smaller "latent" cache
        footprint described in the DeepSeek-V2 paper. For a decoding path that avoids
        re-projecting the whole cached prefix at every step, see `forward_absorbed`.

        The KV cache is an inference-only optimization: it must never be used while the
        module is in training mode (`self.training`), since caching would silently make
        gradients (and thus distributed training with DP/FSDP/TP/PP/CP) incorrect.
        """
        if kv_cache is not None and self.training:
            raise RuntimeError(
                "MLAKVCache must not be used while the model is in training mode; "
                "call model.eval() first, or omit kv_cache during training."
            )
        batch_size, seq_len, _ = x.size()

        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))

        q = q.view(batch_size, seq_len, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_a(x)
        # latent: [batch_size, seq_len, kv_lora_rank]
        # k_pe: [batch_size, seq_len, qk_rope_head_dim]
        latent, k_pe = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # This is the latent that should be cached.
        latent = self.kv_norm(latent)

        # k_pe: [batch_size, seq_len, 1, qk_rope_head_dim]
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        if kv_cache is not None:
            # Cache only the compressed latent representation (shared across all heads),
            # then read back the full cached prefix (previous tokens + these new ones).
            latent, k_pe_flat = kv_cache.update(
                self.layer_id, start_pos, latent, k_pe.squeeze(2)
            )
            k_pe = k_pe_flat.unsqueeze(2)

        # kv: [batch_size, cached_len, n_heads * (qk_nope_head_dim + v_head_dim)]
        kv = self.wkv_b(latent)
        cached_len = kv.size(1)
        # kv: [batch_size, cached_len, n_heads, qk_nope_head_dim + v_head_dim]
        kv = kv.view(
            batch_size, cached_len, -1, self.qk_nope_head_dim + self.v_head_dim
        )
        # k_nope: [batch_size, cached_len, n_heads, qk_nope_head_dim]
        # v: [batch_size, cached_len, n_heads, v_head_dim]
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # k: (batch_size, cached_len, n_heads, qk_head_dim)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        # q: [batch_size, n_heads, seq_len, qk_head_dim]
        q = q.transpose(1, 2)
        # k: [batch_size, n_heads, cached_len, qk_head_dim]
        k = k.transpose(1, 2)
        # v: [batch_size, n_heads, cached_len, v_head_dim]
        v = v.transpose(1, 2)

        # Run attention as usual. When a cache is used, `q` covers only the newest
        # `seq_len` positions while `k`/`v` cover the full cached prefix (`cached_len`);
        # `inner_attention` (`ScaledDotProductAttentionWrapper`) detects this and applies
        # bottom-right aligned causal masking, which is exactly correct for append-only KV
        # caches.
        output = self.inner_attention(q, k, v, scale=self.softmax_scale)

        # Reshape and project output
        # output: [batch_size, seq_len, n_heads, v_head_dim]
        output = output.transpose(1, 2).contiguous()
        # merge all the heads as usual
        # output: [batch_size, seq_len, n_heads * v_head_dim]
        output = output.view(batch_size, seq_len, -1)
        # apply Wo as usual
        # returns [batch_size, seq_len, dim]
        return self.wo(output)

    @torch.no_grad()
    def absorb_mla_weights(self) -> None:
        """Fold the up-projection weights into the query/output projections (weight absorption) to enable the cheaper `forward_absorbed` inference path.

        Only supported when `q_lora_rank == 0` (no query LoRA): `forward_absorbed` applies
        `wq_abs` directly to the raw layer input `x`, bypassing `wq_a`/`q_norm`, which is only
        mathematically equivalent to the vanilla path when there is no query low-rank
        bottleneck to begin with.
        """
        if self.q_lora_rank != 0:
            raise NotImplementedError(
                "MLA weight absorption is only supported when q_lora_rank == 0."
            )

        n_heads = self.n_heads
        dim = self.dim
        qk_nope_head_dim = self.qk_nope_head_dim
        qk_rope_head_dim = self.qk_rope_head_dim
        v_head_dim = self.v_head_dim
        kv_lora_rank = self.kv_lora_rank

        device = self.wq.weight.device
        dtype = self.wq.weight.dtype

        wq = self.wq.weight.view(
            n_heads,
            qk_nope_head_dim + qk_rope_head_dim,
            dim,
        )
        # qk_nope: [n_heads, qk_nope_head_dim, dim]
        # qk_rope: [n_heads, qk_rope_head_dim, dim]
        wq_nope, wq_rope = torch.split(
            wq,
            [qk_nope_head_dim, qk_rope_head_dim],
            dim=1,
        )

        wkv_b = self.wkv_b.weight.view(
            n_heads,
            qk_nope_head_dim + v_head_dim,
            kv_lora_rank,
        )
        # w_uk: [n_heads, qk_nope_head_dim, kv_lora_rank]
        # w_uv: [n_heads, v_head_dim, kv_lora_rank]
        w_uk, w_uv = torch.split(
            wkv_b,
            [qk_nope_head_dim, v_head_dim],
            dim=1,
        )

        # [n_heads, kv_lora_rank, dim]
        wq_abs_nope = torch.bmm(
            w_uk.float().transpose(
                1, 2
            ),  # [n_heads, qk_nope_head_dim, kv_lora_rank] -> [n_heads, kv_lora_rank, qk_nope_head_dim]
            wq_nope.float(),  # [n_heads, qk_nope_head_dim, dim]
        ).to(dtype=dtype)

        wq_abs = torch.cat(
            [wq_abs_nope, wq_rope],
            dim=1,
        ).reshape(
            n_heads * (kv_lora_rank + qk_rope_head_dim),
            dim,
        )

        self.wq_abs = nn.Linear(
            dim,
            n_heads * (kv_lora_rank + qk_rope_head_dim),
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.wq_abs.weight.copy_(wq_abs)  # type: ignore
        self.wq_abs.requires_grad_(False)  # type: ignore

        # [dim, n_heads, v_head_dim] -> [n_heads, dim, v_head_dim]
        w_o = self.wo.weight.view(
            dim,
            n_heads,
            v_head_dim,
        ).permute(1, 0, 2)

        # [n_heads, dim, v_head_dim] @ [n_heads, v_head_dim, kv_lora_rank] -> [n_heads, dim, kv_lora_rank]
        w_o_abs_per_head = torch.bmm(
            w_o.float(),
            w_uv.float(),
        ).to(dtype=dtype)

        # [n_heads, dim, kv_lora_rank] -> [dim, n_heads, kv_lora_rank] -> [dim, n_heads * kv_lora_rank]
        w_o_abs = w_o_abs_per_head.permute(
            1,
            0,
            2,
        ).reshape(
            dim,
            n_heads * kv_lora_rank,
        )

        self.wo_abs = nn.Linear(
            n_heads * kv_lora_rank,
            dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.wo_abs.weight.copy_(w_o_abs)  # type: ignore
        self.wo_abs.requires_grad_(False)  # type: ignore

    def forward_absorbed(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: MLAKVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Forward pass using the weight-absorbed projections produced by `absorb_mla_weights`.

        When `kv_cache` is provided, attention is computed directly against the cached
        compressed latent + rotary key -- per-head keys/values are never materialized for
        cached (past) tokens, avoiding the repeated `wkv_b` up-projection that the vanilla
        `forward` path pays on every decoding step.

        As with `forward`, the KV cache is an inference-only optimization and must never be
        used while the module is in training mode.
        """
        if kv_cache is not None and self.training:
            raise RuntimeError(
                "MLAKVCache must not be used while the model is in training mode; "
                "call model.eval() first, or omit kv_cache during training."
            )
        assert self.wq_abs is not None
        assert self.wo_abs is not None

        batch_size, seq_len, _ = x.shape

        q = self.wq_abs(x)
        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )

        # q_nope: [batch_size, seq_len, n_heads, kv_lora_rank]
        # q_rope: [batch_size, seq_len, n_heads, qk_rope_head_dim]
        q_nope, q_rope = torch.split(
            q,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        q_rope = apply_rotary_emb(q_rope, freqs_cis)

        # [batch_size, seq_len, n_heads, kv_lora_rank + qk_rope_head_dim] -> [batch_size, n_heads, seq_len, kv_lora_rank + qk_rope_head_dim]
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)

        # latent_raw: [batch_size, seq_len, kv_lora_rank]
        # k_rope: [batch_size, seq_len, qk_rope_head_dim]
        latent_raw, k_rope = torch.split(
            self.wkv_a(
                x
            ),  # [batch_size, seq_len, dim] -> [batch_size, seq_len, kv_lora_rank + qk_rope_head_dim]
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )

        # This is the latent that should be cached.
        # latent: [batch_size, seq_len, kv_lora_rank]
        latent = self.kv_norm(latent_raw)

        # [batch_size, seq_len, 1, qk_rope_head_dim]
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis)

        if kv_cache is not None:
            # Cache only the compressed latent + decoupled rotary key (shared across all
            # heads), then read back the full cached prefix (previous tokens + these new
            # ones). This is the memory-saving property of MLA: no per-head K/V is stored.
            latent, k_rope_flat = kv_cache.update(
                self.layer_id, start_pos, latent, k_rope.squeeze(2)
            )
            k_rope = k_rope_flat.unsqueeze(2)

        # A single shared storage tensor:
        # shared_cache: [batch_size, cached_len, 1, kv_lora_rank + qk_rope_head_dim] -> [batch_size, 1, cached_len, kv_lora_rank + qk_rope_head_dim]
        shared_cache = torch.cat(
            [latent.unsqueeze(2), k_rope],
            dim=-1,
        ).transpose(1, 2)

        # k: [batch_size, 1, cached_len, kv_lora_rank + qk_rope_head_dim]
        k = shared_cache

        # v: [batch_size, 1, cached_len, kv_lora_rank]
        v = shared_cache[..., : self.kv_lora_rank]

        # latent_output: [batch_size, n_heads, seq_len, kv_lora_rank]
        # `q` covers only the newest `seq_len` positions while `k`/`v` cover the full
        # cached prefix (`cached_len`); see the note on bottom-right causal alignment in
        # `ScaledDotProductAttentionWrapper`.
        latent_output = self.inner_attention(q, k, v, scale=self.softmax_scale)

        # [batch_size, seq_len, n_heads * kv_lora_rank]
        latent_output = (
            latent_output.transpose(
                1, 2
            )  # [batch_size, n_heads, seq_len, kv_lora_rank] -> [batch_size, seq_len, n_heads, kv_lora_rank]
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.n_heads * self.kv_lora_rank,
            )
        )

        # output: [batch_size, seq_len, dim]
        return self.wo_abs(latent_output)

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device | None = None,
    ):
        """Initialize attention projection weights and reset RMSNorm parameters."""
        linear_list = [
            self.wkv_a,
            self.wkv_b,
        ]
        if self.q_lora_rank > 0:
            linear_list.extend([self.wq_a, self.wq_b])
        else:
            linear_list.append(self.wq)

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        self.kv_norm.reset_parameters()
        if self.q_lora_rank > 0:
            self.q_norm.reset_parameters()


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, model_args: DeepSeekV3ModelArgs):
        """Build the attention and (dense or MoE) feed-forward sublayers for one transformer layer."""
        super().__init__()
        self.attention = Attention(layer_id, model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.moe_enabled = layer_id >= model_args.n_dense_layers
        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(model_args.dim, model_args.inter_dim)

        self.weight_init_std: float = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: MLAKVCache | None = None,
        start_pos: int = 0,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            kv_cache (MLAKVCache, optional): Latent KV cache for incremental decoding.
            start_pos (int): Position of the first token in `x` within the full sequence.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        attention_fn = (
            self.attention.forward_absorbed
            if self.attention.is_absorbed
            else self.attention.forward
        )
        x = x + attention_fn(
            self.attention_norm(x), freqs_cis, kv_cache=kv_cache, start_pos=start_pos
        )
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(
        self,
        init_std: float | None = None,
        buffer_device: torch.device | None = None,
    ):
        """Reset norms and initialize the attention and feed-forward/MoE sublayer weights."""
        if buffer_device is None:
            raise ValueError(
                "buffer_device must be provided for TransformerBlock weight initialization"
            )
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()

        std = init_std or self.weight_init_std
        self.attention.init_weights(std)
        if self.moe_enabled:
            self.moe.init_weights(init_std=std, buffer_device=buffer_device)
        else:
            self.feed_forward.init_weights(std)


class DeepSeekV3Model(nn.Module):
    """DeepSeek-V3 decoder-only transformer: token embedding, stacked transformer blocks, and an output projection."""

    def __init__(self, model_args: DeepSeekV3ModelArgs):
        """Build the embedding, transformer layers, final norm, and output projection."""
        super().__init__()
        self.model_args = model_args
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args), persistent=False
        )

        self.layers = nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )

    def init_weights(
        self,
        init_std: float | None = None,
        buffer_device: torch.device | None = None,
    ):
        """Initialize all model weights and recompute the rotary embedding buffer on `buffer_device`."""
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)

        nn.init.normal_(self.tok_embeddings.weight)

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(init_std=init_std, buffer_device=buffer_device)  # type: ignore

        self.norm.reset_parameters()

        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        kv_cache: MLAKVCache | None = None,
        start_pos: int = 0,
    ):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices for the ranks on the first pipeline stage. This will be the activation of the previous pipeline stage if the current rank is not on the first stage.
            kv_cache (MLAKVCache, optional): Latent KV cache for incremental decoding, built via
                `build_kv_cache`. When provided, `tokens` should contain only the new tokens
                starting at `start_pos` (e.g. a single token per decoding step).
            start_pos (int): Position of the first token in `tokens` within the full sequence.
                Used to select the matching rotary embeddings and to index into `kv_cache`.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        if kv_cache is not None and self.training:
            raise RuntimeError(
                "MLAKVCache must not be used while the model is in training mode; "
                "call model.eval() first, or omit kv_cache during training."
            )

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        seq_len = h.size(1)
        # During training seq_len always equals model_args.max_seq_len (see train.py), so
        # this slice is a no-op there; it only takes effect for incremental decoding, where
        # `tokens` covers just the newest positions starting at `start_pos`.
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        for layer in self.layers.values():
            h = layer(h, freqs_cis, kv_cache=kv_cache, start_pos=start_pos)
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output

    def build_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> MLAKVCache:
        """Allocate an `MLAKVCache` sized for this model's MLA latent dimensions, for incremental-decoding inference.

        Inference-only: the returned cache must only be passed to `forward` (or `Attention`)
        while the model is in eval mode (`model.eval()`). Passing it to a model still in
        training mode raises `RuntimeError`, since caching activations would make gradients
        (and thus distributed training under DP/FSDP/TP/PP/CP) incorrect. Also prefer running
        inference on a separate copy of the model (e.g. loaded from a checkpoint) rather than
        the live training instance, so switching train/eval modes never races with concurrent
        training steps.
        """
        param = next(self.parameters())
        return MLAKVCache(
            n_layers=self.model_args.n_layers,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            kv_lora_rank=self.model_args.kv_lora_rank,
            qk_rope_head_dim=self.model_args.qk_rope_head_dim,
            dtype=dtype or param.dtype,
            device=device or param.device,
        )

    @torch.no_grad()
    def absorb_mla_weights(self) -> None:
        """Apply MLA weight absorption to every layer's attention (see `Attention.absorb_mla_weights`).

        Inference-only: this permanently rewrites each attention layer to require
        `forward_absorbed` (and thus a KV cache) going forward, so it must not be called on
        a model that will continue training. Apply it only to a dedicated inference copy of
        the model (e.g. loaded from a checkpoint), never to the live `Trainer.model_parts`.

        After calling this, layers automatically use the cheaper `forward_absorbed` decoding
        path (see `TransformerBlock.forward`), which pairs with `build_kv_cache` to attend
        directly against the compressed latent cache without re-materializing per-head K/V
        for previously cached tokens.
        """
        for layer in self.layers.values():
            layer.attention.absorb_mla_weights()
