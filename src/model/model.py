
import math

import torch
from src.model.attention import ScaledDotProductAttentionWrapper  # type: ignore
from src.model.moe import FeedForward, MoE  # type: ignore
from torch import nn

from src.model.args import DeepSeekV3ModelArgs
from src.model.rope import apply_rotary_emb, precompute_freqs_cis


class Attention(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()

        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank  # 0
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = (
            model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        )
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


    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
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
        # kv: [batch_size, seq_len, kv_lora_rank]
        # k_pe: [batch_size, seq_len, qk_rope_head_dim]
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # k_pe: [batch_size, seq_len, 1, qk_rope_head_dim]
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        # kv: [batch_size, seq_len, n_heads * (qk_nope_head_dim + v_head_dim)]
        kv = self.wkv_b(self.kv_norm(kv))
         # kv: [batch_size, seq_len, n_heads * (qk_nope_head_dim + v_head_dim)]
        kv = self.wkv_b(self.kv_norm(kv))
        # kv: [batch_size, seq_len, n_heads, qk_nope_head_dim + v_head_dim]
        kv = kv.view(batch_size, seq_len, -1, self.qk_nope_head_dim + self.v_head_dim)
        # k_nope: [batch_size, seq_len, n_heads, qk_nope_head_dim]
        # v: [batch_size, seq_len, n_heads, v_head_dim]
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # k: (batch_size, seq_len, n_heads, qk_head_dim)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        # q: [batch_size, n_heads, seq_len, qk_head_dim]
        q = q.transpose(1, 2)
        # k: [batch_size, n_heads, seq_len, qk_head_dim]
        k = k.transpose(1, 2)
        # v: [batch_size, n_heads, seq_len, v_head_dim]
        v = v.transpose(1, 2)

        # run attention as usual
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



    

    def init_weights(self, init_std: float):
        pass


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.attention = Attention(model_args)
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

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        x = x + self.attention(self.attention_norm(x), freqs_cis)
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
        if buffer_device is None:
            raise ValueError(
                "buffer_device must be provided for TransformerBlock weight initialization"
            )
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()

        std = init_std or self.weight_init_std
        self.attention.init_weights(std)
        if self.moe_enabled:
            self.moe.init_weights(
                init_std=std, buffer_device=buffer_device
            )
        else:
            self.feed_forward.init_weights(std)


class DeepSeekV3Model(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
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
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)
        
        nn.init.normal_(self.tok_embeddings.weight)

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(init_std=init_std, buffer_device=buffer_device) # type: ignore
        
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

    def forward(self, tokens: torch.Tensor):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices for the ranks on the first pipeline stage. This will be the activation of the previous pipeline stage if the current rank is not on the first stage.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """

        h = self.tok_embeddings(tokens)

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h)
        output = self.output(h)
        return output