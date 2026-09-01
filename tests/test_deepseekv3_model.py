"""Small end-to-end tests for `DeepSeekV3Model`: training-mode forward pass and eval-mode KV-cache decoding."""

import torch

from src.model.args import DeepSeekV3ModelArgs
from src.model.model import DeepSeekV3Model


def _tiny_model_args() -> DeepSeekV3ModelArgs:
    return DeepSeekV3ModelArgs(
        max_seq_len=16,
        vocab_size=32,
        dim=16,
        inter_dim=32,
        moe_inter_dim=16,
        n_layers=2,
        n_dense_layers=1,  # layer 0 dense, layer 1 MoE
        n_heads=2,
        q_lora_rank=0,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        original_seq_len=16,
    )


def _build_and_init_model() -> DeepSeekV3Model:
    torch.manual_seed(0)
    model_args = _tiny_model_args()
    model = DeepSeekV3Model(model_args)
    model.init_weights(buffer_device=torch.device("cpu"))
    return model


def test_model_forward_training_mode_produces_logits():
    """A default (training-mode) forward pass should produce (batch, seq, vocab) logits."""
    model = _build_and_init_model()
    model.train()

    batch_size, seq_len = 2, 5
    tokens = torch.randint(0, model.model_args.vocab_size, (batch_size, seq_len))
    logits = model(tokens)

    assert logits.shape == (batch_size, seq_len, model.model_args.vocab_size)


def test_model_forward_rejects_kv_cache_in_training_mode():
    """Passing a kv_cache while the model is in training mode must raise, protecting the training loop."""
    model = _build_and_init_model()
    model.train()
    cache = model.build_kv_cache(max_batch_size=1, max_seq_len=4)

    tokens = torch.randint(0, model.model_args.vocab_size, (1, 1))
    try:
        model(tokens, kv_cache=cache, start_pos=0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError when using kv_cache in training mode")


def test_model_eval_incremental_decode_matches_full_forward():
    """Token-by-token decoding with a KV cache in eval mode should match the equivalent full-sequence forward pass."""
    model = _build_and_init_model()
    model.eval()

    batch_size, seq_len = 1, 4
    tokens = torch.randint(0, model.model_args.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits_full = model(tokens)

        cache = model.build_kv_cache(max_batch_size=batch_size, max_seq_len=seq_len)
        logits_chunks = []
        for t in range(seq_len):
            logits_t = model(tokens[:, t : t + 1], kv_cache=cache, start_pos=t)
            logits_chunks.append(logits_t)
        logits_incremental = torch.cat(logits_chunks, dim=1)

    assert torch.allclose(logits_full, logits_incremental, atol=1e-4, rtol=1e-4)
