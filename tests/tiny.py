"""A tiny random Qwen3 checkpoint in the exact on-disk format of the real one."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def make_tiny(path: Path, layers: int = 2, hidden: int = 256, heads: int = 8, kv_heads: int = 2,
              head_dim: int = 128, intermediate: int = 512, vocab: int = 1024, seed: int = 0) -> Path:
    if (path / "config.json").exists():
        return path
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim,
        vocab_size=vocab, rms_norm_eps=1e-6, rope_theta=5_000_000, tie_word_embeddings=True,
        max_position_embeddings=8192, torch_dtype=torch.bfloat16, use_sliding_window=False,
        sliding_window=None, attention_bias=False, hidden_act="silu",
    )
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn_like(p, dtype=torch.float32))
            else:
                p.copy_(torch.randn_like(p, dtype=torch.float32) * 0.05)
    model.save_pretrained(path, safe_serialization=True)
    return path


def load_reference(path: Path):
    return (
        Qwen3ForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
        .eval()
        .to("cuda:0")
    )
