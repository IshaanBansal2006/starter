"""Qwen3 weights and forward passes over static buffers.

Everything here is written so that a whole prefill or decode step touches only
preallocated tensors and launches no host-synchronising op, which is what lets
``engine.py`` capture each step into a CUDA graph and replay it.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from kernels import DecodeAttention, add_rms_norm, pick_matmul, qk_norm_rope_cache, rms_norm, swiglu


@dataclass(frozen=True)
class Config:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    eps: float
    rope_theta: float
    tie_embeddings: bool

    @classmethod
    def load(cls, model_path: Path) -> "Config":
        raw = json.loads((model_path / "config.json").read_text())
        return cls(
            hidden=raw["hidden_size"],
            intermediate=raw["intermediate_size"],
            layers=raw["num_hidden_layers"],
            heads=raw["num_attention_heads"],
            kv_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"],
            vocab=raw["vocab_size"],
            eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            tie_embeddings=raw.get("tie_word_embeddings", False),
        )


@dataclass
class Layer:
    in_norm: torch.Tensor
    wqkv: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    wo: torch.Tensor
    post_norm: torch.Tensor
    wgu: torch.Tensor
    wd: torch.Tensor


def _read_all(model_path: Path, device) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device=str(device)) as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name)
    if not tensors:
        raise FileNotFoundError(f"no *.safetensors under {model_path}; the checkpoint must be there")
    return tensors


def rope_tables(cfg: Config, max_pos: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin in bf16 exactly as ``Qwen3RotaryEmbedding.forward`` builds them.

    ``inv_freq`` is fp32, ``freqs`` is an fp32 matmul with TF32 disabled, the
    table is ``cat(freqs, freqs)`` so column ``i`` and ``i + D/2`` share an
    angle, and the cast to bf16 happens after cos/sin.
    """
    D = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2, dtype=torch.int64, device=device).float() / D))
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    freqs = (inv_freq[:, None].float() @ positions[None, :].float()).transpose(0, 1)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16).contiguous(), emb.sin().to(torch.bfloat16).contiguous()


class Model:
    def __init__(self, model_path: str | Path, device="cuda:0", max_pos: int = 4096):
        model_path = Path(model_path)
        self.cfg = Config.load(model_path)
        self.device = torch.device(device)
        raw = _read_all(model_path, self.device)
        self.embed = raw.pop("model.embed_tokens.weight")
        self.lm_head = raw.pop("lm_head.weight", None)
        if self.lm_head is None:
            if not self.cfg.tie_embeddings:
                raise KeyError("lm_head.weight missing and embeddings are not tied")
            self.lm_head = self.embed
        self.final_norm = raw.pop("model.norm.weight")
        self.layers: list[Layer] = []
        for i in range(self.cfg.layers):
            p = f"model.layers.{i}."
            wqkv = torch.cat(
                [raw.pop(p + "self_attn.q_proj.weight"), raw.pop(p + "self_attn.k_proj.weight"), raw.pop(p + "self_attn.v_proj.weight")],
                dim=0,
            )
            wgu = torch.cat([raw.pop(p + "mlp.gate_proj.weight"), raw.pop(p + "mlp.up_proj.weight")], dim=0)
            self.layers.append(Layer(
                in_norm=raw.pop(p + "input_layernorm.weight"),
                wqkv=wqkv,
                q_norm=raw.pop(p + "self_attn.q_norm.weight"),
                k_norm=raw.pop(p + "self_attn.k_norm.weight"),
                wo=raw.pop(p + "self_attn.o_proj.weight"),
                post_norm=raw.pop(p + "post_attention_layernorm.weight"),
                wgu=wgu,
                wd=raw.pop(p + "mlp.down_proj.weight"),
            ))
        if raw:
            raise KeyError(f"unexpected tensors in checkpoint: {sorted(raw)[:5]}")
        self.cos, self.sin = rope_tables(self.cfg, max_pos, self.device)
        self.scale = 1.0 / math.sqrt(self.cfg.head_dim)

    def ensure_rope(self, max_pos: int) -> None:
        if self.cos.shape[0] < max_pos:
            self.cos, self.sin = rope_tables(self.cfg, max_pos, self.device)


class Plan:
    """Static buffers, KV cache and step functions for one (B, T, max_new) shape."""

    def __init__(self, model: Model, B: int, T: int, max_new: int):
        self.model = model
        cfg = model.cfg
        self.B, self.T, self.max_new = B, T, max_new
        self.cap = T + max_new
        model.ensure_rope(self.cap)
        dev = model.device
        bf16 = torch.bfloat16
        HQ, HKV, D = cfg.heads, cfg.kv_heads, cfg.head_dim
        self.ids = torch.zeros((B, T), dtype=torch.int64, device=dev)
        self.tok = torch.zeros((B,), dtype=torch.int64, device=dev)
        self.pos = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.k_cache = torch.zeros((cfg.layers, B, HKV, self.cap, D), dtype=bf16, device=dev)
        self.v_cache = torch.zeros((cfg.layers, B, HKV, self.cap, D), dtype=bf16, device=dev)
        self.q_prefill = torch.empty((B, HQ, T, D), dtype=bf16, device=dev)
        self.q_decode = torch.empty((B, HQ, 1, D), dtype=bf16, device=dev)
        self.attn_decode = torch.empty((B, HQ, D), dtype=bf16, device=dev)
        self.attention = DecodeAttention(B, HQ, HKV, D, self.cap, dev)
        self.mm = self._pick_decode_matmuls()

    def _pick_decode_matmuls(self) -> dict[str, object]:
        """Time cuBLAS against the Triton skinny GEMM for every decode shape."""
        m, cfg, B = self.model, self.model.cfg, self.B
        layer = m.layers[0]
        x = torch.randn((B, cfg.hidden), dtype=torch.bfloat16, device=m.device)
        a = torch.randn((B, cfg.heads * cfg.head_dim), dtype=torch.bfloat16, device=m.device)
        act = torch.randn((B, cfg.intermediate), dtype=torch.bfloat16, device=m.device)
        log = lambda s: print(f"[engine] {s}", file=sys.stderr, flush=True)
        return {
            "qkv": pick_matmul(x, layer.wqkv, log),
            "o": pick_matmul(a, layer.wo, log),
            "gu": pick_matmul(x, layer.wgu, log),
            "d": pick_matmul(act, layer.wd, log),
            "lm": pick_matmul(x, m.lm_head, log),
        }

    def _next_norm(self, i: int) -> torch.Tensor:
        layers = self.model.layers
        return layers[i + 1].in_norm if i + 1 < len(layers) else self.model.final_norm

    @torch.inference_mode()
    def prefill(self) -> torch.Tensor:
        """Consume ``self.ids`` from position 0; returns [B, V] logits at the last position."""
        m, cfg = self.model, self.model.cfg
        B, T = self.B, self.T
        HQ, D = cfg.heads, cfg.head_dim
        self.pos.zero_()
        x = F.embedding(self.ids.view(-1), m.embed)
        h = rms_norm(x, m.layers[0].in_norm, cfg.eps)
        for i, layer in enumerate(m.layers):
            qkv = h @ layer.wqkv.t()
            qk_norm_rope_cache(qkv, layer.q_norm, layer.k_norm, m.cos, m.sin, self.pos,
                               self.q_prefill, self.k_cache[i], self.v_cache[i], T, cfg.eps)
            a = F.scaled_dot_product_attention(
                self.q_prefill, self.k_cache[i, :, :, :T], self.v_cache[i, :, :, :T],
                is_causal=True, scale=m.scale, enable_gqa=True,
            )
            o = a.transpose(1, 2).reshape(B * T, HQ * D) @ layer.wo.t()
            h2 = add_rms_norm(x, o, layer.post_norm, cfg.eps)
            d = swiglu(h2 @ layer.wgu.t()) @ layer.wd.t()
            h = add_rms_norm(x, d, self._next_norm(i), cfg.eps)
        logits = h.view(B, T, cfg.hidden)[:, -1] @ m.lm_head.t()
        self.pos.fill_(T)
        return logits

    @torch.inference_mode()
    def decode(self) -> torch.Tensor:
        """Consume ``self.tok`` at ``self.pos``; returns [B, V] logits; advances ``pos``."""
        m, cfg = self.model, self.model.cfg
        B = self.B
        HQ, D = cfg.heads, cfg.head_dim
        mm = self.mm
        x = F.embedding(self.tok, m.embed)
        h = rms_norm(x, m.layers[0].in_norm, cfg.eps)
        for i, layer in enumerate(m.layers):
            qkv = mm["qkv"](h, layer.wqkv)
            qk_norm_rope_cache(qkv, layer.q_norm, layer.k_norm, m.cos, m.sin, self.pos,
                               self.q_decode, self.k_cache[i], self.v_cache[i], 1, cfg.eps)
            self.attention(self.q_decode.view(B, HQ, D), self.k_cache[i], self.v_cache[i], self.pos, self.attn_decode)
            o = mm["o"](self.attn_decode.view(B, HQ * D), layer.wo)
            h2 = add_rms_norm(x, o, layer.post_norm, cfg.eps)
            d = mm["d"](swiglu(mm["gu"](h2, layer.wgu)), layer.wd)
            h = add_rms_norm(x, d, self._next_norm(i), cfg.eps)
        logits = mm["lm"](h, m.lm_head)
        self.pos.add_(1)
        return logits
