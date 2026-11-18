"""
model_scale.py - Modern Scalable Transformer Architecture (15M -> 3B)
Features:
  - Rotary Position Embeddings (RoPE)
  - RMSNorm for fast, stable normalization
  - SwiGLU feed-forward activations
  - Grouped-Query Attention (GQA) for 1B+ parameter models
  - Dynamic CPU quantization (Int8) for running locally on Intel Iris Xe / CPU
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ScaledConfig:
    vocab_size: int = 8192
    block_size: int = 1024
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: Optional[int] = None  # None = standard MHA; int = Grouped Query Attention (GQA)
    hidden_dim: Optional[int] = None  # SwiGLU hidden dim (default: 4 * dim or 8/3 * dim)
    norm_eps: float = 1e-5


def get_preset_config(scale: str, vocab_size: int = 8192, block_size: int = 1024) -> ScaledConfig:
    presets = {
        "15m": ScaledConfig(vocab_size=vocab_size, block_size=block_size, dim=384, n_layers=6, n_heads=6),
        "125m": ScaledConfig(vocab_size=vocab_size, block_size=block_size, dim=768, n_layers=12, n_heads=12),
        "500m": ScaledConfig(vocab_size=vocab_size, block_size=block_size, dim=1024, n_layers=24, n_heads=16, n_kv_heads=4),
        "1b": ScaledConfig(vocab_size=vocab_size, block_size=2048, dim=2048, n_layers=22, n_heads=16, n_kv_heads=4),
        "3b": ScaledConfig(vocab_size=vocab_size, block_size=2048, dim=3072, n_layers=28, n_heads=24, n_kv_heads=8),
    }
    if scale.lower() not in presets:
        raise ValueError(f"Unknown preset scale: {scale}. Choose from: {list(presets.keys())}")
    return presets[scale.lower()]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def precompute_rope_frequencies(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reshape for complex multiplication
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[1]].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, config: ScaledConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape
        xq = self.wq(x).view(b, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(b, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(b, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rope(xq, xk, freqs_cis)

        # Expand KV heads for Grouped Query Attention if needed
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # PyTorch FlashAttention / SDPA
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(b, seq_len, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network (Modern LLaMA/Mistral style)"""
    def __init__(self, config: ScaledConfig):
        super().__init__()
        hidden_dim = config.hidden_dim or int(2 * (4 * config.dim) / 3)
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ScaledConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class ScaledTransformer(nn.Module):
    def __init__(self, config: ScaledConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight  # Weight tying

        self.freqs_cis = precompute_rope_frequencies(config.dim // config.n_heads, config.block_size * 2)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        device = tokens.device
        _, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(device)

        for layer in self.layers:
            h = layer(h, freqs_cis)

        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.output(h[:, [-1], :])
            return logits, None

    def quantize_for_cpu(self):
        """Quantize model linear layers to int8 for fast execution on laptop CPU / Iris Xe"""
        print("Applying dynamic int8 quantization for ultra-fast local CPU inference...")
        return torch.ao.quantization.quantize_dynamic(
            self, {nn.Linear}, dtype=torch.qint8
        )
