"""Acoustic model in the style of PiVo (text -> 20 Opus features every 10 ms).

Architecture rebuilt from the PiVo-1.4M technical report and ONNX graphs (shapes and operators):
  * encoder: token + voice embeddings; 6 blocks alternating GQA attention (RoPE, RMSNorm, gated-SiLU FFN)
    and dilated temporal convolution (k=5, dilations 1,2,4); final RMSNorm.
  * duration predictor (log1p of frames per token): 2 depthwise k=3 layers + pointwise.
  * 4 phase signals per frame (progress, sin, cos, log-duration) enter through a projection.
  * decoder: 2 causal attention blocks with a 64-frame window (RoPE only in the first) -> 20 features.
The code is ours; only the design and dimensions follow PiVo (Apache-2.0).
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 128
    voices: int = 2
    width: int = 96
    ff: int = 192
    heads: int = 4
    kv_heads: int = 2
    text_layers: int = 6
    decoder_layers: int = 2
    kernel: int = 5
    dur_width: int = 64
    window: int = 64
    feature_dim: int = 20
    max_token_frames: int = 200
    pad: int = 0

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(n)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope(x, pos):
    """x: (B,H,T,D), pos: (T,) absolute positions."""
    d = x.shape[-1] // 2
    inv = 1.0 / (10000 ** (torch.arange(d, device=x.device, dtype=torch.float32) / d))
    ang = pos.float()[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :d], x[..., d:]
    return torch.cat((a * cos - b * sin, a * sin + b * cos), -1)


class Attention(nn.Module):
    def __init__(self, c, use_rope=True):
        super().__init__()
        self.h, self.kv, self.hd = c.heads, c.kv_heads, c.width // c.heads
        self.q = nn.Linear(c.width, c.heads * self.hd, bias=False)
        self.k = nn.Linear(c.width, c.kv_heads * self.hd, bias=False)
        self.v = nn.Linear(c.width, c.kv_heads * self.hd, bias=False)
        self.o = nn.Linear(c.heads * self.hd, c.width, bias=False)
        self.use_rope = use_rope

    def forward(self, x, mask):
        """mask: (B or 1, 1, T, T) booleano, True = pode atender."""
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv, self.hd).transpose(1, 2)
        if self.use_rope:
            pos = torch.arange(t, device=x.device)
            q, k = rope(q, pos), rope(k, pos)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(b, t, -1))


class GatedFFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Linear(c.width, c.ff, bias=False)
        self.up = nn.Linear(c.width, c.ff, bias=False)
        self.down = nn.Linear(c.ff, c.width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class AttnBlock(nn.Module):
    def __init__(self, c, use_rope=True):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.width), RMSNorm(c.width)
        self.attention, self.ffn = Attention(c, use_rope), GatedFFN(c)

    def forward(self, x, mask):
        x = x + self.attention(self.norm1(x), mask)
        return x + self.ffn(self.norm2(x))


class ConvBlock(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.width), RMSNorm(c.width)
        pad = dilation * (c.kernel - 1) // 2
        self.conv = nn.Conv1d(c.width, c.width, c.kernel, padding=pad, dilation=dilation, bias=False)

    def forward(self, x, keep):
        h = self.norm1(x) * keep
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        return x + F.silu(self.norm2(h))


class DurationPredictor(nn.Module):
    def __init__(self, c):
        super().__init__()
        w = c.dur_width
        self.in_proj = nn.Linear(c.width, w)
        self.dw = nn.ModuleList([nn.Conv1d(w, w, 3, padding=1, groups=w) for _ in range(2)])
        self.pw = nn.ModuleList([nn.Conv1d(w, w, 1) for _ in range(2)])
        self.norms = nn.ModuleList([RMSNorm(w) for _ in range(2)])
        self.out = nn.Linear(w, 1)

    def forward(self, x, keep):
        h = F.silu(self.in_proj(x)) * keep
        for dw, pw, n in zip(self.dw, self.pw, self.norms):
            y = pw(dw((n(h) * keep).transpose(1, 2))).transpose(1, 2)
            h = (h + F.silu(y)) * keep
        return self.out(h).squeeze(-1)


def frame_index(durations):
    """For each frame, the index of the token that generates it. -> idx (B,F), mask (B,F), total count (B,)."""
    total = durations.sum(1)
    n = int(total.max())
    ends = torch.cumsum(durations, 1)                                    # (B,L)
    frames = torch.arange(n, device=durations.device)
    idx = (frames[None, :, None] >= ends[:, None, :]).sum(-1).clamp(max=durations.shape[1] - 1)
    return idx, frames[None] < total[:, None], ends - durations


def phase_features(durations, max_token_frames=200):
    """Per-frame phase signals. durations: (B,L) int -> (B,F,4), mask (B,F)."""
    idx, mask, starts = frame_index(durations)
    counts = durations.gather(1, idx).clamp(min=1).float()
    st = starts.gather(1, idx).float()
    frac = (torch.arange(idx.shape[1], device=idx.device)[None].float() - st + 0.5) / counts
    out = torch.stack((2 * frac - 1, torch.sin(math.pi * frac), torch.cos(math.pi * frac),
                       torch.log1p(counts) / math.log1p(max_token_frames)), -1)
    return out * mask.unsqueeze(-1), mask


def expand(encoded, durations):
    """Repeat each token encoding for its duration. -> (B,F,W), mask (B,F)."""
    idx, mask, _ = frame_index(durations)
    out = encoded.gather(1, idx.unsqueeze(-1).expand(-1, -1, encoded.shape[-1]))
    return out * mask.unsqueeze(-1), mask


class Acoustic(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.tokens = nn.Embedding(c.vocab, c.width, padding_idx=c.pad)
        self.voices = nn.Embedding(c.voices, c.width)
        self.stress = nn.Embedding(2, c.width)
        self.text_blocks = nn.ModuleList(
            [AttnBlock(c) if i % 2 == 0 else ConvBlock(c, 2 ** (i // 2)) for i in range(c.text_layers)])
        self.text_norm = RMSNorm(c.width)
        self.durations = DurationPredictor(c)
        self.phase = nn.Linear(4, c.width, bias=False)
        self.decoder = nn.ModuleList([AttnBlock(c, use_rope=(i == 0)) for i in range(c.decoder_layers)])
        self.feature_norm = RMSNorm(c.width)
        self.head = nn.Linear(c.width, c.feature_dim, bias=False)
        # feature statistics: the head predicts normalized values; the export folds mean/std into the weights
        self.register_buffer("f_mean", torch.zeros(c.feature_dim))
        self.register_buffer("f_std", torch.ones(c.feature_dim))

    def encode(self, tokens, stress, voice, lengths):
        b, t = tokens.shape
        keep = (torch.arange(t, device=tokens.device)[None] < lengths[:, None]).unsqueeze(-1).float()
        x = (self.tokens(tokens) + self.stress(stress) + self.voices(voice)[:, None]) * keep
        amask = (keep.squeeze(-1) > 0)[:, None, None, :].expand(b, 1, t, t)
        for blk in self.text_blocks:
            x = blk(x, amask) * keep if isinstance(blk, AttnBlock) else blk(x, keep) * keep
        x = self.text_norm(x) * keep
        return x, self.durations(x, keep)            # predicted log1p(duration)

    def decode(self, expanded, phase, fmask):
        """expanded (B,F,W), phase (B,F,4), fmask (B,F) -> (B,F,20)."""
        b, n, _ = expanded.shape
        i = torch.arange(n, device=expanded.device)
        causal = (i[None] <= i[:, None]) & (i[:, None] - i[None] < self.c.window)
        mask = (causal[None, None] & fmask[:, None, None, :]) | torch.eye(n, dtype=torch.bool, device=expanded.device)[None, None]
        # (keeping the diagonal open avoids fully masked rows in padding frames, which give NaN in CUDA SDPA)
        x = expanded + self.phase(phase)
        for blk in self.decoder:
            x = blk(x, mask)
        return self.head(self.feature_norm(x))

    def forward(self, tokens, stress, voice, lengths, durations):
        enc, log_dur = self.encode(tokens, stress, voice, lengths)
        expanded, fmask = expand(enc, durations)
        phase, _ = phase_features(durations, self.c.max_token_frames)
        return self.decode(expanded, phase, fmask), log_dur, fmask


if __name__ == "__main__":
    for w, ff in ((96, 192), (128, 256)):
        m = Acoustic(Config(width=w, ff=ff))
        n = sum(p.numel() for p in m.parameters())
        tok = torch.randint(1, 60, (2, 12)); lens = torch.tensor([12, 9]); voice = torch.tensor([0, 1])
        dur = torch.randint(1, 6, (2, 12)); dur[1, 9:] = 0
        feats, log_dur, fmask = m(tok, torch.zeros_like(tok), voice, lens, dur)
        loss = feats.pow(2).mean() + log_dur.pow(2).mean(); loss.backward()
        print(f"width {w}: {n:,} parameters | features {tuple(feats.shape)} | log_dur {tuple(log_dur.shape)}")
