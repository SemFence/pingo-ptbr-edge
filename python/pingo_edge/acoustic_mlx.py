"""Pingo acoustic model in MLX (copy of the acoustic part of semfence/pingo-ptbr: pingo_mlx.py), used with the edge 4-bit weights."""
import json, math
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def rope(x, pos):
    d = x.shape[-1] // 2
    inv = 1.0 / (10000 ** (mx.arange(d, dtype=mx.float32) / d))
    ang = pos.astype(mx.float32)[:, None] * inv[None]
    cos, sin = mx.cos(ang), mx.sin(ang)
    a, b = x[..., :d], x[..., d:]
    return mx.concatenate([a * cos - b * sin, a * sin + b * cos], -1)


class RMSNorm(nn.Module):
    def __init__(self, n):
        super().__init__(); self.weight = mx.ones((n,))

    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(x * x, -1, keepdims=True) + 1e-6) * self.weight


class Attention(nn.Module):
    def __init__(self, c, use_rope=True):
        super().__init__()
        self.h, self.kv, self.hd, self.use_rope = c["heads"], c["kv_heads"], c["width"] // c["heads"], use_rope
        w = c["width"]
        self.q = nn.Linear(w, self.h * self.hd, bias=False); self.k = nn.Linear(w, self.kv * self.hd, bias=False)
        self.v = nn.Linear(w, self.kv * self.hd, bias=False); self.o = nn.Linear(self.h * self.hd, w, bias=False)

    def __call__(self, x, mask):
        b, t, _ = x.shape
        q = self.q(x).reshape(b, t, self.h, self.hd).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(b, t, self.kv, self.hd).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(b, t, self.kv, self.hd).transpose(0, 2, 1, 3)
        if self.use_rope:
            pos = mx.arange(t); q, k = rope(q, pos), rope(k, pos)
        rep = self.h // self.kv
        k, v = mx.repeat(k, rep, 1), mx.repeat(v, rep, 1)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask=mask)
        return self.o(y.transpose(0, 2, 1, 3).reshape(b, t, -1))


class GatedFFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Linear(c["width"], c["ff"], bias=False); self.up = nn.Linear(c["width"], c["ff"], bias=False)
        self.down = nn.Linear(c["ff"], c["width"], bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class AttnBlock(nn.Module):
    def __init__(self, c, use_rope=True):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c["width"]), RMSNorm(c["width"])
        self.attention, self.ffn = Attention(c, use_rope), GatedFFN(c)

    def __call__(self, x, mask):
        x = x + self.attention(self.norm1(x), mask)
        return x + self.ffn(self.norm2(x))


class ConvBlock(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c["width"]), RMSNorm(c["width"])
        k = c["kernel"]
        self.conv = nn.Conv1d(c["width"], c["width"], k, padding=dilation * (k - 1) // 2, dilation=dilation, bias=False)

    def __call__(self, x, keep):
        return x + nn.silu(self.norm2(self.conv(self.norm1(x) * keep)))


class DurationPredictor(nn.Module):
    def __init__(self, c):
        super().__init__()
        w = c["dur_width"]
        self.in_proj = nn.Linear(c["width"], w)
        self.dw = [nn.Conv1d(w, w, 3, padding=1, groups=w) for _ in range(2)]
        self.pw = [nn.Conv1d(w, w, 1) for _ in range(2)]
        self.norms = [RMSNorm(w) for _ in range(2)]
        self.out = nn.Linear(w, 1)

    def __call__(self, x, keep):
        h = nn.silu(self.in_proj(x)) * keep
        for dw, pw, n in zip(self.dw, self.pw, self.norms):
            h = (h + nn.silu(pw(dw(n(h) * keep)))) * keep
        return self.out(h).squeeze(-1)


class Acoustic(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        w = c["width"]
        self.tokens, self.voices, self.stress = nn.Embedding(c["vocab"], w), nn.Embedding(c["voices"], w), nn.Embedding(2, w)
        self.text_blocks = [AttnBlock(c) if i % 2 == 0 else ConvBlock(c, 2 ** (i // 2)) for i in range(c["text_layers"])]
        self.text_norm = RMSNorm(w)
        self.durations = DurationPredictor(c)
        self.phase = nn.Linear(4, w, bias=False)
        self.decoder = [AttnBlock(c, use_rope=(i == 0)) for i in range(c["decoder_layers"])]
        self.feature_norm = RMSNorm(w)
        self.head = nn.Linear(w, c["feature_dim"], bias=False)
        self.f_mean, self.f_std = mx.zeros((c["feature_dim"],)), mx.ones((c["feature_dim"],))

    def encode(self, tokens, stress, voice):
        """tokens, stress: (1,L) int; voice: (1,) int -> (1,L,W), log1p(duration) (1,L)."""
        keep = mx.ones(tokens.shape + (1,))
        x = self.tokens(tokens) + self.stress(stress) + self.voices(voice)[:, None]
        for blk in self.text_blocks:
            x = blk(x, None) if isinstance(blk, AttnBlock) else blk(x, keep)
        x = self.text_norm(x)
        return x, self.durations(x, keep)

    def decode(self, expanded, phase):
        n = expanded.shape[1]
        i = mx.arange(n)
        causal = (i[None] <= i[:, None]) & (i[:, None] - i[None] < self.c["window"])
        x = expanded + self.phase(phase)
        for blk in self.decoder:
            x = blk(x, causal)
        return self.head(self.feature_norm(x))


def phase_features(durations, max_token_frames=200):
    """durations: ndarray (L,) int -> (idx (F,), phase (F,4)) in numpy (cheap, no graph)."""
    d = np.asarray(durations, np.int64)
    idx = np.repeat(np.arange(len(d)), d)
    ends = np.cumsum(d); starts = ends - d
    counts = np.maximum(d[idx], 1).astype(np.float32)
    frac = (np.arange(len(idx), dtype=np.float32) - starts[idx] + 0.5) / counts
    phase = np.stack([2 * frac - 1, np.sin(np.pi * frac), np.cos(np.pi * frac),
                      np.log1p(counts) / np.log1p(max_token_frames)], -1).astype(np.float32)
    return idx, phase


# Duration limits, in 10 ms frames, from the training data (95th percentile of vowel length; median of the first phone of an utterance).
# The narrow model stretches the start of short utterances ("Olá!" gets a 300 ms "o"); see docs/EXPERIMENTS.md, section 9.
PHONE_LIMIT, FIRST_PHONE_LIMIT = 20, 17


def acoustic_features(model, ids, stress, voice, speed=1.0, phone_mask=None, limit=True):
    """ids/stress: lists; voice: 0/1 -> (T,20) Opus features, durations (L,)."""
    c = model.c
    tok = mx.array([ids]); st = mx.array([stress]); v = mx.array([voice])
    enc, log_dur = model.encode(tok, st, v)
    ld = np.asarray(log_dur)[0]
    dur = np.rint(np.expm1(np.clip(ld, 0, math.log1p(c["max_token_frames"]))) / speed).astype(np.int64)
    if phone_mask is not None:
        if limit:
            cap = np.where(phone_mask, max(1, round(PHONE_LIMIT / speed)), c["max_token_frames"])
            first = int(np.argmax(phone_mask)) if phone_mask.any() else None
            if first is not None: cap[first] = max(1, round(FIRST_PHONE_LIMIT / speed))
            dur = np.minimum(dur, cap)
        dur = np.where(phone_mask, np.maximum(dur, 1), dur)
    dur = np.minimum(dur, c["max_token_frames"])
    dur[np.asarray(ids) == 0] = 0
    if dur.sum() > 3000:
        raise ValueError("texto longo demais: divida em frases")
    idx, phase = phase_features(dur, c["max_token_frames"])
    expanded = enc[0][mx.array(idx)][None]
    z = model.decode(expanded, mx.array(phase)[None])[0]
    return np.asarray(z * model.f_std + model.f_mean, np.float32), dur
