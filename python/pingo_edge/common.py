"""Partes sem framework (numpy): durações em quadros e sinais de fase. Mesmas regras de c/acoustic.c e de pingo_mlx.acoustic_features."""
import math
import numpy as np

PHONE_LIMIT, FIRST_PHONE_LIMIT = 20, 17      # quadros de 10 ms: percentil 95 das vogais e mediana do 1º fonema da fala


def durations(log_dur, ids, phone_mask, max_token_frames, speed=1.0):
    d = np.rint(np.expm1(np.clip(log_dur, 0, math.log1p(max_token_frames))) / speed).astype(np.int64)
    cap = np.where(phone_mask, max(1, round(PHONE_LIMIT / speed)), max_token_frames)
    if phone_mask.any(): cap[int(np.argmax(phone_mask))] = max(1, round(FIRST_PHONE_LIMIT / speed))
    d = np.minimum(d, cap); d = np.where(phone_mask, np.maximum(d, 1), d); d = np.minimum(d, max_token_frames)
    d[np.asarray(ids) == 0] = 0
    if d.sum() > 3000: raise ValueError("texto longo demais: divida em frases")
    return d


def phase_features(dur, max_token_frames=200):
    d = np.asarray(dur, np.int64); idx = np.repeat(np.arange(len(d)), d)
    ends = np.cumsum(d); starts = ends - d; counts = np.maximum(d[idx], 1).astype(np.float32)
    frac = (np.arange(len(idx), dtype=np.float32) - starts[idx] + 0.5) / counts
    return idx, np.stack([2 * frac - 1, np.sin(np.pi * frac), np.cos(np.pi * frac), np.log1p(counts) / np.log1p(max_token_frames)], -1).astype(np.float32)
