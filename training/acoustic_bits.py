"""Tests the pingo acoustic model with 8/6/5/4-bit weights (simulated, per row): feature error and audio with the xxs vocoder.
  ../../pivo-ptbr/.venv-mlx/bin/python acoustic_bits.py OUT_DIR"""
import sys, os, types, numpy as np, torch, soundfile as sf
sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "release", "hf", "pingo"))
import mlx.core as mx
from mlx.utils import tree_unflatten
import qpack, speak as sp, pingo_mlx, fargan_small as fargan
from scipy.signal import lfilter
from tiny_voca.text import PHONE_IDS, phonemize
out = sys.argv[1]; os.makedirs(out, exist_ok=True); mx.set_default_device(mx.cpu)
ac, voc, meta = qpack.load_packed(os.path.join(HERE, "..", "release", "hf", "weights", "pingo-ptbr.bin")); cfg = meta["config"]
def quant(w, bits):
    if bits is None or w.ndim < 2: return w
    qm = 2 ** (bits - 1) - 1; w2 = w.reshape(w.shape[0], -1); s = np.maximum(np.abs(w2).max(1, keepdims=True) / qm, 1e-12)
    return (np.clip(np.rint(w2 / s), -qm, qm) * s).reshape(w.shape).astype(np.float32)
def model(bits):
    m = pingo_mlx.Acoustic(cfg); m.update(tree_unflatten([(k, mx.array(quant(v, bits))) for k, v in ac.items()])); mx.eval(m.parameters()); return m
ck = torch.load(os.path.join(HERE, "runs", "xxs_int8sim.pth"), map_location="cpu")
voc = fargan.FARGAN(softquant=False, sizes=ck["sizes"]); voc.load_state_dict(ck["state_dict"]); voc.eval(); fargan.n = lambda x: torch.clamp(x, -1., 1.)
def vocode(feats):
    f = np.concatenate([np.repeat(feats[:1], 3, 0), feats, feats[-1:]]).astype(np.float32)
    per = np.rint(256. / 2 ** (np.clip(f[:, 18], -10, 10) + 1.5)).clip(32, 255).astype("int64")
    with torch.no_grad(): sig, _ = voc(torch.tensor(f)[None], torch.tensor(per)[None], len(feats))
    return lfilter([1.0], [1.0, -0.85], sig[0].numpy()).astype(np.float32)
FR = {"07_ola_nome": "Olá, eu sou a voz da SemFence. Em que posso ajudar?", "05_boa_noite": "Boa noite! Tenha um ótimo descanso.", "08_ate_logo": "Até logo! Foi um prazer conversar com você."}
models = {b: model(b) for b in (None, 8, 6, 5, 4)}
for k, t in FR.items():
    ph = sp.joined(phonemize(sp.spoken(t))); ids, st = list(ph.ids), list(ph.stress); mask = np.array([i in PHONE_IDS for i in ids]); res = {}
    for b, m in models.items():
        feats, dur = pingo_mlx.acoustic_features(m, ids, st, 0, 1.0, mask); res[b] = np.asarray(feats, np.float32)
    ref = res[None]
    for b in (8, 6, 5, 4):
        f = res[b]; n = min(len(f), len(ref)); err = np.linalg.norm(f[:n] - ref[:n]) / np.linalg.norm(ref[:n]) if len(f) == len(ref) else float("nan")
        print(f"{k} {b} bits: quadros {len(f)} (float {len(ref)}) | erro relativo das features {err*100:.2f}%", flush=True)
    for b in (None, 8, 6, 5, 4):
        w = vocode(res[b]); w = w * (0.5 / max(float(np.abs(w).max()), 1e-6)); sf.write(f"{out}/{k}_ac{'float' if b is None else b}.wav", w, 16000)
