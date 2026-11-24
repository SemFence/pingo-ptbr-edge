"""Weighted training PCM for the small vocoder: more clips with foreign terms and expressive sentences, fewer plain readings.
Weights are expected repetitions of each clip (the fractional part is a coin flip)."""
import io, sys, glob, re, numpy as np, pyarrow.parquet as pq, soundfile as sf
from scipy.signal import resample_poly
root = sys.argv[1]; rng = np.random.default_rng(0)
FOREIGN = re.compile(r"\b(?:[\w-]*[kwy][\w-]*|\w+ing|\w*th\w*|\w*(?:ck|sh)\w*|wi-?fi|app|email|online|deploy|software|download|upload|streaming|marketing|startup|feedback|delivery|smartphone|notebook|mouse|link|post|chat|bug|site|web|hacker|cloud)\b", re.I)
CAT = dict(exclamacao=1.5, pergunta=1.5, conversa=1.5, narrativa=1.0, leitura=0.6, numero=1.0, data=1.0, hora=1.0, nome=1.0, dinheiro=1.0)
clips = []
for p in sorted(glob.glob(f"{root}/data/train-*.parquet")):
    clips += pq.read_table(p, columns=["audio", "text", "category"]).to_pylist()
order = rng.permutation(len(clips)); secs = 0.0; foreign = 0; n = 0
with open("train_w.s16", "wb") as f:
    for i in order:
        c = clips[i]; w = CAT.get(c["category"], 1.0); fo = bool(FOREIGN.search(c["text"]))
        if fo: w *= 3; 
        reps = int(w) + (rng.random() < w - int(w))
        if reps == 0: continue
        x, sr = sf.read(io.BytesIO(c["audio"]["bytes"]), dtype="float32")
        if x.ndim > 1: x = x.mean(1)
        y = (np.clip(resample_poly(x, 2, 3), -1, 1) * 32767).astype("<i2").tobytes()
        for _ in range(reps): f.write(y); secs += len(y) / 32000; n += 1; foreign += fo
print(f"{n} clip copies, {secs/3600:.1f} h, foreign terms: {foreign} ({100*foreign/n:.1f}%)")
