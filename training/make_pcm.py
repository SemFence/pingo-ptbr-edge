"""Parquet (FLAC 24 kHz) -> raw 16 kHz int16 PCM, one file per split, as `dump_data -train` expects."""
import io, sys, glob, numpy as np, pyarrow.parquet as pq, soundfile as sf
from scipy.signal import resample_poly
root = sys.argv[1]
for split, out in (("train", "train.s16"), ("validation", "val.s16")):
    n = 0; secs = 0.0
    with open(out, "wb") as f:
        for p in sorted(glob.glob(f"{root}/data/{split}-*.parquet")):
            t = pq.read_table(p, columns=["audio"]).column("audio")
            for a in t.to_pylist():
                x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
                if x.ndim > 1: x = x.mean(1)
                assert sr == 24000, sr
                y = resample_poly(x, 2, 3)
                f.write((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()); n += 1; secs += len(y) / 16000
    print(split, n, "clips", round(secs / 3600, 2), "h", flush=True)
