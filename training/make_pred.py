"""Features PREVISTAS pelo acústico do pingo (durações do alinhamento, para casar com o áudio real) + o PCM do mesmo clipe.
Serve para ajustar o vocoder leve ao tipo de feature que ele vai receber de verdade (mais suave que a extraída do áudio).
  .venv-mlx/bin/python make_pred.py SPLIT N OUT_PREFIX      -> OUT_PREFIX.f32 (20 colunas) e OUT_PREFIX.s16
Usa os pesos EMPACOTADOS do pingo (release/hf/weights/pingo-ptbr.bin), os mesmos do produto."""
import sys, io, json, glob, os, numpy as np, soundfile as sf, pyarrow.parquet as pq
from pathlib import Path
from scipy.signal import resample_poly
import mlx.core as mx
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "release" / "hf" / "pingo"))
import speak as sp
pm = sp.pingo_mlx
split, N, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
prep = Path.home() / "Projects/fun/pivo-ptbr/data/prep"
DATA = Path.home() / "Projects/fun/ptbr-voz-dataset/hf_release/speech/data"
ids = [json.loads(l)["id"] for l in (prep / f"{split}.jsonl").read_text().splitlines()][:N]
want = set(ids)
model = sp.Pingo().model
c = model.c
fo, po = open(out + ".f32", "wb"), open(out + ".s16", "wb")
done = 0
for path in sorted(glob.glob(str(DATA / f"{split}-*.parquet"))):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        for r in batch.to_pylist():
            if r["id"] not in want:
                continue
            z = np.load(prep / split / f"{r['id']}.npz")
            frames = len(z["features"]); dur = z["durations"].astype(np.int64)
            tok, st, v = z["tokens"].astype(np.int64), z["stress"].astype(np.int64), int(z["voice"])
            enc, _ = model.encode(mx.array([tok.tolist()]), mx.array([st.tolist()]), mx.array([v]))
            idx, phase = pm.phase_features(dur, c["max_token_frames"])
            if len(idx) == 0:
                continue
            feats = np.asarray(model.decode(enc[0][mx.array(idx)][None], mx.array(phase)[None])[0] * model.f_std + model.f_mean, np.float32)
            if len(feats) < frames: feats = np.concatenate([feats, np.repeat(feats[-1:], frames - len(feats), 0)])
            feats = feats[:frames]
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            w = resample_poly(a, 2, 3).astype(np.float32)[:frames * 160]
            if len(w) < frames * 160: w = np.pad(w, (0, frames * 160 - len(w)))
            fo.write(feats.tobytes()); po.write((np.clip(w, -1, 1) * 32767).astype("<i2").tobytes()); done += 1
            if done % 500 == 0: print(done, flush=True)
fo.close(); po.close(); print("clipes:", done)
