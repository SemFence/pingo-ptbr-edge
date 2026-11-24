"""Reads the pingo edge C headers (c/xs_weights.h = int8 xxs vocoder, c/ac_weights.h = 4-bit acoustic model) and writes the SAME values to
weights/pingo-edge.npz, so PyTorch and MLX use exactly the weights that go into the firmware.   python tools/headers_to_npz.py"""
import re, json, sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
meta, arrays = {}, {}
for header in ("xs_weights.h", "ac_weights.h"):
    text = (ROOT / "c" / header).read_text()
    for name, value in re.findall(r"#define (\w+) (-?\d+)\b", text):
        meta[name] = int(value)
    for ctype, name, body in re.findall(r"static const (\w+) (\w+)\[\d*\] = \{([^}]*)\};", text):
        dtype = {"int8_t": np.int8, "uint8_t": np.uint8, "float": np.float32}[ctype]
        arrays[name] = np.array([float(x.rstrip("f")) for x in body.split(",") if x.strip()]).astype(dtype)
np.savez_compressed(ROOT / "weights" / "pingo-edge.npz", meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arrays)
print(len(arrays), "arrays,", sum(a.nbytes for a in arrays.values()) // 1024, "KiB ->", ROOT / "weights" / "pingo-edge.npz")
