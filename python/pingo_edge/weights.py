"""Carrega weights/pingo-edge.npz (gerado de c/*.h por tools/headers_to_npz.py) e desquantiza: acústico 4 bits, vocoder int8."""
import json, re
from pathlib import Path
import numpy as np

DEFAULT = Path(__file__).resolve().parent.parent.parent / "weights" / "pingo-edge.npz"


def load(path=DEFAULT):
    z = np.load(path)
    meta = json.loads(bytes(z["meta"]).decode())
    return z, meta


def vocoder(z, meta):
    """-> dict nome -> float32 (rows, cols) para as matrizes; vetores (viés) como estão. Nomes como no C: fdense1, gru1_ih, cond_gain, ..."""
    out = {}
    for key in z.files:
        if not key.startswith("xs_"): continue
        name = key[3:]
        if name.endswith("_w"):
            n = name[:-2]; rows, cols = meta[f"XS_{n}_ROWS"], meta[f"XS_{n}_COLS"]
            out[n] = (z[key].astype(np.float32).reshape(rows, cols) * z[f"xs_{n}_s"][:, None]).astype(np.float32)
        elif not name.endswith("_s"):
            out[name] = z[key].astype(np.float32)           # cond_gain_b, gain_out_b
    return out


def vocoder_sizes(meta):
    return {k[3:]: v for k, v in meta.items() if re.fullmatch(r"XS_(C|D1|D2|F|H1|H2|H3|S)", k)}


def acoustic(z, meta):
    """-> (config dict, dict nome_c -> float32). Os pesos 2-D vêm como (rows, cols) desquantizados; os vetores como estão."""
    cfg = {k[7:].lower(): v for k, v in meta.items() if k.startswith("AC_CFG_")}
    out = {}
    for key in z.files:
        if not key.startswith("ac_"): continue
        name = key[3:]
        if name.endswith("_w"):
            n = name[:-2]; rows, cols = meta[f"AC_{n}_ROWS"], meta[f"AC_{n}_COLS"]
            raw = z[key]
            if meta["AC_WBITS"] == 4:
                lo = (raw & 0xF).astype(np.int8); hi = (raw >> 4).astype(np.int8)
                q = np.empty(raw.size * 2, np.int8); q[0::2], q[1::2] = lo, hi
                q = np.where(q > 7, q - 16, q)[: rows * cols]                 # complemento de dois, -7..7
            else:
                q = raw.astype(np.int8)
            out[n] = (q.astype(np.float32).reshape(rows, cols) * z[f"ac_{n}_s"][:, None]).astype(np.float32)
        elif not name.endswith("_s"):
            out[name] = z[key].astype(np.float32)
    return cfg, out


def cname(k):
    return re.sub(r"[^0-9a-zA-Z]+", "_", k).strip("_")
