"""Exporta o acústico do pingo (do .bin empacotado) para C: pesos int8 por linha + escalas, vetores em float.
Gera também um vetor de teste: referência do MLX com os MESMOS pesos arredondados.
  ../../pivo-ptbr/.venv-mlx/bin/python export_acoustic_c.py c/ "Olá, eu sou a voz da SemFence."
"""
import sys, os, re, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "release", "hf", "pingo"))
import mlx.core as mx
from mlx.utils import tree_unflatten
import qpack, speak as sp, pingo_mlx
from tiny_voca.text import PHONE_IDS, phonemize

out, text = sys.argv[1], sys.argv[2]; BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 8; QM = 2 ** (BITS - 1) - 1; os.makedirs(out, exist_ok=True)
ac, voc, meta = qpack.load_packed(os.path.join(HERE, "..", "release", "hf", "weights", "pingo-ptbr.bin"))
cfg = meta["config"]
def cname(k): return re.sub(r"[^0-9a-zA-Z]+", "_", k).strip("_")
def quant(w):
    w2 = w.reshape(w.shape[0], -1); s = np.maximum(np.abs(w2).max(1, keepdims=True) / QM, 1e-12)
    q = np.clip(np.rint(w2 / s), -QM, QM); return q.astype(np.int8), s[:, 0].astype(np.float32), (q * s).reshape(w.shape).astype(np.float32)
ref = {}; nq = 0; nrows = 0; nf = 0; nq4 = 0
with open(os.path.join(out, "ac_weights.h"), "w") as f:
    f.write("/* gerado por export_acoustic_c.py: acústico do pingo, int8 por linha + escala float; vetores em float. Não editar. */\n#pragma once\n#include <stdint.h>\n#ifndef AC_ATTR\n#define AC_ATTR   /* pesos do acústico ficam na flash (const) */\n#endif\n")
    f.write(f"#define AC_WBITS {BITS}   /* 8: um int8 por peso; 4: dois pesos por byte (nibble baixo = índice par), complemento de dois, valores -7..7 */\n")
    for k, v in cfg.items():
        if isinstance(v, int): f.write(f"#define AC_CFG_{k.upper()} {v}\n")
    for k, a in ac.items():
        n = cname(k)
        if a.ndim >= 2:
            q, s, wd = quant(a); ref[k] = wd; nq += q.size; nrows += len(s)
            f.write(f"#define AC_{n}_ROWS {q.shape[0]}\n#define AC_{n}_COLS {q.shape[1]}\n")
            if BITS == 4:
                flat = q.flatten().astype(np.int16); flat = np.concatenate([flat, [0]]) if len(flat) % 2 else flat
                packed = ((flat[0::2] & 0xF) | ((flat[1::2] & 0xF) << 4)).astype(np.uint8)
                f.write(f"static const uint8_t ac_{n}_w[] = {{" + ",".join(map(str, packed.tolist())) + "};\n"); nq4 += len(packed)
            else:
                f.write(f"static const int8_t ac_{n}_w[] = {{" + ",".join(map(str, q.flatten().tolist())) + "};\n")
            f.write(f"static const float ac_{n}_s[] = {{" + ",".join(f"{x:.9g}f" for x in s.tolist()) + "};\n")
        else:
            ref[k] = a; nf += a.size
            f.write(f"static const float ac_{n}[] = {{" + ",".join(f"{x:.9g}f" for x in a.tolist()) + "};\n")
print(f"acústico ({BITS} bits): {nq} pesos int8 = {nq/1024:.1f} KiB + {nrows} escalas float ({nrows*4/1024:.1f} KiB) + {nf} floats ({nf*4/1024:.1f} KiB)" + (f" | empacotado: {nq4/1024:.1f} KiB" if BITS == 4 else ""))

# ---- referência MLX com os pesos arredondados
mx.set_default_device(mx.cpu)
model = pingo_mlx.Acoustic(cfg); model.update(tree_unflatten([(k, mx.array(v)) for k, v in ref.items()])); mx.eval(model.parameters())
ph = sp.joined(phonemize(sp.spoken(text))); ids, stress = list(ph.ids), list(ph.stress); L = len(ids); voice = 0
mask = np.array([i in PHONE_IDS for i in ids])
enc, ld = model.encode(mx.array([ids]), mx.array([stress]), mx.array([voice])); ld = np.asarray(ld)[0].astype(np.float32)
feats, dur = pingo_mlx.acoustic_features(model, ids, stress, voice, 1.0, mask)
np.array([L, voice] + ids + stress + mask.astype(int).tolist(), dtype=np.int32).tofile(os.path.join(out, "ac_test_in.i32"))
np.concatenate([[len(feats)], ld, feats.flatten()]).astype(np.float32).tofile(os.path.join(out, "ac_test_ref.f32"))
print("tokens:", L, "| quadros:", len(feats), "| f_std/f_mean no .h")
