"""Confere os backends contra o C: python check.py   (precisa de torch e mlx)
Mede o erro rms relativo do acústico contra a referência do C/MLX (c/ac_test_ref.f32) e do vocoder MLX contra o PyTorch."""
import pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from pingo_edge.acoustic_torch import AcousticTorch
from pingo_edge.acoustic_mlx_edge import AcousticMLX
from pingo_edge.vocoder_torch import Vocoder as VT
from pingo_edge.vocoder_mlx import Vocoder as VM

a = np.fromfile(HERE.parent / "c" / "ac_test_in.i32", np.int32); L, voice = int(a[0]), int(a[1])
ids, st, mask = a[2:2 + L], a[2 + L:2 + 2 * L], a[2 + 2 * L:2 + 3 * L].astype(bool)
ref = np.fromfile(HERE.parent / "c" / "ac_test_ref.f32", np.float32); T = int(ref[0]); rf = ref[1 + L:].reshape(T, 20)
rel = lambda x, y: float(np.sqrt(((x - y) ** 2).mean() / (y ** 2).mean()))
ft, _ = AcousticTorch().features(ids.tolist(), st.tolist(), voice, 1.0, mask)
fm, _ = AcousticMLX().features(ids.tolist(), st.tolist(), voice, 1.0, mask)
print(f"quadros: torch {len(ft)} | mlx {len(fm)} | referência {T}")
print(f"acústico, erro rms relativo contra a referência: torch {rel(ft, rf[:len(ft)]):.2e} | mlx {rel(fm, rf[:len(fm)]):.2e}")
wt, wm = VT()(ft), VM()(ft)
print(f"vocoder, MLX contra PyTorch (mesmas features): {rel(wm, wt):.2e} | rms {np.sqrt((wt ** 2).mean()) * 32768:.0f} / {np.sqrt((wm ** 2).mean()) * 32768:.0f}")
