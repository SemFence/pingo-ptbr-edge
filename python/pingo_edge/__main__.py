"""python -m pingo_edge --backend torch|mlx --text "Olá, eu sou a voz da SemFence." -o out.wav"""
import argparse, time
import soundfile as sf
from .frontend import Pingo, speak, VOICES, RATE

ap = argparse.ArgumentParser(prog="pingo_edge", description="pingo-ptbr edge (acústico 4 bits + vocoder xxs int8) em PyTorch ou MLX")
ap.add_argument("--backend", choices=["torch", "mlx"], default="torch")
ap.add_argument("--voice", choices=list(VOICES), default="F_s4")
ap.add_argument("--text", required=True); ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--no-respelling", action="store_true", help="não reescrever termos em inglês")
ap.add_argument("-o", "--output", default="out.wav")
a = ap.parse_args()
t0 = time.time(); p = Pingo(a.backend); t1 = time.time()
y = speak(p, a.text, a.voice, a.speed, not a.no_respelling); t2 = time.time()
sf.write(a.output, y, RATE); print(f"{a.output}: {len(y) / RATE:.2f} s de áudio | carga {t1 - t0:.1f} s | síntese {t2 - t1:.1f} s ({len(y) / RATE / (t2 - t1):.1f}x tempo real) | {a.backend}")
