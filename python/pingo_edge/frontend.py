"""Text -> speech. Phonemization, sentence splitting and respelling are the same as in semfence/pingo-ptbr (speak.py); the backend is PyTorch or MLX.
Requires: pip install git+https://github.com/SemFence/dizione-ptbr git+https://github.com/SemFence/tiny-voca-ptbr"""
import numpy as np
from .pronunciation import joined, spoken
from tiny_voca.infer import finished, sentences, shorter
from tiny_voca.text import PHONE_IDS, NothingToPronounce, phonemize

VOICES = {"F_s4": 0, "M_sm4": 1}
RATE, PEAK = 16000, 0.99


class Pingo:
    def __init__(self, backend="torch", path=None):
        kw = {"path": path} if path else {}
        if backend == "torch":
            from .acoustic_torch import AcousticTorch as A
            from .vocoder_torch import Vocoder as V
        elif backend == "mlx":
            import mlx.core as mx
            mx.set_default_device(mx.cpu)                         # tiny model, sequential steps: the CPU beats the GPU
            from .acoustic_mlx_edge import AcousticMLX as A
            from .vocoder_mlx import Vocoder as V
        else:
            raise ValueError("backend: torch or mlx")
        self.acoustic, self.vocoder, self.backend = A(**kw), V(**kw), backend

    def __call__(self, ph, voice, speed):
        mask = np.array([i in PHONE_IDS for i in ph.ids])
        feats, _ = self.acoustic.features(list(ph.ids), list(ph.stress), voice, speed, mask)
        return self.vocoder(feats)


def speak(backend, text, voice="F_s4", speed=1.0, respelling=True):
    waves = []
    if respelling: text = spoken(text)

    def read(piece):
        try: ph = joined(phonemize(piece)) if respelling else phonemize(piece)
        except NothingToPronounce: return
        if len(ph.ids) <= 400:
            try: waves.append(np.asarray(backend(ph, VOICES[voice], speed))); return
            except ValueError: pass
        parts = shorter(piece)
        if not parts: raise ValueError(f"word too long: {piece[:40]!r}")
        for part in parts: read(part)

    for sentence in sentences(finished(text)): read(sentence)
    if not waves: raise NothingToPronounce("nothing to read")
    wave = np.concatenate(waves); peak = float(np.abs(wave).max())
    return wave * (PEAK / peak) if peak > PEAK else wave
