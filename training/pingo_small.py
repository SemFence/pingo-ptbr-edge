"""Frases do pingo com o vocoder original e com os vocoders leves (s, xs), mesmas features previstas pelo acústico.
  python3 pingo_small.py OUT_DIR [--voices F_s4,M_sm4]
Alinhamento: o FARGAN torch descarta 2 quadros à esquerda e usa 3 quadros no conv (centro = j), então a entrada é
[3 quadros de preenchimento] + feats + [1], e o quadro n da saída corresponde a feats[n] (igual ao pivo/pingo MLX)."""
import sys, os, argparse, types, numpy as np, torch, soundfile as sf
sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "release", "hf", "pingo"))
import fargan_small as fargan
from scipy.signal import lfilter
import speak as sp

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--voices", default="F_s4,M_sm4")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

FRASES = {
    "01_ola": "Olá!",
    "03_bom_dia": "Bom dia! Como você está hoje?",
    "05_boa_noite": "Boa noite! Tenha um ótimo descanso.",
    "07_ola_nome": "Olá, eu sou a voz da SemFence. Em que posso ajudar?",
    "08_ate_logo": "Até logo! Foi um prazer conversar com você.",
}

pingo = sp.Pingo()


class Small:
    def __init__(self, ckpt):
        ck = torch.load(ckpt, map_location="cpu")
        self.m = fargan.FARGAN(softquant=ck.get("softquant", False), sizes=ck["sizes"])
        self.m.load_state_dict(ck["state_dict"]); self.m.eval()
        self.last_feats = None

    def vocode(self, feats):
        feats = np.asarray(feats, np.float32)
        f = np.concatenate([np.repeat(feats[:1], 3, 0), feats, feats[-1:]])
        per = np.rint(256. / 2 ** (np.clip(f[:, 18], -10, 10) + 1.5)).clip(32, 255).astype("int64")
        with torch.no_grad():
            sig, _ = self.m(torch.tensor(f)[None], torch.tensor(per)[None], len(feats))
        return lfilter([1.0], [1.0, -0.85], sig[0].numpy()).astype(np.float32)

    def __call__(self, ph, voice, speed):
        mask = np.array([i in sp.PHONE_IDS for i in ph.ids])
        feats, _ = sp.pingo_mlx.acoustic_features(pingo.model, list(ph.ids), list(ph.stress), voice, speed, mask)
        return self.vocode(feats)


models = {"xs_int8": Small("runs/xs_int8sim.pth"), "xxs_int8": Small("runs/xxs_int8sim.pth")}
for voice in a.voices.split(","):
    for k, t in FRASES.items():
        for name, be in models.items():
            w = sp.speak(be, t, voice=voice)
            w = w * (0.5 / max(float(np.abs(w).max()), 1e-6))          # mesmo volume em todos, para comparar de ouvido
            sf.write(f"{a.out}/{k}_{voice}_{name}.wav", w, sp.RATE)
        print(k, voice, "ok", flush=True)
