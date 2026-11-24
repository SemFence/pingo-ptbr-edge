"""Pingo acoustic model in PyTorch with the firmware's 4-bit weights, dequantized (model from semfence/pingo-ptbr: training/model.py)."""
import numpy as np
import torch
from . import weights as W
from ._torch_acoustic_model import Acoustic, Config
from .common import durations, phase_features


class AcousticTorch:
    def __init__(self, path=W.DEFAULT):
        z, meta = W.load(path); cfg, ac = W.acoustic(z, meta)
        self.cfg = cfg; self.model = Acoustic(Config(**cfg)).eval()
        sd = {}
        for k, v in self.model.state_dict().items():
            a = ac[W.cname(k)]
            if v.ndim == 3:                                    # conv: stored as (out, k, in) (MLX layout); in PyTorch it is (out, in, k)
                a = a.reshape(v.shape[0], v.shape[2], v.shape[1]).transpose(0, 2, 1)
            sd[k] = torch.tensor(np.ascontiguousarray(a.reshape(v.shape)))
        self.model.load_state_dict(sd)

    @torch.no_grad()
    def features(self, ids, stress, voice, speed=1.0, phone_mask=None):
        """-> (T,20) Opus features, durations (L,)."""
        m, c = self.model, self.cfg; L = len(ids)
        tok, st = torch.tensor([ids]), torch.tensor([stress])
        enc, log_dur = m.encode(tok, st, torch.tensor([voice]), torch.tensor([L]))
        dur = durations(log_dur[0].numpy(), ids, np.asarray(phone_mask), c["max_token_frames"], speed)
        idx, phase = phase_features(dur, c["max_token_frames"])
        expanded = enc[0][torch.tensor(idx)][None]
        fmask = torch.ones(1, len(idx), dtype=torch.bool)
        z = m.decode(expanded, torch.tensor(phase)[None], fmask)[0]
        return (z * m.f_std + m.f_mean).numpy().astype(np.float32), dur
