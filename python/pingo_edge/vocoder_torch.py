"""Vocoder xxs (FARGAN reduzido) em PyTorch, com os pesos int8 do firmware desquantizados. Mesma conta de c/fargan_xs.c, em float.
Não usa o código de treino (weight_norm, soft_quant): é só inferência."""
import numpy as np
import torch
from scipy.signal import lfilter
from . import weights as W


def _gru(wi, wh, nh, x, h):
    gi, gh = wi @ x, wh @ h
    r = torch.sigmoid(gi[:nh] + gh[:nh]); z = torch.sigmoid(gi[nh:2 * nh] + gh[nh:2 * nh])
    c = torch.tanh(gi[2 * nh:] + r * gh[2 * nh:])
    return (1 - z) * c + z * h


class Vocoder:
    def __init__(self, path=W.DEFAULT):
        z, meta = W.load(path)
        self.w = {k: torch.tensor(v) for k, v in W.vocoder(z, meta).items()}
        self.sz = W.vocoder_sizes(meta)

    def _cond(self, fa, per):
        w = self.w
        e = w["pembed"][torch.tensor(per - 32)]
        d1 = torch.tanh(torch.cat([fa, e], -1) @ w["fdense1"].T)                          # (T+4, d1)
        T = fa.shape[0] - 4
        cat = torch.cat([d1[2:2 + T], d1[3:3 + T], d1[4:4 + T]], -1)                      # camadas no tempo: n+2, n+3, n+4
        c2 = torch.tanh(cat @ w["fconv1"].T)
        return torch.tanh(c2 @ w["fdense2"].T)                                            # (T, 4*C)

    @torch.no_grad()
    def __call__(self, feats):
        """feats (T,20) float32 -> áudio 16 kHz float32 (com de-ênfase 0,85)."""
        w, sz = self.w, self.sz; C, F, H1, H2, H3 = sz["C"], sz["F"], sz["H1"], sz["H2"], sz["H3"]
        feats = np.asarray(feats, np.float32)
        fa = np.concatenate([np.repeat(feats[:1], 3, 0), feats, feats[-1:]])             # quadros n..n+4 de cada saída
        per = np.rint(256 / 2 ** (np.clip(fa[:, 18], -10, 10) + 1.5)).clip(32, 255).astype(np.int64)
        T = len(feats); cond = self._cond(torch.tensor(fa), per)
        exc = torch.zeros(256); h1, h2, h3 = torch.zeros(H1), torch.zeros(H2), torch.zeros(H3); fst = torch.zeros(2 * 40 + C + 4)
        out = []
        for n in range(T):
            period = int(per[n + 3])
            for k in range(4):
                cs = cond[n, k * C:(k + 1) * C]
                gain = torch.exp(w["cond_gain"][0] @ cs + w["cond_gain_b"][0])
                idx = 256 - period + torch.arange(44) - 2
                idx = torch.where(idx >= 256, idx - period, idx).clamp(min=0)
                pred = (exc[idx] / (1e-5 + gain)).clamp(-1, 1); prev = (exc[-40:] / (1e-5 + gain)).clamp(-1, 1); fp = pred[2:42]
                tmp = torch.cat([cs, pred, prev])
                fw = torch.tanh(w["fwc0"] @ torch.cat([fst, tmp])); fw = (fw * torch.sigmoid(w["fwc0_glu"] @ fw)).clamp(-1, 1); fst = tmp
                pg = torch.sigmoid(w["gain_out"] @ fw + w["gain_out_b"])
                h1 = _gru(w["gru1_ih"], w["gru1_hh"], H1, torch.cat([fw, pg[0] * fp, prev]), h1)
                o1 = h1.clamp(-1, 1); o1 = (o1 * torch.sigmoid(w["gru1_glu"] @ o1)).clamp(-1, 1)
                h2 = _gru(w["gru2_ih"], w["gru2_hh"], H2, torch.cat([o1, pg[1] * fp, prev]), h2)
                o2 = h2.clamp(-1, 1); o2 = (o2 * torch.sigmoid(w["gru2_glu"] @ o2)).clamp(-1, 1)
                h3 = _gru(w["gru3_ih"], w["gru3_hh"], H3, torch.cat([o2, pg[2] * fp, prev]), h3)
                o3 = h3.clamp(-1, 1); o3 = (o3 * torch.sigmoid(w["gru3_glu"] @ o3)).clamp(-1, 1)
                so = torch.tanh(w["skip_dense"] @ torch.cat([o1, o2, o3, fw, pg[3] * fp, prev])).clamp(-1, 1)
                so = so * torch.sigmoid(w["skip_glu"] @ so)
                sig = torch.tanh(w["sig_out"] @ so) * gain
                exc = torch.cat([exc[40:], sig]); out.append(sig)
        x = torch.cat(out).numpy()
        return lfilter([1.0], [1.0, -0.85], x).astype(np.float32)
