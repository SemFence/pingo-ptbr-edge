"""xxs vocoder (reduced FARGAN) in MLX, with the firmware's int8 weights dequantized. Same computation as c/fargan_xs.c, in float."""
import numpy as np
import mlx.core as mx
from scipy.signal import lfilter
from . import weights as W


def _gru(wi, wh, nh, x, h):
    gi, gh = wi @ x, wh @ h
    r = mx.sigmoid(gi[:nh] + gh[:nh]); z = mx.sigmoid(gi[nh:2 * nh] + gh[nh:2 * nh])
    c = mx.tanh(gi[2 * nh:] + r * gh[2 * nh:])
    return (1 - z) * c + z * h


class Vocoder:
    def __init__(self, path=W.DEFAULT):
        z, meta = W.load(path)
        self.w = {k: mx.array(v) for k, v in W.vocoder(z, meta).items()}
        self.sz = W.vocoder_sizes(meta)
        self._sub = mx.compile(self._subframe)

    def _cond(self, fa, per):
        w = self.w
        e = w["pembed"][mx.array(per - 32)]
        d1 = mx.tanh(mx.concatenate([fa, e], -1) @ w["fdense1"].T)
        T = fa.shape[0] - 4
        cat = mx.concatenate([d1[2:2 + T], d1[3:3 + T], d1[4:4 + T]], -1)
        c2 = mx.tanh(cat @ w["fconv1"].T)
        return mx.tanh(c2 @ w["fdense2"].T)

    def _subframe(self, cs, period, exc, h1, h2, h3, fst):
        w, sz = self.w, self.sz; H1, H2, H3 = sz["H1"], sz["H2"], sz["H3"]
        gain = mx.exp(w["cond_gain"][0] @ cs + w["cond_gain_b"][0])
        idx = 256 - period + mx.arange(44) - 2
        idx = mx.maximum(mx.where(idx >= 256, idx - period, idx), 0)
        pred = mx.clip(exc[idx] / (1e-5 + gain), -1, 1); prev = mx.clip(exc[-40:] / (1e-5 + gain), -1, 1); fp = pred[2:42]
        tmp = mx.concatenate([cs, pred, prev])
        fw = mx.tanh(w["fwc0"] @ mx.concatenate([fst, tmp])); fw = mx.clip(fw * mx.sigmoid(w["fwc0_glu"] @ fw), -1, 1)
        pg = mx.sigmoid(w["gain_out"] @ fw + w["gain_out_b"])
        h1 = _gru(w["gru1_ih"], w["gru1_hh"], H1, mx.concatenate([fw, pg[0] * fp, prev]), h1)
        o1 = mx.clip(h1, -1, 1); o1 = mx.clip(o1 * mx.sigmoid(w["gru1_glu"] @ o1), -1, 1)
        h2 = _gru(w["gru2_ih"], w["gru2_hh"], H2, mx.concatenate([o1, pg[1] * fp, prev]), h2)
        o2 = mx.clip(h2, -1, 1); o2 = mx.clip(o2 * mx.sigmoid(w["gru2_glu"] @ o2), -1, 1)
        h3 = _gru(w["gru3_ih"], w["gru3_hh"], H3, mx.concatenate([o2, pg[2] * fp, prev]), h3)
        o3 = mx.clip(h3, -1, 1); o3 = mx.clip(o3 * mx.sigmoid(w["gru3_glu"] @ o3), -1, 1)
        so = mx.clip(mx.tanh(w["skip_dense"] @ mx.concatenate([o1, o2, o3, fw, pg[3] * fp, prev])), -1, 1)
        so = so * mx.sigmoid(w["skip_glu"] @ so)
        sig = mx.tanh(w["sig_out"] @ so) * gain
        return sig, mx.concatenate([exc[40:], sig]), h1, h2, h3, tmp

    def __call__(self, feats):
        """feats (T,20) float32 -> 16 kHz float32 audio (with the 0.85 de-emphasis)."""
        sz = self.sz; C = sz["C"]
        feats = np.asarray(feats, np.float32)
        fa = np.concatenate([np.repeat(feats[:1], 3, 0), feats, feats[-1:]])
        per = np.rint(256 / 2 ** (np.clip(fa[:, 18], -10, 10) + 1.5)).clip(32, 255).astype(np.int32)
        T = len(feats); cond = self._cond(mx.array(fa), per); mx.eval(cond)
        exc = mx.zeros(256); h1, h2, h3 = mx.zeros(sz["H1"]), mx.zeros(sz["H2"]), mx.zeros(sz["H3"]); fst = mx.zeros(2 * 40 + C + 4)
        out = []
        for n in range(T):
            period = mx.array(int(per[n + 3]))
            for k in range(4):
                sig, exc, h1, h2, h3, fst = self._sub(cond[n, k * C:(k + 1) * C], period, exc, h1, h2, h3, fst); out.append(sig)
            if n % 8 == 7: mx.eval(exc, h1, h2, h3, fst, out[-32:])
        x = np.concatenate([np.asarray(o) for o in out])
        return lfilter([1.0], [1.0, -0.85], x).astype(np.float32)
