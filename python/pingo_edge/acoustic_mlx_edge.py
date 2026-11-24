"""Acústico do pingo em MLX com os pesos de 4 bits do firmware desquantizados."""
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from . import weights as W
from . import acoustic_mlx as A


class AcousticMLX:
    def __init__(self, path=W.DEFAULT):
        z, meta = W.load(path); cfg, ac = W.acoustic(z, meta)
        self.cfg = cfg; self.model = A.Acoustic(cfg)
        flat = [(k, mx.array(np.ascontiguousarray(ac[W.cname(k)].reshape(v.shape)))) for k, v in tree_flatten(self.model.parameters())]
        self.model.update(tree_unflatten(flat)); mx.eval(self.model.parameters())

    def features(self, ids, stress, voice, speed=1.0, phone_mask=None):
        return A.acoustic_features(self.model, [int(i) for i in ids], [int(i) for i in stress], int(voice), speed, np.asarray(phone_mask))
