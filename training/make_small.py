"""Builds fargan_small.py: the Opus FARGAN with configurable layer sizes (defaults reproduce the original, 817,525 weights)."""
import re, pathlib
src = pathlib.Path("../refs/opus/dnn/torch/fargan/fargan.py").read_text()
def sub(old, new, count=1):
    global src
    assert src.count(old) == count, (old, src.count(old))
    src = src.replace(old, new)

# --- conditioning net
sub("def __init__(self, feature_dim=20, cond_size=256, pembed_dims=12, softquant=False):",
    "def __init__(self, feature_dim=20, cond_size=256, pembed_dims=12, softquant=False, sub_cond=80, d1=64, d2=128):")
sub("self.fdense1 = nn.Linear(self.feature_dim + pembed_dims, 64, bias=False)", "self.fdense1 = nn.Linear(self.feature_dim + pembed_dims, d1, bias=False)")
sub("self.fconv1 = nn.Conv1d(64, 128, kernel_size=3, padding='valid', bias=False)", "self.fconv1 = nn.Conv1d(d1, d2, kernel_size=3, padding='valid', bias=False)")
sub("self.fdense2 = nn.Linear(128, 80*4, bias=False)", "self.fdense2 = nn.Linear(d2, sub_cond*4, bias=False)")
# --- subframe net
sub("def __init__(self, subframe_size=40, nb_subframes=4, cond_size=256, softquant=False):\n        super(FARGANSub",
    "def __init__(self, subframe_size=40, nb_subframes=4, cond_size=256, softquant=False, sub_cond=80, F=192, H1=160, H2=128, H3=128, S=128):\n        super(FARGANSub")
sub("self.cond_gain_dense = nn.Linear(80, 1)", "self.cond_gain_dense = nn.Linear(sub_cond, 1)")
sub("self.fwc0 = FWConv(2*self.subframe_size+80+4, 192, softquant=softquant)", "self.fwc0 = FWConv(2*self.subframe_size+sub_cond+4, F, softquant=softquant)")
sub("nn.GRUCell(192+2*self.subframe_size, 160, bias=False)", "nn.GRUCell(F+2*self.subframe_size, H1, bias=False)")
sub("nn.GRUCell(160+2*self.subframe_size, 128, bias=False)", "nn.GRUCell(H1+2*self.subframe_size, H2, bias=False)")
sub("nn.GRUCell(128+2*self.subframe_size, 128, bias=False)", "nn.GRUCell(H2+2*self.subframe_size, H3, bias=False)")
sub("self.gru1_glu = GLU(160, softquant=softquant)", "self.gru1_glu = GLU(H1, softquant=softquant)")
sub("self.gru2_glu = GLU(128, softquant=softquant)", "self.gru2_glu = GLU(H2, softquant=softquant)")
sub("self.gru3_glu = GLU(128, softquant=softquant)", "self.gru3_glu = GLU(H3, softquant=softquant)")
sub("self.skip_glu = GLU(128, softquant=softquant)", "self.skip_glu = GLU(S, softquant=softquant)")
sub("nn.Linear(192+160+2*128+2*self.subframe_size, 128, bias=False)", "nn.Linear(F+H1+H2+H3+2*self.subframe_size, S, bias=False)")
sub("self.sig_dense_out = nn.Linear(128, self.subframe_size, bias=False)", "self.sig_dense_out = nn.Linear(S, self.subframe_size, bias=False)")
sub("self.gain_dense_out = nn.Linear(192, 4)", "self.gain_dense_out = nn.Linear(F, 4)")
# --- full model
sub("def __init__(self, subframe_size=40, nb_subframes=4, feature_dim=20, cond_size=256, passthrough_size=0, has_gain=False, gamma=None, softquant=False):",
    "def __init__(self, subframe_size=40, nb_subframes=4, feature_dim=20, cond_size=256, passthrough_size=0, has_gain=False, gamma=None, softquant=False, sizes=None):")
sub("        self.cond_net = FARGANCond(feature_dim=feature_dim, cond_size=cond_size, softquant=softquant)\n        self.sig_net = FARGANSub(subframe_size=subframe_size, nb_subframes=nb_subframes, cond_size=cond_size, softquant=softquant)",
    "        z = dict(C=80, d1=64, d2=128, F=192, H1=160, H2=128, H3=128, S=128); z.update(sizes or {})\n        self.sub_cond = z['C']; self.H = (z['H1'], z['H2'], z['H3'])\n"
    "        self.cond_net = FARGANCond(feature_dim=feature_dim, cond_size=cond_size, softquant=softquant, sub_cond=z['C'], d1=z['d1'], d2=z['d2'])\n"
    "        self.sig_net = FARGANSub(subframe_size=subframe_size, nb_subframes=nb_subframes, cond_size=cond_size, softquant=softquant, sub_cond=z['C'], F=z['F'], H1=z['H1'], H2=z['H2'], H3=z['H3'], S=z['S'])")
sub("torch.zeros(batch_size, 160, device=device),\n            torch.zeros(batch_size, 128, device=device),\n            torch.zeros(batch_size, 128, device=device),\n            torch.zeros(batch_size, (2*self.subframe_size+80+4)*1, device=device)",
    "torch.zeros(batch_size, self.H[0], device=device),\n            torch.zeros(batch_size, self.H[1], device=device),\n            torch.zeros(batch_size, self.H[2], device=device),\n            torch.zeros(batch_size, (2*self.subframe_size+self.sub_cond+4)*1, device=device)")
sub("cond[:, n, k*80:(k+1)*80]", "cond[:, n, k*self.sub_cond:(k+1)*self.sub_cond]")
src = src.replace('os.path.join(source_dir, "../dnntools")', 'os.path.join(source_dir, "dnntools_src")')
pathlib.Path("fargan_small.py").write_text(src)
