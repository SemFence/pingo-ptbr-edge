"""val_spect (mesma perda do treino, sem o termo cosseno) de checkpoints em um conjunto (features, pcm).
  python3 eval_set.py FEATURES PCM NB_FEATURES ckpt1 ckpt2 ..."""
import sys, os, types, numpy as np, torch
sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fargan_small as fargan
from dataset import FARGANDataset
from stft_loss import MultiResolutionSTFTLoss
F, P, nf = sys.argv[1], sys.argv[2], int(sys.argv[3]); dev = torch.device("cpu")
ds = FARGANDataset(F, P, sequence_length=15, nb_features=nf); n = min(len(ds), 1024)
idx = np.linspace(0, len(ds) - 1, n).astype(int); sub = torch.utils.data.Subset(ds, idx)
dl = torch.utils.data.DataLoader(sub, batch_size=256, shuffle=False); spect = MultiResolutionSTFTLoss(dev)
for ck_path in sys.argv[4:]:
    ck = torch.load(ck_path, map_location="cpu"); m = fargan.FARGAN(softquant=ck.get("softquant", False), sizes=ck["sizes"]); m.load_state_dict(ck["state_dict"]); m.eval()
    tot = 0; k = 0
    with torch.no_grad():
        for feats, periods, target, lpc in dl:
            target = target[:, :2400]; feats = feats[:, :19]; periods = periods[:, :19]; pre = target[:, :320]
            sig, _ = m(feats, periods, 13, pre=pre); sig = torch.cat([pre, sig], -1)
            tot += spect(sig, target).item(); k += 1
    print(f"{ck_path}: val_spect {tot / k:.4f}  ({n} seq)", flush=True)
