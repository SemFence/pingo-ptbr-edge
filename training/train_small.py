"""Trains a reduced FARGAN (fargan_small.py) with the Opus recipe: spectral loss + 0.03 * cosine loss on the first frame,
AdamW, LR decay 1/(1+1e-4*step), 2 pre-roll frames. Adds a validation loss, a wall-clock budget and a resumable checkpoint."""
import sys, os, time, json, argparse, types, numpy as np, torch
sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))   # only used in commented-out lines of stft_loss.py
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fargan_small as fargan
from dataset import FARGANDataset
from stft_loss import MultiResolutionSTFTLoss
from sizes import SIZES

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="s", choices=list(SIZES))
ap.add_argument("--train", nargs=2, required=True, metavar=("FEATURES", "PCM"))
ap.add_argument("--val", nargs=2, required=True, metavar=("FEATURES", "PCM"))
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--max-minutes", type=float, default=1e9, help="stop (after saving) when this wall-clock budget is used up")
ap.add_argument("--max-steps", type=int, default=10**9)
ap.add_argument("--batch", type=int, default=2048)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--lr-decay", type=float, default=1e-4)
ap.add_argument("--softquant", action="store_true", help="soft quantization of the weights (8 bit), as in the Opus recipe")
ap.add_argument("--device", default="cuda")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--init", default=None, help="load only the weights of this checkpoint (fine-tuning: fresh optimizer and schedule)")
ap.add_argument("--cosine-steps", type=int, default=0, help="if >0: cosine LR decay to 0 over this many steps instead of 1/(1+k*step)")
ap.add_argument("--nb-features", type=int, default=36, help="floats per frame in the feature files (20 when the unused LPC columns were dropped)")
a = ap.parse_args()

SEQ = 15
dev = torch.device(a.device)
os.makedirs(a.out, exist_ok=True)
model = fargan.FARGAN(softquant=a.softquant, sizes=SIZES[a.size]).to(dev)
nparams = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=[0.8, 0.95], eps=1e-8)
import math
sched = torch.optim.lr_scheduler.LambdaLR(opt, (lambda s: 0.5 * (1 + math.cos(math.pi * min(s, a.cosine_steps) / a.cosine_steps))) if a.cosine_steps else (lambda s: 1 / (1 + a.lr_decay * s)))
spect = MultiResolutionSTFTLoss(dev).to(dev)
tr = FARGANDataset(*a.train, sequence_length=SEQ, nb_features=a.nb_features); va = FARGANDataset(*a.val, sequence_length=SEQ, nb_features=a.nb_features)
dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers, persistent_workers=a.workers > 0)
vdl = torch.utils.data.DataLoader(torch.utils.data.Subset(va, range(0, min(len(va), 1024))), batch_size=256, shuffle=False, num_workers=0)
step = epoch0 = 0; best = 1e9
last = os.path.join(a.out, "last.pth")
if a.init and not os.path.exists(last):
    model.load_state_dict(torch.load(a.init, map_location='cpu')['state_dict']); print('init from', a.init, flush=True)
if os.path.exists(last):
    ck = torch.load(last, map_location="cpu"); model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"])
    step, epoch0, best = ck["step"], ck["epoch"], ck["best"]; sched.last_epoch = step
    print(f"resumed at epoch {epoch0}, step {step}", flush=True)

def run(feats, periods, target, train):
    feats, periods, target = feats.to(dev), periods.to(dev), target.to(dev)
    target = target[:, :SEQ * 160]; feats = feats[:, :SEQ + 4]; periods = periods[:, :SEQ + 4]
    pre = target[:, :2 * 160]
    sig, _ = model(feats, periods, target.size(1) // 160 - 2, pre=pre)
    sig = torch.cat([pre, sig], -1)
    cont = fargan.sig_loss(target[:, 320:480], sig[:, 320:480]); sp = spect(sig, target.detach())
    return .03 * cont + sp, sp

log = open(os.path.join(a.out, "log.jsonl"), "a"); t0 = time.time()
print(f"{a.size}: {nparams} params | train {len(tr)} seq ({len(dl)} steps/epoch) | val {len(vdl.dataset)} seq | batch {a.batch}", flush=True)
stop = False
for epoch in range(epoch0 + 1, a.epochs + 1):
    model.train(); tl = 0; n = 0; te = time.time()
    for feats, periods, target, lpc in dl:
        opt.zero_grad(); loss, sp = run(feats, periods, target, True); loss.backward(); opt.step(); sched.step(); step += 1
        tl += loss.item(); n += 1
        if step % 50 == 0 or step <= 3: print(f"  step {step} loss {loss.item():.4f} {(time.time()-t0):.0f}s", flush=True)
        if step >= a.max_steps or (time.time() - t0) / 60 > a.max_minutes: stop = True; break
    model.eval(); vl = 0; vn = 0
    with torch.no_grad():
        for feats, periods, target, lpc in vdl:
            l, sp = run(feats, periods, target, False); vl += sp.item(); vn += 1
    vl /= max(vn, 1)
    rec = dict(epoch=epoch, step=step, train=tl / max(n, 1), val_spect=vl, lr=sched.get_last_lr()[0], epoch_s=round(time.time() - te, 1), minutes=round((time.time() - t0) / 60, 1))
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    ck = dict(state_dict=model.state_dict(), opt=opt.state_dict(), step=step, epoch=epoch, best=min(best, vl), size=a.size, sizes=SIZES[a.size], softquant=a.softquant)
    torch.save(ck, last)
    if vl < best: best = vl; torch.save(dict(state_dict=model.state_dict(), size=a.size, sizes=SIZES[a.size], epoch=epoch, val_spect=vl, softquant=a.softquant), os.path.join(a.out, "best.pth"))
    if stop: break
print("done", flush=True)
