"""Simulated post-training quantization: symmetric per-row (output channel) int8 weights, converted back to float.
  python3 ptq_int8.py in.pth out.pth [--bits 8]   (reports size and error per tensor)"""
import sys, torch
src, dst = sys.argv[1], sys.argv[2]; bits = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 8
ck = torch.load(src, map_location="cpu"); sd = ck["state_dict"]; new = dict(sd); qmax = 2 ** (bits - 1) - 1
def q(w):
    w2 = w.reshape(w.shape[0], -1); s = (w2.abs().amax(1, keepdim=True) / qmax).clamp_min(1e-12)
    return ((w2 / s).round().clamp(-qmax, qmax) * s).reshape(w.shape)
tot = 0; rep = []
for k in list(sd):
    if k.endswith("weight_v"):
        g = sd[k.replace("weight_v", "weight_g")]; v = sd[k]
        w = g * v / v.reshape(v.shape[0], -1).norm(dim=1).reshape(-1, *[1] * (v.dim() - 1))
        wq = q(w); new[k] = wq; new[k.replace("weight_v", "weight_g")] = wq.reshape(wq.shape[0], -1).norm(dim=1).reshape(g.shape)
        rep.append((k, ((wq - w).norm() / w.norm()).item())); tot += w.numel()
    elif sd[k].dim() >= 2 and not k.endswith("weight_g") and "pembed" not in k:
        w = sd[k]; wq = q(w); new[k] = wq; rep.append((k, ((wq - w).norm() / w.norm()).item())); tot += w.numel()
    elif "pembed" in k: tot += sd[k].numel()
for k, e in rep: print(f"{k:42s} erro rel {e*100:.2f}%")
print(f"pesos quantizados: {tot} -> {tot*bits/8/1024:.1f} KiB a {bits} bits (+ escalas por linha)")
ck = dict(ck); ck["state_dict"] = new; torch.save(ck, dst)
