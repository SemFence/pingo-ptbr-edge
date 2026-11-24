"""Candidate sizes and their per-second cost (MAC = multiply-accumulate; every sub-net weight is used once per 40-sample subframe, 400 subframes/s)."""
import sys, os, types
sys.modules["torchaudio"] = types.ModuleType("torchaudio")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fargan_small as fs
SIZES = {
  "orig": dict(),
  "s":  dict(C=48, d1=48, d2=96, F=96, H1=80, H2=64, H3=64, S=64),
  "xs": dict(C=32, d1=40, d2=64, F=64, H1=64, H2=48, H3=48, S=48),
  "xxs": dict(C=24, d1=32, d2=48, F=48, H1=48, H2=40, H3=40, S=40),
}
if __name__ == "__main__":
    import contextlib, io
    for k, z in SIZES.items():
        with contextlib.redirect_stdout(io.StringIO()):
            m = fs.FARGAN(sizes=z)
        sub = sum(p.numel() for n, p in m.sig_net.named_parameters() if p.dim() == 2)
        cond = sum(p.numel() for n, p in m.cond_net.named_parameters() if p.dim() >= 2 and "pembed" not in n)
        tot = sum(p.numel() for p in m.parameters())
        print(f"{k:5s} params {tot:7d} | subframe MAC {sub:7d} -> {sub*400/1e6:6.1f} MMAC/s (+cond {cond*100/1e6:.1f}) | int8 weights {tot/1024:.0f} KiB")
