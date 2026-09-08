"""Focused study of phase 2 (learning the transparency notch)."""
import time

import torch

import arch
import lab
import two_phase as tp

dev = "cuda"
cfg = arch.default_cfg(C=2, U=2, act="clamp", fix_val_w=(0.0, 1.0))
print("free scalars / member =", arch.n_free_values(cfg), flush=True)

t0 = time.time()
b1, w1, keys = tp.phase1(8192, 3000, 0, cfg)
good = (b1 >= 0.9999).nonzero().flatten()
print(f"phase1: {good.numel()}/8192 exact on n=1  ({time.time()-t0:.0f}s)",
      flush=True)
torch.save(dict(params={k: v.cpu() for k, v in w1.items()}, cfg=cfg,
                acc=b1.cpu(), good=good.cpu()), "phase1.pt")

E2 = 4096
rep = (E2 + good.numel() - 1) // good.numel()
idx = good.repeat(rep)[:E2]
base = {k: v[idx].contiguous() for k, v in w1.items()}
corners = lab.make_eval([8, 5, 11, 3], dev, per_n=1024)

for lr2 in [0.015, 0.03]:
    for ramp in [0.0, 0.4]:
        for steps in [4000]:
            p2 = tp.reinit_unit0(base, cfg, 11)
            t0 = time.time()
            b2, _ = tp.train(cfg, p2, keys, steps, 512, lr2, 3, corners,
                             [2, 3, 5, 8], pct=0.05, ramp=ramp)
            print(f"lr2={lr2} ramp={ramp} steps={steps}: "
                  f"max {b2.max().item():.4f} "
                  f"#>=.9 {(b2>=0.9).sum().item():4d} "
                  f"#>=.99 {(b2>=0.99).sum().item():4d} "
                  f"#>=.999 {(b2>=0.999).sum().item():4d} "
                  f"({time.time()-t0:.0f}s)", flush=True)
