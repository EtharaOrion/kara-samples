"""Diagnose what a multi-place run actually learns: per-place-count accuracy,
per-position accuracy, and the key/value profile as a function of a+b."""
import torch

import arch
import data
import lab
import probe

dev = "cuda"
EV = [8, 5, 11, 3]
cfg = arch.default_cfg(C=2, U=2, act="clamp")

for places, tag in [([2], "n=2 only"), ([3], "n=3 only"), ([8], "n=8 only"),
                    ([1, 2], "n=1,2"), ([1, 2, 3], "n=1,2,3")]:
    best, w = probe.run(cfg, 256, 3000, 512, 0.012, 0, places, cur="flat")
    print(f"train {tag:10s} -> eval same places: max {best.max().item():.4f} "
          f"#>=.99 {(best>=0.99).sum().item()}", flush=True)

# per-position breakdown for the flat multi-place run
best, w = probe.run(cfg, 256, 3000, 512, 0.012, 0, EV, cur="flat")
i = int(best.argmax())
p1 = {k: v[i:i + 1] for k, v in w.items()}
g = torch.Generator(device=dev).manual_seed(5)
for n in [1, 2, 3, 5, 8]:
    a, b = data.sample(4096, n, dev, gen=g, held_out=True)
    tok, tgt = data.tokens(a, b), data.targets(a, b)
    lg = arch.forward(p1, cfg, tok)
    pred = lg.argmax(-1)[0]
    valid = tgt >= 0
    per_pos = ((pred == tgt) | ~valid).float().mean(0)
    em = (((pred == tgt) | ~valid).all(-1)).float().mean()
    print(f"n={n:2d} exact {em:.4f} per-position " +
          " ".join(f"{v:.2f}" for v in per_pos.tolist()), flush=True)

# key / value as a function of a+b for the best member
a = torch.arange(10, device=dev).repeat_interleave(10)
b = torch.arange(10, device=dev).repeat(10)
tokg = torch.stack([a, b], -1)[None]           # (1,100,2)
_, parts = arch.forward(p1, cfg, tokg, return_parts=True)
s = (a + b).cpu()
k = parts["k"][0, 0].detach().cpu()
v = parts["v"][0, 0].detach().cpu()
print("\n  s :  mean key   mean val")
for ss in range(19):
    m = s == ss
    print(f" {ss:2d} : {k[m].mean():9.3f} {v[m].mean():9.3f}")
print("lam", parts["w"]["lam"].item(), "uA", parts["w"]["uA"].tolist(),
      "uB", parts["w"]["uB"].tolist())
