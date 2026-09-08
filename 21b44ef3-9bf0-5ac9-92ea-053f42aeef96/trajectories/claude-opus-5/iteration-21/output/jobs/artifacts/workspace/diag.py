"""Diagnostics for phase-1 training with a free residual scale."""
import torch
import lab
import train as T

dev = "cuda"
E = 8192
torch.manual_seed(1)
p = lab.init_params(E, dev)
p = T.train(p, dev, 3000, 0.012, [1], 128, 4.0, log_every=1000, nocarry_frac=0.35)

pairs, y = lab.all_single_place(dev)
lg = lab.ens_forward(p, pairs)[:, :, 1:, :]
acc = (lg.argmax(-1) == y[None]).all(-1).float().mean(1)
print(f"exact on all 100 one-place problems: {lab.is_exact(acc).sum().item()}/{E}"
      f"  best {acc.max().item():.3f}  q99 {acc.quantile(0.99).item():.3f}")

code = lab.code_of(p)
d = torch.arange(10., device=dev)
slope = ((code - code.mean(1, keepdim=True)) * (d - d.mean())).sum(1) / ((d - d.mean()) ** 2).sum()
resid = (code - (slope[:, None] * (d - d.mean()) + code.mean(1, keepdim=True))).abs().amax(1)
lin = (resid < 0.03 * slope.abs()) & (slope > 0.05)
print(f"members whose code is a clean ramp: {lin.sum().item()}/{E}")
sel = lab.is_exact(acc).nonzero().flatten()[:5]
for i in sel:
    print("  exact member", i.item(), "code",
          [round(v, 3) for v in code[i].tolist()],
          "knee", [round(v, 3) for v in p["knee"][i].tolist()],
          "carry_w", round(p["carry_w"][i].item(), 3),
          "fold", round(p["fold"][i].item(), 3))
if len(sel) == 0:
    for i in resid.argsort()[:4]:
        print("  linear member code", [round(v, 3) for v in code[i].tolist()],
              "slope", round(slope[i].item(), 3), "acc", round(acc[i].item(), 3),
              "knee", [round(v, 3) for v in p["knee"][i].tolist()],
              "fold", round(p["fold"][i].item(), 3))
