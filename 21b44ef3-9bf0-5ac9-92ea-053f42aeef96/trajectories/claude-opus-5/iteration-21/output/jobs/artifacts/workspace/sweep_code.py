"""How often does a cold member learn an additive digit code?

Trains on carry-free problems only (a + b <= 9 in every place), which is the
easy end of the curriculum: nothing about carries, just "the code must add".
Counts members whose code comes out a clean ramp, which is the prerequisite for
everything else.
"""
import itertools
import torch

import lab
import train as T

dev = "cuda"
E = 4096
STEPS = 2500


def ramp_stats(p):
    code = lab.code_of(p)
    d = torch.arange(10., device=code.device)
    slope = (((code - code.mean(1, keepdim=True)) * (d - d.mean())).sum(1)
             / ((d - d.mean()) ** 2).sum())
    fit = slope[:, None] * (d - d.mean()) + code.mean(1, keepdim=True)
    resid = (code - fit).abs().amax(1)
    rel = resid / slope.abs().clamp_min(1e-6)
    return slope, rel


for square, temp, norm, lsr0 in itertools.product(
        (False, True), (0.5, 2.0, 8.0), (True,), (0.0,)):
    torch.manual_seed(3)
    p = lab.init_params(E, dev)
    with torch.no_grad():
        p["lsr"].fill_(lsr0)
    opt = torch.optim.AdamW(list(p.values()), lr=0.012, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.012, total_steps=STEPS)
    for step in range(STEPS):
        pairs, y = lab.batch(128, 1, dev, split="train", nocarry=1.0)
        loss, ok = T.loss_and_acc(p, pairs, y, temp, square=square, norm=norm)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        sq = sum(v.grad.reshape(E, -1).pow(2).sum(1) for v in p.values())
        sc = (1.0 / (sq.sqrt() + 1e-8)).clamp(max=1.0)
        for v in p.values():
            v.grad.mul_(sc.view(E, *([1] * (v.dim() - 1))))
        opt.step()
        sched.step()
    slope, rel = ramp_stats(p)
    good = (rel < 0.02) & (slope.abs() > 0.05)
    print(f"square={int(square)} temp={temp:4} norm={int(norm)} lsr0={lsr0}: "
          f"clean ramps {good.sum().item():5d}/{E}  "
          f"loss {loss.mean().item():.3f}  best rel-resid {rel.min().item():.4f}")
