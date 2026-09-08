"""How many independent members find an additive digit code?

Everything else in this model is built on top of a code with code[a] + code[b]
close to code[a+b].  That constraint is rigid -- up to scale there is only one
family that satisfies it -- so most members never find it and no amount of
later training rescues them.  This measures the yield directly, on the smallest
sub-problem that asks for it: one place, no carries, full batch.

Members that come out with a clean ramp are the parents for everything after.
"""

import argparse
import time

import torch

import lab
import train as T


def carry_free_batch(device):
    """All 55 one-place problems with a + b <= 9, as a fixed full batch."""
    a, b = [], []
    for i in range(10):
        for j in range(10 - i):
            a.append(i)
            b.append(j)
    a = torch.tensor(a, device=device).unsqueeze(1)
    b = torch.tensor(b, device=device).unsqueeze(1)
    return torch.stack([a, b], dim=-1), lab.answer_digits(a, b)


def ramp_stats(code):
    """Slope and worst deviation from the best straight line, in slope units."""
    d = torch.arange(10.0, device=code.device)
    dc = d - d.mean()
    slope = ((code - code.mean(1, keepdim=True)) * dc).sum(1) / dc.pow(2).sum()
    fit = slope[:, None] * dc + code.mean(1, keepdim=True)
    return slope, (code - fit).abs().amax(1) / slope.abs().clamp_min(1e-9)


def run(E, steps, lr, temp, device, seed, B=0, wd=0.0, log_every=1000):
    torch.manual_seed(seed)
    p = lab.init_params(E, device)
    pairs_full, y_full = carry_free_batch(device)
    opt = torch.optim.AdamW(list(p.values()), lr=lr, betas=(0.9, 0.99), weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.2)
    t0 = time.time()
    for step in range(steps):
        if B:
            pairs, y = lab.batch(B, 1, device, split="train", nocarry=1.0)
        else:
            pairs, y = pairs_full, y_full
        loss, ok = T.loss_and_acc(p, pairs, y, temp)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        sq = sum(v.grad.reshape(E, -1).pow(2).sum(1) for v in p.values())
        sc = (1.0 / (sq.sqrt() + 1e-8)).clamp(max=1.0)
        for v in p.values():
            v.grad.mul_(sc.view(E, *([1] * (v.dim() - 1))))
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"    step {step:5d} loss {loss.mean().item():.4f} "
                  f"exact {lab.is_exact(ok).sum().item():5d} {time.time() - t0:5.1f}s", flush=True)
    with torch.no_grad():
        _, ok = T.loss_and_acc(p, pairs_full, y_full, temp)
        slope, rel = ramp_stats(lab.code_of(p))
    return p, ok, slope, rel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--temp", type=float, default=8.0)
    ap.add_argument("--B", type=int, default=0)          # 0 = full batch
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", type=int, default=1024)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = "cuda"
    p, ok, slope, rel = run(a.E, a.steps, a.lr, a.temp, dev, a.seed, B=a.B, wd=a.wd)
    good = lab.is_exact(ok) & (rel < 0.02) & (slope.abs() > 0.05)
    print(f"  E={a.E} steps={a.steps} lr={a.lr} temp={a.temp} B={a.B or 'full'} wd={a.wd}")
    print(f"    exact on all 55 carry-free pairs : {lab.is_exact(ok).sum().item()}")
    print(f"    exact and a clean ramp           : {good.sum().item()}")
    print(f"    best relative ramp residual      : {rel.min().item():.5f}")
    if a.out:
        score = lab.is_exact(ok).float() - rel.clamp(max=1.0)
        keep = score.argsort(descending=True)[:a.keep]
        torch.save({"params": {k: v.detach()[keep].cpu() for k, v in p.items()},
                    "ok": ok[keep].cpu(), "rel": rel[keep].cpu()}, a.out)
        print(f"    wrote {a.out}  ({keep.numel()} parents, "
              f"{(good[keep]).sum().item()} of them clean)")
