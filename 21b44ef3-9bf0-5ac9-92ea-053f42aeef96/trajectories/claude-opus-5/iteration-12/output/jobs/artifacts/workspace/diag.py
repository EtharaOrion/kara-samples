"""Diagnostic: map the accuracy landscape over the two clamp knees.

Takes the code ramps that phase-1 training already learned and asks, for a
fixed code, what n=8 accuracy every possible pair of knee positions gives.
This says whether the good basin exists at all for a learned code, and how
wide it is -- i.e. whether the blocker is expressivity or search.

Diagnostic only; nothing here writes weights or feeds the submission.
"""

import argparse
import torch
from torch.func import functional_call, vmap

import data
from adder import DigitPairAdder


def landscape(code_free, grid=96, batch=256, n=8, device="cuda", seed=7):
    base = DigitPairAdder().to("meta")
    bw = torch.tensor([-2.0, 2.0], device=device)
    xmax = 2.0 * float(torch.cat([torch.zeros(1), code_free.cpu()]).max())
    ax = torch.linspace(0.0, xmax, grid, device=device)
    k0, k1 = torch.meshgrid(ax, ax, indexing="ij")
    knees = torch.stack([k0.reshape(-1), k1.reshape(-1)], -1)      # (G, 2)
    g = knees.shape[0]
    params = {
        "code_free": code_free.to(device).unsqueeze(0).expand(g, 9).contiguous(),
        "bank_bias": (-bw * knees).contiguous(),
        "fold": torch.zeros(g, device=device),
    }
    buffers = {k: v.to(device).unsqueeze(0).expand(g, *v.shape).contiguous()
               for k, v in DigitPairAdder().named_buffers()}

    def fwd(p, b, da, db):
        return functional_call(base, (p, b), (da, db))

    batched = vmap(fwd, in_dims=(0, 0, None, None))
    gen = torch.Generator(device=device).manual_seed(seed)
    da, db, tgt, _ = data.sample(batch, n, device, gen)

    # `fold` is the one remaining free parameter; sweep it coarsely so the
    # landscape is over the knees with fold at its best value, not at zero.
    step = float(code_free[0])
    best = torch.zeros(g, device=device)
    best_fold = torch.zeros(g, device=device)
    for f in torch.linspace(-14.0 * step, 0.0, 57, device=device):
        params["fold"] = torch.full((g,), float(f), device=device)
        with torch.no_grad():
            pred = batched(params, buffers, da, db).argmax(-1)
        acc = (pred[..., 1:] == tgt[..., 1:].unsqueeze(0)).all(-1).float().mean(-1)
        upd = acc > best
        best = torch.where(upd, acc, best)
        best_fold = torch.where(upd, torch.full_like(best_fold, float(f)), best_fold)
    return knees, best, best_fold, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/p1.pt")
    ap.add_argument("--members", type=int, default=6)
    ap.add_argument("--grid", type=int, default=96)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cf = ck["params"]["code_free"]
    for m in range(args.members):
        code = cf[m]
        steps = torch.diff(torch.cat([torch.zeros(1), code]))
        knees, acc, fold, ax = landscape(code, grid=args.grid)
        i = int(acc.argmax())
        frac = float((acc > 0.99).float().mean())
        print(f"member {m}: code step mean {steps.mean():.4f} sd {steps.std():.4f} "
              f"| best acc {acc.max():.4f} at knees "
              f"({knees[i,0]:.3f}, {knees[i,1]:.3f}) fold {fold[i]:.3f} "
              f"| ideal knees ({9.5*steps.mean():.3f}, {8.5*steps.mean():.3f}) "
              f"| grid frac >0.99: {frac:.5f}")


if __name__ == "__main__":
    main()
