"""Isolate the easiest sub-problem: one digit place.

With n = 1 there is no carry to propagate, but everything else is still in play -- the
code table, the generate threshold, the mod-10 fold and the carry slot.  If a
configuration cannot solve this, nothing downstream will work either.  Scores here are
over all 100 digit pairs, so they are not noisy.
"""
import argparse
import time

import torch

import ens


def all_pairs(device):
    a = torch.tensor([i for i in range(10) for _ in range(10)], device=device)
    b = torch.tensor([j for _ in range(10) for j in range(10)], device=device)
    tok = torch.zeros(100, 3, 2, dtype=torch.long, device=device)
    tok[:, 1, 0] = a
    tok[:, 1, 1] = b
    tgt = torch.zeros(100, 3, dtype=torch.long, device=device)
    tgt[:, 1] = (a + b) % 10
    tgt[:, 2] = (a + b) // 10
    return tok, tgt


def run(e, steps, lr, tau, seed, device, clip=1.0, log_every=500, verbose=True,
        margin_w=0.0, margin_target=0.5):
    tok, tgt = all_pairs(device)
    p = ens.init_params(e, device, seed=seed)
    params = list(p.values())
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.15)
    best = torch.zeros(e, device=device)
    bestp = {k: v.detach().clone() for k, v in p.items()}
    t0 = time.time()
    for s in range(steps):
        loss, exact, marg = ens.loss_and_acc(p, tok, tgt, tau=tau,
                                             margin_w=margin_w, margin_target=margin_target)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        with torch.no_grad():
            sq = sum(q.grad.pow(2).flatten(1).sum(1) for q in params)
            sc = (clip / (sq.sqrt() + 1e-12)).clamp(max=1.0)
            for q in params:
                q.grad.mul_(sc.view(-1, *([1] * (q.dim() - 1))))
        opt.step()
        sched.step()
        with torch.no_grad():
            # exact match first, geometric margin as a tie-break, so members that already
            # solve all 100 pairs keep pushing the read-out boundaries apart
            score = exact + 0.02 * marg.clamp(0.0, 2.0)
            upd = score > best
            best = torch.where(upd, score, best)
            for k in bestp:
                bestp[k] = torch.where(upd.view(-1, *([1] * (bestp[k].dim() - 1))),
                                       p[k].detach(), bestp[k])
        if verbose and ((s + 1) % log_every == 0 or s == steps - 1):
            print(f"  step {s+1:6d} loss {loss.mean().item():9.4f} "
                  f"best {exact.max().item():.3f} ever {best.max().item():.3f} "
                  f"n100 {(best >= 1.0).sum().item():5d}  {time.time()-t0:.0f}s",
                  flush=True)
    return best, bestp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin_w", type=float, default=1.0)
    ap.add_argument("--margin_target", type=float, default=0.8)
    ap.add_argument("--keep", type=int, default=512)
    ap.add_argument("--tag", type=str, default="phase1")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    b, bp = run(a.e, a.steps, a.lr, a.tau, a.seed, dev,
                margin_w=a.margin_w, margin_target=a.margin_target)
    tok, tgt = all_pairs(dev)
    _, ex, marg = ens.loss_and_acc(bp, tok, tgt)
    good = ex >= 1.0
    k = int(good.sum())
    print(f"members solving all 100 one-place cases: {k}/{a.e}")
    assert k, "phase 1 found nothing"

    keep = torch.argsort(-torch.where(good, marg, torch.full_like(marg, -1e9)))[:min(k, a.keep)]
    i = int(keep[0])
    code = [0.0, 1.0] + [round(float(x), 4) for x in bp["code_free"][i].tolist()]
    print("  best-margin member:")
    print("    code   ", code)
    print("    carry_w", round(float(bp["carry_w"][i]), 4),
          "knee", [round(float(x), 4) for x in bp["knee"][i].tolist()],
          "fold", round(float(bp["fold"][i]), 4), "margin", round(float(marg[i]), 4))
    out = f"ckpt/{a.tag}.pt"
    torch.save({"params": {kk: vv[keep].cpu() for kk, vv in bp.items()},
                "acc": ex[keep].cpu(), "margin": marg[keep].cpu(),
                "args": vars(a)}, out)
    print(f"saved {out} with {len(keep)} parents")
