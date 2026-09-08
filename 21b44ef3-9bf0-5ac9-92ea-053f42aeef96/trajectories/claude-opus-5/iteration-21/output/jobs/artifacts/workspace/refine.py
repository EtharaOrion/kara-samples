"""Stage 2: teach carrying to members that already have an additive code.

Parents come out of lottery.py knowing only how to add one place with no carry.
Their two gate thresholds are wherever they were born, because a saturated clamp
hands them no gradient, so this stage replicates each parent and redraws the
thresholds from a prior built out of the member's own token values -- a random
restart over the range the model actually operates on.  Then it trains on real
carry data, widening the problem in stages.

The residual scale (code[1]) is still free here; finish.py spends it.
"""

import argparse
import time

import torch

import lab
import train as T


def stages(text):
    out, acc = [], 0.0
    for part in text.split(","):
        w, f = part.split(":")
        acc += float(f)
        out.append((int(w), acc))
    return [(w, f / acc) for w, f in out]


def redraw(p, R, jitter, device):
    """Replicate each parent R times and restart its thresholds.

    The prior is uniform over the member's own token range: the values the gate
    input can actually take, given that member's code.  One replica per parent
    keeps the parent's own thresholds so nothing is ever lost by redrawing.
    """
    E = p["code_free"].shape[0]
    with torch.no_grad():
        code = lab.code_of(p)
        x = (code[:, :, None] + code[:, None, :]).reshape(E, -1)
        lo, hi = x.amin(1), x.amax(1)
        u = torch.rand(E, 2, device=device).sort(dim=1).values
        knee = lo[:, None] + u * (hi - lo)[:, None]
        knee[::R] = p["knee"][::R]
        p["knee"].copy_(knee)
        for k, v in p.items():
            n = torch.randn_like(v) * jitter
            n[::R] = 0
            if k == "knee":
                n.zero_()
            v.add_(n)
    return p


def train_staged(p, device, steps, lr, sched_stages, B, temp, clip=1.0,
                 log_every=500, snap=100, snap_from=0.5, pct_start=0.15):
    E = p["code_free"].shape[0]
    opt = torch.optim.AdamW(list(p.values()), lr=lr, betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                              pct_start=pct_start)
    bounds = [(w, int(f * steps)) for w, f in sched_stages]
    best = {k: v.detach().clone() for k, v in p.items()}
    best_score = torch.full((E,), -1.0, device=device)
    t0, si = time.time(), 0
    for step in range(steps):
        while si < len(bounds) - 1 and step >= bounds[si][1]:
            si += 1
        wmax = bounds[si][0]
        w = int(torch.randint(1, wmax + 1, (1,)).item())
        pairs, y = lab.batch(B, w, device, split="train")
        loss, ok = T.loss_and_acc(p, pairs, y, temp)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        sq = sum(v.grad.reshape(E, -1).pow(2).sum(1) for v in p.values())
        sc = (clip / (sq.sqrt() + 1e-8)).clamp(max=1.0)
        for v in p.values():
            v.grad.mul_(sc.view(E, *([1] * (v.dim() - 1))))
        opt.step()
        sch.step()
        if step >= snap_from * steps and w == wmax and step % snap == 0:
            with torch.no_grad():
                score = ok + 0.01 * T.gate_slack(p).clamp(-1, 1)
                better = score > best_score
                best_score = torch.where(better, score, best_score)
                for k, v in p.items():
                    m = better.view(E, *([1] * (v.dim() - 1)))
                    best[k] = torch.where(m, v.detach(), best[k])
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:6d} w<={wmax} loss {loss.mean().item():.4f} "
                  f"best {ok.max().item():.4f} n>=0.99 {(ok >= 0.99).sum().item():5d} "
                  f"{time.time() - t0:6.1f}s", flush=True)
    return best


@torch.no_grad()
def report(p, device, tag, widths=(8, 5, 12)):
    acc = T.evaluate(p, device, widths=widths, split="eval", B=2048)
    slack = T.gate_slack(p)
    up = lab.code_of(p)[:, 1] > 0
    usable = lab.is_exact(acc) & (slack >= 0) & up
    print(f"  [{tag}] members exact on every eval width: {lab.is_exact(acc).sum().item()}   "
          f"of those gate-saturated and ascending: {usable.sum().item()}   "
          f"best acc {acc.max().item():.5f}")
    return acc, slack, up


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", default="/workspace/parents1.pt")
    ap.add_argument("--R", type=int, default=24)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--B", type=int, default=192)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--temp", type=float, default=8.0)
    ap.add_argument("--jitter", type=float, default=0.01)
    ap.add_argument("--stages", default="1:0.15,2:0.25,3:0.20,5:0.15,8:0.25")
    ap.add_argument("--keep", type=int, default=512)
    ap.add_argument("--nparents", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/parents2.pt")
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(a.seed)
    ck = torch.load(a.parents)
    par = {k: v[:a.nparents].to(dev) for k, v in ck["params"].items()}
    n = par["code_free"].shape[0]
    p = {k: v.repeat_interleave(a.R, dim=0).clone().requires_grad_(True)
         for k, v in par.items()}
    print(f"stage 2: {n} parents x {a.R} = {n * a.R} members", flush=True)
    p = redraw(p, a.R, a.jitter, dev)
    p = train_staged(p, dev, a.steps, a.lr, stages(a.stages), a.B, a.temp)
    acc, slack, up = report(p, dev, "stage2")
    score = acc + 0.002 * slack.clamp(-2, 0) + 0.001 * up.float()
    keep = score.argsort(descending=True)[:a.keep]
    torch.save({"params": {k: v.detach()[keep].cpu() for k, v in p.items()},
                "acc": acc[keep].cpu(), "slack": slack[keep].cpu()}, a.out)
    print(f"  wrote {a.out}", flush=True)
