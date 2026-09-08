"""Two-phase random-restart training of the digit-pair adder.

Phase 1 trains a large ensemble of independent members on one-place problems.
That identifies the digit code, the carry write and the mod-10 fold, but leaves
the *transparent* class boundary (knee 0) unidentified, because with a single
place a carry never has to travel through anything.

Phase 2 replicates the phase-1 winners, redraws knee 0 from a data-driven prior
(the gate bank is saturated, so that threshold has no gradient and has to be
found by restart), and trains on several widths at once.

Usage:  python train.py phase1 ...   /   python train.py phase2 ...
"""

import argparse
import math
import time

import torch

import lab


def loss_and_acc(p, pairs, y, temp, square=False, norm=True):
    """Cross-entropy per member over the answer positions, plus exact match."""
    logits = lab.ens_forward(p, pairs, square=square, temp=temp, norm=norm)
    logits = logits[:, :, 1:, :]
    logp = torch.log_softmax(logits, dim=-1)
    tgt = y[None, :, :, None].expand(logp.shape[0], -1, -1, 1)
    picked = logp.gather(-1, tgt).squeeze(-1)                      # (E, B, P-1)
    loss = -picked.mean(dim=(1, 2))
    with torch.no_grad():
        ok = (logits.argmax(-1) == y[None]).all(-1).float().mean(1)
    return loss, ok


@torch.no_grad()
def evaluate(p, device, widths=(8,), split="eval", B=4096, reps=1, square=True):
    """Exact-match rate per member, averaged over the requested widths."""
    E = p["code_free"].shape[0]
    tot = torch.zeros(E, device=device)
    n = 0
    for w in widths:
        for _ in range(reps):
            pairs, y = lab.batch(B, w, device, split=split)
            chunk = max(1, int(2e8 // max(1, pairs.shape[0] * (w + 2) ** 2)))
            acc = torch.zeros(E, device=device)
            for i in range(0, E, chunk):
                sub = {k: v[i:i + chunk] for k, v in p.items()}
                logits = lab.ens_forward(sub, pairs, square=square)[:, :, 1:, :]
                acc[i:i + chunk] = (logits.argmax(-1) == y[None]).all(-1).float().mean(1)
            tot += acc
            n += 1
    return tot / n


@torch.no_grad()
def gate_slack(p):
    """How far the gate bank is from saturation on every possible token.

    >= 0 means every one of the 100 digit pairs (and the (0,0) pad) drives both
    clamp units fully to 0 or 1, so key and value depend only on the carry class.
    """
    code = lab.code_of(p)                                  # (E, 10)
    x = code[:, :, None] + code[:, None, :]                # (E, 10, 10)
    x = torch.cat([x.reshape(code.shape[0], -1), torch.zeros_like(code[:, :1])], dim=1)
    t = lab.GATE_SLOPE * (x[:, :, None] - p["knee"][:, None, :])
    return torch.maximum(-t, t - 1).amin(-1).amin(-1)


def train(p, device, steps, lr, widths, B, temp, clip=1.0, pct_start=0.3,
          log_every=500, track_best=None, seed=0, warm_frac=0.0, nocarry_frac=0.0):
    opt = torch.optim.AdamW(list(p.values()), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=pct_start)
    E = p["code_free"].shape[0]
    best_score = torch.full((E,), -1.0, device=device)
    best = {k: v.detach().clone() for k, v in p.items()} if track_best else None
    g = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    for step in range(steps):
        w = 1 if step < warm_frac * steps else widths[step % len(widths)]
        # curriculum: start on problems that never carry, so the only thing to
        # learn is a digit code that adds, then fade the carries back in
        nc = 0.0
        if nocarry_frac > 0:
            nc = max(0.0, 1.0 - step / (nocarry_frac * steps))
        pairs, y = lab.batch(B, w, device, split="train", nocarry=nc)
        loss, ok = loss_and_acc(p, pairs, y, temp)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        sq = torch.zeros(E, device=device)
        for v in p.values():
            gr = v.grad.reshape(E, -1)
            sq = sq + gr.pow(2).sum(1)
        scale = (clip / (sq.sqrt() + 1e-8)).clamp(max=1.0)
        for v in p.values():
            v.grad.mul_(scale.view(E, *([1] * (v.dim() - 1))))
        opt.step()
        sched.step()
        if track_best and (step % track_best == 0 or step == steps - 1):
            with torch.no_grad():
                score = ok + 0.01 * gate_slack(p).clamp(-1, 1)
                better = score > best_score
                best_score = torch.where(better, score, best_score)
                for k, v in p.items():
                    m = better.view(E, *([1] * (v.dim() - 1)))
                    best[k] = torch.where(m, v.detach(), best[k])
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:6d}  w={w}  loss {loss.mean().item():.4f}"
                  f"  best-batch-acc {ok.max().item():.4f}"
                  f"  n>=0.99 {(ok >= 0.99).sum().item():5d}"
                  f"  {time.time() - t0:6.1f}s", flush=True)
        del pairs, y, loss, ok
    return best if track_best else {k: v.detach() for k, v in p.items()}


def phase1(args):
    dev = "cuda"
    torch.manual_seed(args.seed)
    p = lab.init_params(args.E, dev)
    widths = [int(w) for w in args.widths.split(",")]
    print(f"phase 1: E={args.E} members, widths {widths}, warm {args.warm_frac}")
    p = train(p, dev, args.steps, args.lr, widths, args.B, args.temp,
              track_best=args.snap, seed=args.seed, warm_frac=args.warm_frac,
              nocarry_frac=args.nocarry_frac)
    pairs, y = lab.all_single_place(dev)
    acc = torch.zeros(args.E, device=dev)
    chunk = 8192
    for i in range(0, args.E, chunk):
        sub = {k: v[i:i + chunk] for k, v in p.items()}
        logits = lab.ens_forward(sub, pairs)[:, :, 1:, :]
        acc[i:i + chunk] = (logits.argmax(-1) == y[None]).all(-1).float().mean(1)
    order = torch.argsort(acc, descending=True)
    keep = order[:args.keep]
    print("  acc1 quantiles", [round(acc.quantile(q).item(), 3)
                               for q in (0.5, 0.9, 0.99, 0.999, 1.0)])
    print(f"  exact on all 100 one-place problems: {lab.is_exact(acc).sum().item()} / {args.E}")
    print(f"  keeping {keep.numel()}, worst kept acc {acc[keep][-1].item():.4f}")
    out = {k: v[keep].contiguous().cpu() for k, v in p.items()}
    torch.save({"params": out, "acc1": acc[keep].cpu()}, args.out)
    print(f"  wrote {args.out}")


def phase2(args):
    dev = "cuda"
    torch.manual_seed(args.seed)
    ck = torch.load(args.parents)
    par = {k: v.to(dev) for k, v in ck["params"].items()}
    n_par = par["code_free"].shape[0]
    R = args.replicas
    p = {k: v.repeat_interleave(R, dim=0).clone() for k, v in par.items()}
    E = n_par * R
    print(f"phase 2: {n_par} parents x {R} replicas = {E} members")

    # Redraw the transparent-class threshold: the gate bank is saturated, so it
    # gets no gradient and has to be found by restart.  Two priors are mixed:
    # a step below the other knee (the residual unit is pinned to 1), and a
    # uniform draw over the member's own token range.
    with torch.no_grad():
        code = lab.code_of(p)
        x = (code[:, :, None] + code[:, None, :]).reshape(E, -1)
        lo, hi = x.amin(1), x.amax(1)
        u = torch.rand(E, device=dev)
        near = p["knee"][:, 1] - torch.rand(E, device=dev) * 2.5 - 0.02
        wide = lo + torch.rand(E, device=dev) * (hi - lo)
        knee0 = torch.where(u < 0.7, near, wide)
        knee0[::R] = p["knee"][::R, 0]        # keep one replica per parent untouched
        p["knee"][:, 0] = knee0
        for k, v in p.items():                # a little jitter everywhere else
            noise = torch.randn_like(v) * args.jitter
            noise[::R] = 0
            if k == "knee":
                noise[:, 0] = 0
            v.add_(noise)
    p = {k: v.requires_grad_(True) for k, v in p.items()}

    widths = [int(w) for w in args.widths.split(",")]
    p = train(p, dev, args.steps, args.lr, widths, args.B, args.temp,
              pct_start=args.pct_start, track_best=args.snap, seed=args.seed + 1,
              nocarry_frac=args.nocarry_frac)
    acc = evaluate(p, dev, widths=(8, 5, 12), split="eval", B=args.evalB)
    slack = gate_slack(p)
    score = acc + 0.001 * slack.clamp(-1, 0)
    order = torch.argsort(score, descending=True)
    keep = order[:args.keep]
    print(f"  members exact on every eval width: {lab.is_exact(acc).sum().item()} / {E}")
    print(f"  of those, gate-saturated: {(lab.is_exact(acc) & (slack >= 0)).sum().item()}")
    print(f"  best kept: acc {acc[keep][0].item():.6f} slack {slack[keep][0].item():.4f}")
    out = {k: v.detach()[keep].contiguous().cpu() for k, v in p.items()}
    torch.save({"params": out, "acc": acc[keep].cpu(), "slack": slack[keep].cpu()}, args.out)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a1 = sub.add_parser("phase1")
    a1.add_argument("--E", type=int, default=65536)
    a1.add_argument("--steps", type=int, default=3000)
    a1.add_argument("--B", type=int, default=128)
    a1.add_argument("--lr", type=float, default=0.012)
    a1.add_argument("--temp", type=float, default=4.0)
    a1.add_argument("--keep", type=int, default=512)
    a1.add_argument("--snap", type=int, default=100)
    a1.add_argument("--seed", type=int, default=0)
    a1.add_argument("--widths", default="1")
    a1.add_argument("--warm_frac", type=float, default=0.0)
    a1.add_argument("--nocarry_frac", type=float, default=0.0)
    a1.add_argument("--out", default="/workspace/p1.pt")
    a1.set_defaults(fn=phase1)

    a2 = sub.add_parser("phase2")
    a2.add_argument("--parents", default="/workspace/p1.pt")
    a2.add_argument("--replicas", type=int, default=32)
    a2.add_argument("--steps", type=int, default=4000)
    a2.add_argument("--B", type=int, default=256)
    a2.add_argument("--lr", type=float, default=0.004)
    a2.add_argument("--pct_start", type=float, default=0.1)
    a2.add_argument("--temp", type=float, default=4.0)
    a2.add_argument("--jitter", type=float, default=0.02)
    a2.add_argument("--nocarry_frac", type=float, default=0.0)
    a2.add_argument("--widths", default="8,3,5,2")
    a2.add_argument("--keep", type=int, default=256)
    a2.add_argument("--snap", type=int, default=100)
    a2.add_argument("--evalB", type=int, default=2048)
    a2.add_argument("--seed", type=int, default=0)
    a2.add_argument("--out", default="/workspace/p2.pt")
    a2.set_defaults(fn=phase2)

    args = ap.parse_args()
    args.fn(args)
