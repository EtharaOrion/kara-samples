"""Train the digit-pair adder as a large ensemble of independent random restarts.

Every member has the same 12 free values and its own random initialisation; they share a
batch and are advanced by one set of kernel launches.  Members never interact -- Adam is
elementwise and gradients are clipped per member -- so this is exactly E independent runs.

The run is done in two phases because of the shape of the loss surface.  Once the clamp
bank saturates, the loss is piecewise constant in the two thresholds and their gradient is
identically zero, so they can only be found by restarting.  Phase 1 trains on a single
digit place, where only one threshold has to be right, and produces members with a correct
code table, carry weight and mod-10 fold.  Phase 2 replicates those members, re-randomises
the second threshold, and trains on many places, where carries actually have to travel.

    python train.py --phase 1 --e 8192 --steps 8000 --lr 0.12 --tag p1
    python train.py --phase 2 --init_from ckpt/p1.pt --replicas 32 --tag p2
"""
import argparse
import json
import os
import time

import torch

import data
import ens

WORK = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def evaluate(p, batches, chunk=128):
    """Exact-match rate and worst read-out margin per member over a fixed held-out set."""
    e = p["code_free"].shape[0]
    dev = p["code_free"].device
    hits = torch.zeros(e, device=dev)
    marg = torch.full((e,), float("inf"), device=dev)
    total = 0
    for tok, tgt in batches:
        for i in range(0, tok.shape[0], chunk):
            t, y = tok[i:i + chunk], tgt[i:i + chunk]
            _, exact, m = ens.loss_and_acc(p, t, y)
            hits += exact * t.shape[0]
            marg = torch.minimum(marg, m)
        total += tok.shape[0]
    return hits / total, marg


def clip_and_step(params, opt, sched, clip):
    with torch.no_grad():
        sq = sum(q.grad.pow(2).flatten(1).sum(1) for q in params)
        sc = (clip / (sq.sqrt() + 1e-12)).clamp(max=1.0)
        for q in params:
            q.grad.mul_(sc.view(-1, *([1] * (q.dim() - 1))))
    opt.step()
    sched.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--e", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.12)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--margin_w", type=float, default=1.0)
    ap.add_argument("--margin_target", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--places", type=str, default="1,2,3,5,8")
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--init_from", type=str, default="")
    ap.add_argument("--replicas", type=int, default=32)
    ap.add_argument("--keep", type=int, default=1024)
    ap.add_argument("--knee_lo", type=float, default=-1.0)
    ap.add_argument("--knee_hi", type=float, default=20.0)
    ap.add_argument("--redraw_carry", type=int, default=1)
    ap.add_argument("--redraw_knee", type=int, default=1)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed + 1)
    places = [int(x) for x in args.places.split(",")]

    if args.phase == 1:
        p = ens.init_params(args.e, dev, seed=args.seed)
        e = args.e
    else:
        src = torch.load(os.path.join(WORK, args.init_from), map_location=dev)
        sp = src["params"]
        w = sp["code_free"].shape[0]
        e = w * args.replicas
        p = {k: v.repeat_interleave(args.replicas, dim=0).clone() for k, v in sp.items()}
        # The lower threshold is what separates carry-transparent places from absorbing
        # ones.  A single place never needs it, so phase 1 leaves it anywhere; draw it
        # fresh across the whole span the residual stream occupies.
        if args.redraw_knee:
            u = torch.rand(e, device=dev, generator=g)
            p["knee"] = p["knee"].clone()
            p["knee"][:, 0] = args.knee_lo + u * (args.knee_hi - args.knee_lo)
        if args.redraw_carry:
            # A single place also cannot separate the carry-in write from the mod-10
            # write-back: at the carry slot both heads land on the same generating place,
            # so phase 1 only ever pins their sum.  Draw the carry write fresh from the
            # same prior the run started with rather than inheriting a value that is only
            # meaningful in combination with the fold.
            p["carry_w"] = (torch.rand(e, 1, device=dev, generator=g) * 4.0 - 2.0)
        p = {k: v.contiguous().requires_grad_(True) for k, v in p.items()}
        print(f"phase 2: {w} parents x {args.replicas} replicas = {e} members")

    params = list(p.values())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.15)

    ev = [data.mixed_batch(1536, 8, dev, g, want_holdout=True),
          data.mixed_batch(768, 5, dev, g, want_holdout=True),
          data.mixed_batch(768, 3, dev, g, want_holdout=True)]

    best_score = torch.full((e,), -1.0, device=dev)
    best = {k: v.detach().clone() for k, v in p.items()}

    log = []
    t0 = time.time()
    for step in range(args.steps):
        n = places[step % len(places)]
        tok, tgt = data.mixed_batch(args.bs, n, dev, g)
        loss, _, _ = ens.loss_and_acc(p, tok, tgt, tau=args.tau,
                                      margin_w=args.margin_w,
                                      margin_target=args.margin_target)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        clip_and_step(params, opt, sched, args.clip)

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            acc, marg = evaluate(p, ev)
            # exact match first, worst-case margin as a tie-break so members that are
            # already perfect keep pushing the decision boundary away from the data
            score = acc + 0.01 * marg.clamp(0.0, 2.0)
            upd = score > best_score
            best_score = torch.where(upd, score, best_score)
            with torch.no_grad():
                for k in best:
                    best[k] = torch.where(upd.view(-1, *([1] * (best[k].dim() - 1))),
                                          p[k].detach(), best[k])
            rec = dict(step=step + 1, n=n, loss=round(float(loss.mean()), 4),
                       best_acc=round(acc.max().item(), 5),
                       ever=round(best_score.max().item(), 5),
                       n_999=int((acc >= 0.999).sum()),
                       ever_1=int((best_score >= 1.0).sum()),
                       secs=round(time.time() - t0, 1))
            log.append(rec)
            print(json.dumps(rec), flush=True)

    acc, marg = evaluate(best, ev)
    order = torch.argsort(-(acc + 0.01 * marg.clamp(0, 2)))
    keep = order[:args.keep]
    out = os.path.join(WORK, "ckpt", f"{args.tag}.pt")
    torch.save({"params": {k: v[keep].detach().cpu() for k, v in best.items()},
                "acc": acc[keep].cpu(), "margin": marg[keep].cpu(),
                "args": vars(args)}, out)
    with open(os.path.join(WORK, "logs", f"{args.tag}.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")
    print(f"saved {out}: top acc {acc[keep][0]:.6f} margin {marg[keep][0]:.4f}; "
          f"{int((acc >= 0.9999).sum())}/{e} members exact on the held-out yardstick")


if __name__ == "__main__":
    main()
