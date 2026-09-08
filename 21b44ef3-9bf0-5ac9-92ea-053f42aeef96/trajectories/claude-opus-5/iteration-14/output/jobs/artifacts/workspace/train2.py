"""Cold-train E independent members at once, with two fixes over train.py.

  * leaky clamp.  The bank's knees only get gradient while some inputs land in the
    linear part of the clamp.  Members that saturate early freeze with the wrong
    knees.  A small leak outside [0,1] keeps that gradient alive; it is annealed to
    exactly zero over the first `anneal` fraction of training, so the last part of
    the run optimises the real function.

  * an evaluation that means something.  train.py selected on held-out exact match
    at the training width; at n=1 the held-out split is ~5 digit pairs, so "perfect"
    members were nothing of the sort.  Selection here is always at n=8.

Reported at each eval: held-out exact match at n=8, per-position digit accuracy (a smooth
progress signal), and how linear the learned code is -- code[d] = d*sigma is forced
by the task, so R^2 -> 1 is the sign a member is on the right track.
"""
import argparse, os, time
import torch

import arch, data


def ce_loss(dlog, tgt):
    logp = torch.log_softmax(dlog, dim=-1)
    t = tgt[None, :, :, None].expand(dlog.shape[0], -1, -1, 1)
    nll = -logp.gather(-1, t).squeeze(-1)
    per = nll.mean(dim=(1, 2))
    return per.sum(), per.detach()


@torch.no_grad()
def evaluate(p, cfg, n, device, gen, N=4096, chunk=1024):
    """-> (exact-match, per-digit accuracy), both per member, on held-out pairs."""
    E = p["code"].shape[0]
    good = torch.zeros(E, device=device)
    dig = torch.zeros(E, device=device)
    tot = 0
    while tot < N:
        b = min(chunk, N - tot)
        ta, tb, tgt, _, _ = data.batch(b, n, device, gen, heldout=True)
        pred = arch.forward(p, ta, tb, cfg).argmax(-1)          # (E,b,P)
        hit = pred[:, :, 1:] == tgt[None, :, 1:]
        good += hit.all(-1).float().sum(1)
        dig += hit.float().mean(-1).sum(1)
        tot += b
    return good / tot, dig / tot


@torch.no_grad()
def code_r2(p, cfg):
    """How close the learned code is to the forced ramp code[d] = d*sigma."""
    c = arch.full_code(p, cfg)[:, :, 0].double()                # (E,10)
    d = torch.arange(10, device=c.device, dtype=torch.float64)
    sig = (c * d).sum(-1) / (d * d).sum()
    resid = ((c - sig[:, None] * d) ** 2).sum(-1)
    total = ((c - c.mean(-1, keepdim=True)) ** 2).sum(-1) + 1e-30
    return (1.0 - resid / total).float()


def per_member_clip(p, maxnorm):
    sq = None
    for v in p.values():
        if v.grad is None:
            continue
        g = v.grad.reshape(v.shape[0], -1)
        s = (g * g).sum(1)
        sq = s if sq is None else sq + s
    if sq is None:
        return
    scale = (maxnorm / (sq.sqrt() + 1e-12)).clamp(max=1.0)
    for v in p.values():
        if v.grad is None:
            continue
        v.grad.mul_(scale.reshape((v.shape[0],) + (1,) * (v.dim() - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--pct_start", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--C", type=int, default=1)
    ap.add_argument("--U", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--places", type=str, default="2,3,4,6,8")
    ap.add_argument("--leak", type=float, default=0.05)
    ap.add_argument("--anneal", type=float, default=0.6, help="leak hits 0 at this fraction")
    ap.add_argument("--lam_init", type=float, default=-1.5)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_n", type=int, default=8)
    ap.add_argument("--eval_N", type=int, default=2048)
    ap.add_argument("--keep", type=int, default=64)
    ap.add_argument("--out", type=str, default="ckpt/p2.pt")
    ap.add_argument("--init_from", type=str, default="")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    cfg = arch.default_cfg(C=args.C, U=args.U, lam_init=args.lam_init)
    places = [int(s) for s in args.places.split(",")]

    p = arch.init_params(cfg, args.E, device, seed=args.seed)
    if args.init_from:
        st = torch.load(args.init_from, map_location=device)
        cfg = st["cfg"]
        src = st["p"]
        rep = args.E // src["code"].shape[0]
        p = {k: v.to(device).repeat_interleave(rep, 0).contiguous() for k, v in src.items()}
        args.E = p["code"].shape[0]
    for v in p.values():
        v.requires_grad_(True)
    print(f"members={args.E}  free params/member={arch.count_free(p)}  cfg={cfg}  "
          f"places={places}  leak={args.leak}->0 at {args.anneal:.0%}", flush=True)

    opt = torch.optim.AdamW(list(p.values()), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=args.pct_start)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    egen = torch.Generator(device=device).manual_seed(12345)

    best = torch.zeros(args.E, device=device)
    best_p = {k: v.detach().clone() for k, v in p.items()}
    t0 = time.time()
    for step in range(args.steps):
        leak = args.leak * max(0.0, 1.0 - step / (args.anneal * args.steps))
        n = places[step % len(places)]
        ta, tb, tgt, _, _ = data.batch(args.batch, n, device, gen)
        loss, per = ce_loss(arch.forward(p, ta, tb, cfg, leak=leak), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        per_member_clip(p, args.clip)
        opt.step()
        sched.step()

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            acc, dig = evaluate(p, cfg, args.eval_n, device, egen, N=args.eval_N)
            r2 = code_r2(p, cfg)
            upd = acc > best
            if upd.any():
                for k in p:
                    sel = upd.reshape((-1,) + (1,) * (p[k].dim() - 1))
                    best_p[k] = torch.where(sel, p[k].detach(), best_p[k])
                best = torch.maximum(best, acc)
            print(f"step {step+1:6d} leak {leak:.4f} loss {per.mean():.4f} | "
                  f"digit {dig.max():.4f} exact {acc.max():.4f} best {best.max():.5f} "
                  f">=.99 {int((best>=0.99).sum()):4d} | r2 max {r2.max():.4f} "
                  f"#r2>.999 {int((r2>0.999).sum()):5d} | {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keep = torch.argsort(best, descending=True)[:args.keep]
    torch.save(dict(p={k: v[keep].cpu() for k, v in best_p.items()}, cfg=cfg,
                    acc=best[keep].cpu(), args=vars(args)), args.out)
    print(f"saved {args.out}: kept {len(keep)}, top acc {best[keep][:8].tolist()}")


if __name__ == "__main__":
    main()
