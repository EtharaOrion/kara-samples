"""Cold-train E independent members of the parent architecture simultaneously.

Every member sees the same batch but has its own random initialisation and its own
Adam state (Adam is elementwise, so stacking members on a leading axis trains them
independently).  This is the "seed lottery": at these sizes the outcome is
seed-dominated, so many members are run at once and the best is kept.
"""
import argparse, json, math, os, time
import torch

import arch, data


def ce_loss(dlog, tgt):
    """dlog (E,B,P,10), tgt (B,P) -> scalar (summed over members, mean over B,P)."""
    logp = torch.log_softmax(dlog, dim=-1)
    t = tgt[None, :, :, None].expand(dlog.shape[0], -1, -1, 1)
    nll = -logp.gather(-1, t).squeeze(-1)
    return nll.mean(dim=(1, 2)).sum(), nll.mean(dim=(1, 2)).detach()


@torch.no_grad()
def exact_match(p, cfg, n, device, gen, N=8192, chunk=1024, heldout=True):
    """Per-member fraction of samples whose full answer is exactly right."""
    E = p["code"].shape[0]
    good = torch.zeros(E, device=device)
    tot = 0
    while tot < N:
        b = min(chunk, N - tot)
        ta, tb, tgt, _, _ = data.batch(b, n, device, gen, heldout=heldout)
        dlog = arch.forward(p, ta, tb, cfg)
        pred = dlog.argmax(-1)                      # (E,b,P)
        ok = (pred[:, :, 1:] == tgt[None, :, 1:]).all(-1)
        good += ok.float().sum(1)
        tot += b
    return good / tot


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
    nrm = sq.sqrt()
    scale = (maxnorm / (nrm + 1e-12)).clamp(max=1.0)
    for v in p.values():
        if v.grad is None:
            continue
        shape = (v.shape[0],) + (1,) * (v.dim() - 1)
        v.grad.mul_(scale.reshape(shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--pct_start", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--C", type=int, default=1)
    ap.add_argument("--U", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--places", type=str, default="8,5,11,3")
    ap.add_argument("--warm_places", type=str, default="")
    ap.add_argument("--warm_frac", type=float, default=0.0)
    ap.add_argument("--vb", type=int, default=1)
    ap.add_argument("--rb", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_n", type=int, default=8)
    ap.add_argument("--out", type=str, default="ckpt/parent.pt")
    ap.add_argument("--init_from", type=str, default="")
    ap.add_argument("--keep", type=int, default=64)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    cfg = arch.default_cfg(C=args.C, U=args.U, use_vb=bool(args.vb), use_rb=bool(args.rb))
    places = [int(s) for s in args.places.split(",")]
    warm = [int(s) for s in args.warm_places.split(",")] if args.warm_places else places
    nwarm = int(args.warm_frac * args.steps)

    p = arch.init_params(cfg, args.E, device, seed=args.seed)
    if args.init_from:
        st = torch.load(args.init_from, map_location=device)
        src = st["p"]
        for k in p:
            p[k] = src[k].clone()
        cfg = st["cfg"]
    for v in p.values():
        v.requires_grad_(True)

    nfree = arch.count_free(p)
    print(f"members={args.E}  free params/member={nfree}  cfg={cfg}", flush=True)

    opt = torch.optim.AdamW(list(p.values()), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=args.pct_start)

    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    egen = torch.Generator(device=device).manual_seed(12345)

    best = torch.zeros(args.E, device=device)
    best_p = {k: v.detach().clone() for k, v in p.items()}
    t0 = time.time()
    for step in range(args.steps):
        n = warm[step % len(warm)] if step < nwarm else places[step % len(places)]
        ta, tb, tgt, _, _ = data.batch(args.batch, n, device, gen)
        dlog = arch.forward(p, ta, tb, cfg)
        loss, per = ce_loss(dlog, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        per_member_clip(p, args.clip)
        opt.step()
        sched.step()

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            acc = exact_match(p, cfg, args.eval_n, device, egen, N=4096)
            upd = acc > best
            if upd.any():
                for k in p:
                    sel = upd.reshape((-1,) + (1,) * (p[k].dim() - 1))
                    best_p[k] = torch.where(sel, p[k].detach(), best_p[k])
                best = torch.maximum(best, acc)
            nperf = int((best >= 1.0).sum())
            print(f"step {step+1:6d}  loss {per.mean():.4f}  "
                  f"acc max {acc.max():.4f} best {best.max():.5f}  "
                  f">=0.99: {int((best>=0.99).sum()):4d}  perfect(4k): {nperf:4d}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    order = torch.argsort(best, descending=True)
    keep = order[:args.keep]
    torch.save(dict(p={k: v[keep].cpu() for k, v in best_p.items()},
                    cfg=cfg, acc=best[keep].cpu(), args=vars(args)), args.out)
    print(f"saved {args.out}: kept {len(keep)}, top acc {best[keep][:8].tolist()}")


if __name__ == "__main__":
    main()
