"""Two-stage curriculum trainer.

Cold training collapses into a wrapped code -- something like
[0,-.68,-1.23,-1.67,-2.13, 2.72,2.29,1.81,1.35,.88] -- because the mod-10 read-out
scores an embedding that has the wrap built in.  A wrapped code is non-monotone in
a+b, so no clamp threshold can separate "carries" from "does not carry", and the
whole carry mechanism is then unreachable.  Every cold run in this workspace dies
that way, at ~0.5 per-digit accuracy.

Stage A fixes it by training on carry-free data only (every place has a+b <= 9).
There the answer digit *is* a+b, so the loss directly forces
code[a] + code[b] = code[a+b] on that sub-domain, which admits only the ramp
code[d] = d*sigma.  Nothing tells the model what sigma is or what the bank should
do; the restricted data does the work.

Stage B keeps the stage-A codes, re-initialises everything else at random (many
restarts per surviving code), and trains on the real distribution, where the carry
mechanism is the only thing left to learn.  The code keeps training at a reduced
learning rate so it can be refined without falling back into the wrap.
"""
import argparse, os, time
import torch

import arch, data
from train2 import ce_loss, evaluate, code_r2, per_member_clip

NOCARRY = dict(mix=(1.0, 0.0, 0.0), regimes=("nocarry", "uniform", "chain"),
               full_width=False)


def stage_a(args, device):
    cfg = arch.default_cfg(C=args.C, U=args.U, lam_init=args.lam_init)
    E = args.EA
    p = arch.init_params(cfg, E, device, seed=args.seed)
    for v in p.values():
        v.requires_grad_(True)
    opt = torch.optim.AdamW(list(p.values()), lr=args.lrA, betas=(0.9, 0.99), weight_decay=0.0)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lrA, total_steps=args.stepsA,
                                              pct_start=0.15)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    places = [2, 3]
    t0 = time.time()
    for step in range(args.stepsA):
        leak = args.leak * max(0.0, 1.0 - step / (0.6 * args.stepsA))
        n = places[step % len(places)]
        ta, tb, tgt, _, _ = data.batch(args.batch, n, device, gen, **NOCARRY)
        loss, per = ce_loss(arch.forward(p, ta, tb, cfg, leak=leak), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        per_member_clip(p, args.clip)
        opt.step()
        sch.step()
        if (step + 1) % 250 == 0:
            with torch.no_grad():
                ta, tb, tgt, _, _ = data.batch(1024, 3, device, gen, **NOCARRY)
                pred = arch.forward(p, ta, tb, cfg).argmax(-1)
                acc = (pred[:, :, 1:] == tgt[None, :, 1:]).all(-1).float().mean(1)
                r2 = code_r2(p, cfg)
            print(f"  A {step+1:5d} loss {per.mean():.4f} nocarry-exact max {acc.max():.4f} "
                  f"#==1 {int((acc>=1.0).sum()):5d} | r2 max {r2.max():.5f} "
                  f"#>.9999 {int((r2>0.9999).sum()):5d} | {time.time()-t0:.0f}s", flush=True)
    with torch.no_grad():
        ta, tb, tgt, _, _ = data.batch(4096, 3, device, gen, **NOCARRY)
        pred = arch.forward(p, ta, tb, cfg).argmax(-1)
        acc = (pred[:, :, 1:] == tgt[None, :, 1:]).all(-1).float().mean(1)
        r2 = code_r2(p, cfg)
    # a code is useful only if it is both exact on carry-free data and a true ramp
    score = acc + r2.clamp(min=0)
    keep = torch.argsort(score, descending=True)[:args.keepA]
    print(f"stage A: kept {len(keep)}; acc {acc[keep][:5].tolist()} r2 {r2[keep][:5].tolist()}")
    return {k: v.detach()[keep] for k, v in p.items()}, cfg, acc[keep], r2[keep]


def stage_b_init(codes, cfg, R, seed, device):
    """Replicate each stage-A code R times, random fresh init for everything else.

    Scales are set relative to the code step sigma the member actually learned -- the
    residual lives on [0, 18 sigma], so knees are drawn across that range and the
    carry write is drawn over a wide band of octaves around sigma.  Where the knees
    and the write *should* sit is not supplied.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    E0 = codes.shape[0]
    E = E0 * R
    C, U = cfg["C"], cfg["U"]
    code = codes.repeat_interleave(R, 0).contiguous()               # (E,9,C)
    d = torch.arange(1, 10, device=device, dtype=code.dtype)
    sig = (code[:, :, 0] * d).sum(-1) / (d * d).sum()               # (E,) learned step
    sgn = torch.where(sig >= 0, 1.0, -1.0)
    asig = sig.abs().clamp(min=1e-4)

    def U01(*sh):
        return torch.rand(*sh, generator=g, device=device)

    knee = asig[:, None] * (-2.0 + 22.0 * U01(E, U))                # (E,U) across [0,18 sigma]
    width = asig[:, None] * torch.pow(2.0, -1.0 + 3.0 * U01(E, U))  # 0.5..4 sigma
    slope = sgn[:, None] * torch.where(U01(E, U) < 0.5, -1.0, 1.0) / width
    p = dict(
        code=code.clone(),
        Bw=slope[:, None, :].expand(E, C, U).contiguous(),
        bb=(-slope * knee).contiguous(),
        kw=torch.randn(E, U, generator=g, device=device) * 2.0,
        vw=torch.randn(E, U, generator=g, device=device),
        q=torch.ones(E, device=device),
        lam=torch.full((E,), cfg["lam_init"], device=device)
            + torch.randn(E, generator=g, device=device) * 0.3,
        w1=(sig[:, None] * torch.pow(2.0, -2.0 + 6.0 * U01(E, C))
            * torch.where(U01(E, C) < 0.5, -1.0, 1.0)).contiguous(),
        w2=(sig[:, None] * torch.pow(2.0, -2.0 + 6.0 * U01(E, C))
            * torch.where(U01(E, C) < 0.5, -1.0, 1.0)).contiguous(),
        vb=torch.randn(E, generator=g, device=device) * 0.2,
        rb=torch.randn(E, C, generator=g, device=device) * 0.2 * asig[:, None],
        ls=(1.0 / (asig * asig)).contiguous(),                      # logits start O(1)
    )
    return {k: v.contiguous() for k, v in p.items()}


def stage_b(args, p, cfg, device):
    E = p["code"].shape[0]
    for v in p.values():
        v.requires_grad_(True)
    groups = [{"params": [v for k, v in p.items() if k != "code"], "lr": args.lrB},
              {"params": [p["code"]], "lr": args.lrB * args.code_mult}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=0.0)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lrB, args.lrB * args.code_mult],
        total_steps=args.stepsB, pct_start=0.1)
    gen = torch.Generator(device=device).manual_seed(args.seed + 2)
    egen = torch.Generator(device=device).manual_seed(12345)
    places = [int(s) for s in args.places.split(",")]

    best = torch.zeros(E, device=device)
    best_p = {k: v.detach().clone() for k, v in p.items()}
    t0 = time.time()
    for step in range(args.stepsB):
        leak = args.leak * max(0.0, 1.0 - step / (0.6 * args.stepsB))
        n = places[step % len(places)]
        ta, tb, tgt, _, _ = data.batch(args.batch, n, device, gen)
        loss, per = ce_loss(arch.forward(p, ta, tb, cfg, leak=leak), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        per_member_clip(p, args.clip)
        opt.step()
        sch.step()
        if (step + 1) % args.eval_every == 0 or step == args.stepsB - 1:
            acc, dig = evaluate(p, cfg, 8, device, egen, N=args.eval_N)
            upd = acc > best
            if upd.any():
                for k in p:
                    sel = upd.reshape((-1,) + (1,) * (p[k].dim() - 1))
                    best_p[k] = torch.where(sel, p[k].detach(), best_p[k])
                best = torch.maximum(best, acc)
            r2 = code_r2(p, cfg)
            print(f"  B {step+1:6d} leak {leak:.4f} loss {per.mean():.4f} | digit {dig.max():.4f} "
                  f"exact {acc.max():.4f} best {best.max():.5f} >=.99 {int((best>=0.99).sum()):4d} "
                  f"| r2>.999 {int((r2>0.999).sum()):5d} | {time.time()-t0:.0f}s", flush=True)
    return best_p, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--EA", type=int, default=4096)
    ap.add_argument("--keepA", type=int, default=64)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--stepsA", type=int, default=2500)
    ap.add_argument("--stepsB", type=int, default=8000)
    ap.add_argument("--lrA", type=float, default=0.03)
    ap.add_argument("--lrB", type=float, default=0.02)
    ap.add_argument("--code_mult", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--leak", type=float, default=0.05)
    ap.add_argument("--lam_init", type=float, default=-1.5)
    ap.add_argument("--C", type=int, default=1)
    ap.add_argument("--U", type=int, default=2)
    ap.add_argument("--places", type=str, default="2,3,4,6,8")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_N", type=int, default=2048)
    ap.add_argument("--keep", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stageA_ckpt", type=str, default="ckpt/stageA.pt")
    ap.add_argument("--out", type=str, default="ckpt/stageB.pt")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    os.makedirs("ckpt", exist_ok=True)

    if os.path.exists(args.stageA_ckpt):
        st = torch.load(args.stageA_ckpt, map_location=device)
        pa, cfg = st["p"], st["cfg"]
        print(f"stage A loaded from {args.stageA_ckpt}: {pa['code'].shape[0]} codes")
    else:
        pa, cfg, acc, r2 = stage_a(args, device)
        torch.save(dict(p={k: v.cpu() for k, v in pa.items()}, cfg=cfg,
                        acc=acc.cpu(), r2=r2.cpu(), args=vars(args)), args.stageA_ckpt)
        pa = {k: v.to(device) for k, v in pa.items()}

    p = stage_b_init(pa["code"], cfg, args.R, args.seed + 991, device)
    print(f"stage B: {p['code'].shape[0]} members "
          f"({pa['code'].shape[0]} codes x {args.R} restarts)", flush=True)
    best_p, best = stage_b(args, p, cfg, device)
    keep = torch.argsort(best, descending=True)[:args.keep]
    torch.save(dict(p={k: v[keep].cpu() for k, v in best_p.items()}, cfg=cfg,
                    acc=best[keep].cpu(), args=vars(args)), args.out)
    print(f"saved {args.out}: top acc {best[keep][:8].tolist()}")


if __name__ == "__main__":
    main()
