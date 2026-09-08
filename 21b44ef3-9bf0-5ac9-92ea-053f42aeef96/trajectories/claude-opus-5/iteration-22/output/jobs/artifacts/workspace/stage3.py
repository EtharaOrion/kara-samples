"""Stage 3 -- gauge fix, constant substitution, and margin polish.

A scalar residual stream has exactly two coordinate freedoms: where its origin
sits and what its unit is.  Stage 1 spent the first by holding code[0] = 0;
here the second is spent by rescaling everything that lives in code units so
that code[1] = 1.  That is a change of coordinates, not a fitted value, and it
leaves the computed function untouched (checked below to float64).

The three sharpness values -- gate slope, key contrast, recency -- are then
replaced by the constants the graded file carries.  They are not gauge, they
are don't-cares: any slope that saturates the bank, any key contrast that
outruns the recency spread over the sequence, and any recency strong enough to
order candidates by distance give the same answers.  Substitution is therefore
checked rather than assumed, and members it changes are dropped.

What is left is the 12 values the model actually learned, which are then
polished on the read-out margin.
"""

import argparse
import os
import time
import torch

import lab


def gauge_fix(p):
    """Rescale the residual axis so that code[1] = 1."""
    lam = 1.0 / p["code_free"][:, 0].clone()
    out = {k: v.clone() for k, v in p.items()}
    for k in ("code_free", "knee"):
        out[k] = p[k] * lam[:, None]
    for k in ("carry_w", "fold"):
        out[k] = p[k] * lam
    out["log_slope"] = p["log_slope"] - lam.log()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="ckpt/stage2.pt")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.92)
    ap.add_argument("--satw", type=float, default=0.5)
    ap.add_argument("--satm", type=float, default=2.5)
    ap.add_argument("--widths", type=str, default="1,2,3,5,8")
    ap.add_argument("--out", type=str, default="ckpt/stage3.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    src = {k: v.to(lab.DEV) for k, v in torch.load(args.src).items()}
    g = gauge_fix(src)

    # the gauge move must not change what the model computes
    checks = [lab.sample(n, 512, split="heldout") for n in (2, 3, 5, 8)]
    with torch.no_grad():
        a0, _ = lab.exact_counts(src, checks)
        a1, tot = lab.exact_counts(g, checks)
        moved = 0.0
        for da, db, tgt in checks:
            r0 = lab.residual(src, da, db)[0]
            r1 = lab.residual(g, da, db)[0]
            lam = (1.0 / src["code_free"][:, 0])[:, None, None]
            moved = max(moved, float((r1 - r0 * lam).abs().max()))
    print(f"gauge fix: exact {int((a0 == tot).sum())} -> {int((a1 == tot).sum())} "
          f"of {g['fold'].shape[0]}, max residual drift {moved:.2e}")

    # substitute the sharpness constants and keep only members it leaves intact
    with torch.no_grad():
        a2, _ = lab.exact_counts(g, checks, ship_consts=True)
        sat = lab.sat_penalty(g, margin=1.0, ship_consts=True)
        survived = (a2 == tot) & (sat == 0)
    print(f"constant substitution: {int(survived.sum())} of {g['fold'].shape[0]} "
          f"members still exact with a saturated bank")
    if int(survived.sum()) == 0:
        raise SystemExit("no members survive substitution")
    g = lab.take(g, torch.nonzero(survived).squeeze(1))

    p = lab.repeat(g, args.reps)
    E = p["code_free"].shape[0]
    for k in ("log_slope", "log_key", "log_rec", "log_ls"):
        p[k] = torch.zeros_like(p[k])                 # unused once ship_consts=True
    pin = p["code_free"][:, 0].clone()
    print(f"polishing {E} members (code[1] pinned at "
          f"{float(pin.min()):.6f}..{float(pin.max()):.6f})")

    trainable = ["code_free", "carry_w", "knee", "fold"]
    for k in trainable:
        p[k].requires_grad_(True)
    opt = torch.optim.AdamW([p[k] for k in trainable], lr=args.lr,
                            betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    widths = [int(w) for w in args.widths.split(",")]
    probe = [lab.sample(n, 128, split="heldout") for n in (3, 8)]
    probe += [lab.uniform_sample(8, 128, split="heldout")]
    best_score = torch.full((E,), -1e9, device=lab.DEV)
    best = {k: v.detach().clone() for k, v in p.items()}

    t0 = time.time()
    for step in range(args.steps):
        n = widths[step % len(widths)]
        da, db, tgt = lab.sample(n, args.batch, split="train")
        opt.zero_grad(set_to_none=True)
        m = lab.margins(p, da, db, tgt, ship_consts=True)
        sat = lab.sat_penalty(p, margin=args.satm, ship_consts=True)
        loss = torch.relu(args.target - m).mean(dim=(1, 2)) + args.satw * sat
        loss.sum().backward()
        lab.per_member_clip(p, 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            p["code_free"][:, 0] = pin                # code[1] = 1 is the gauge

        if step % 100 == 0 or step == args.steps - 1:
            with torch.no_grad():
                wm = None
                for da, db, tgt in probe:
                    mm = lab.margins(p, da, db, tgt, ship_consts=True).amin(dim=(1, 2))
                    wm = mm if wm is None else torch.minimum(wm, mm)
                satv = lab.sat_penalty(p, margin=args.satm, ship_consts=True)
                score = wm - satv
                better = score > best_score
                best_score = torch.where(better, score, best_score)
                for k in p:
                    msk = better.reshape(-1, *([1] * (p[k].dim() - 1)))
                    best[k] = torch.where(msk, p[k].detach(), best[k])
            if step % 500 == 0 or step == args.steps - 1:
                print(f"step {step:5d}  loss {float(loss.mean()):.4f}  "
                      f"worst margin {float(wm.max()):.4f}  "
                      f"sat0 {int((satv == 0).sum())}  [{time.time() - t0:.0f}s]",
                      flush=True)

    with torch.no_grad():
        big = [lab.sample(n, 2048, split="heldout") for n in (2, 3, 5, 8)]
        big += [lab.uniform_sample(8, 4096, split="heldout")]
        hits, tot = lab.exact_counts(best, big, ship_consts=True)
        wm = None
        for da, db, tgt in big:
            mm = lab.margins(best, da, db, tgt, ship_consts=True).amin(dim=(1, 2))
            wm = mm if wm is None else torch.minimum(wm, mm)
        satv = lab.sat_penalty(best, margin=1.0, ship_consts=True)
        ok = (hits == tot) & (satv == 0)
        print(f"exact on {tot} held-out: {int((hits == tot).sum())}, "
              f"of which saturated: {int(ok.sum())}, best worst-margin "
              f"{float(wm[ok].max()) if int(ok.sum()) else float('nan'):.4f}")
        if int(ok.sum()) == 0:
            raise SystemExit("nothing shippable")
        rank = torch.where(ok, wm, torch.full_like(wm, -1e9))
        order = torch.argsort(rank, descending=True)[: 256]
        order = order[ok[order]]
        keep = lab.take(best, order)
        c = lab.full_code(keep)
        print("best member: code =",
              [round(float(x), 5) for x in c[0]],
              "\n  carry_w", round(float(keep["carry_w"][0]), 5),
              " knee", [round(float(x), 5) for x in keep["knee"][0]],
              " fold", round(float(keep["fold"][0]), 5))

    os.makedirs("ckpt", exist_ok=True)
    torch.save({k: v.cpu() for k, v in keep.items()}, args.out)
    print(f"wrote {args.out} with {keep['fold'].shape[0]} members")


if __name__ == "__main__":
    main()
