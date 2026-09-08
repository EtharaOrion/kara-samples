"""Stage 2 -- widen to many places.

The one-place parents know the code, the fold and the carry write-back but
have never met a transparent place (a + b == 9), which is the only reason the
block needs attention at all.  Here each parent is replicated, the knees are
restarted from the parent's own token range, and training runs over widths
1..8 on carry-structured data with the held-out split enforced.

A saturation penalty pushes the clamp bank to a hard 0/1 gate, so key and
value end up depending on the carry class of a place and nothing else.
"""

import argparse
import os
import time
import torch

import lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", type=str, default="ckpt/stage1.pt")
    ap.add_argument("--reps", type=int, default=128)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=0.006)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--redraw", type=int, default=200)
    ap.add_argument("--satw", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--keep", type=int, default=1024)
    ap.add_argument("--widths", type=str, default="1,2,3,5,8")
    ap.add_argument("--out", type=str, default="ckpt/stage2.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    parents = {k: v.to(lab.DEV) for k, v in torch.load(args.parents).items()}
    p = lab.repeat(parents, args.reps)
    E = p["code_free"].shape[0]
    widths = [int(w) for w in args.widths.split(",")]
    print(f"{parents['fold'].shape[0]} parents x {args.reps} = {E} members, "
          f"widths {widths}")

    gen = torch.Generator(device=lab.DEV).manual_seed(args.seed + 7717)
    fresh = torch.arange(E, device=lab.DEV) % args.reps != 0   # keep replica 0 as-is

    def redraw_knees(mask):
        with torch.no_grad():
            code = lab.full_code(p)
            lo = 2.0 * code.min(dim=1).values
            hi = 2.0 * code.max(dim=1).values
            r = torch.rand(E, 2, device=lab.DEV, generator=gen)
            new = lo[:, None] + r * (hi - lo)[:, None]
            p["knee"].data = torch.where(mask[:, None], new, p["knee"].data)
            for st in opt.state.get(p["knee"], {}).values():
                if torch.is_tensor(st) and st.shape == p["knee"].shape:
                    st.mul_((~mask[:, None]).to(st.dtype))

    for t in p.values():
        t.requires_grad_(True)
    opt = torch.optim.AdamW(list(p.values()), lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    redraw_knees(fresh)

    # fixed held-out probe used for snapshot selection
    probe = [lab.sample(n, 96, split="heldout") for n in (3, 8)]
    probe += [lab.uniform_sample(8, 96, split="heldout")]
    probe_total = sum(x[0].shape[0] for x in probe)

    best_score = torch.full((E,), -1e9, device=lab.DEV)
    best = {k: v.detach().clone() for k, v in p.items()}
    t0 = time.time()
    for step in range(args.steps):
        if args.redraw and step % args.redraw == 0 and step < 0.6 * args.steps:
            with torch.no_grad():
                hits, _ = lab.exact_counts(p, probe)
            redraw_knees(fresh & (hits < probe_total))

        n = widths[step % len(widths)]
        da, db, tgt = lab.sample(n, args.batch, split="train")
        opt.zero_grad(set_to_none=True)
        nll, _ = lab.loss_and_exact(p, da, db, tgt)
        ramp = min(1.0, max(0.0, (step / args.steps - 0.2) / 0.3))
        sat = lab.sat_penalty(p, margin=args.margin)
        (nll + args.satw * ramp * sat).sum().backward()
        lab.per_member_clip(p, 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0 or step == args.steps - 1:
            with torch.no_grad():
                hits, _ = lab.exact_counts(p, probe)
                satv = lab.sat_penalty(p, margin=args.margin)
                score = hits.float() - 50.0 * satv
                better = score > best_score
                best_score = torch.where(better, score, best_score)
                for k in p:
                    m = better.reshape(-1, *([1] * (p[k].dim() - 1)))
                    best[k] = torch.where(m, p[k].detach(), best[k])
            if step % 500 == 0 or step == args.steps - 1:
                solved = int((hits == probe_total).sum())
                clean = int(((hits == probe_total) & (satv == 0)).sum())
                print(f"step {step:5d}  nll {nll.mean():.4f}  sat {satv.mean():.4f}  "
                      f"probe best {int(hits.max())}/{probe_total}  solved {solved}  "
                      f"solved+saturated {clean}  [{time.time() - t0:.0f}s]", flush=True)

    with torch.no_grad():
        big = [lab.sample(n, 512, split="heldout") for n in (2, 3, 5, 8)]
        big += [lab.uniform_sample(8, 512, split="heldout")]
        hits, tot = lab.exact_counts(best, big)
        satv = lab.sat_penalty(best, margin=args.margin)
        code = lab.full_code(best)
        steps_ = code[:, 1:] - code[:, :-1]
        ok = (hits == tot) & (satv == 0) & (steps_ > 0).all(1) & (code[:, 1] > 0)
        print(f"exact on {tot} held-out: {int((hits == tot).sum())}, "
              f"also saturated and canonically oriented: {int(ok.sum())}")
        if int(ok.sum()) == 0:
            raise SystemExit("no shippable members at stage 2")
        rank = torch.where(ok, -lab.sat_penalty(best, margin=6.0),
                           torch.full_like(satv, -1e9))
        order = torch.argsort(rank, descending=True)[: args.keep]
        order = order[ok[order]]
        keep = lab.take(best, order)
        s0 = steps_[order[0]].mean()
        print(f"keeping {len(order)}; best: code step {float(s0):.4f}, "
              f"fold/step {float(best['fold'][order[0]] / s0):.4f}, "
              f"carry_w/step {float(best['carry_w'][order[0]] / s0):.4f}, "
              f"knee/step {[round(float(x / s0), 3) for x in best['knee'][order[0]]]}")

    os.makedirs("ckpt", exist_ok=True)
    torch.save({k: v.cpu() for k, v in keep.items()}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
