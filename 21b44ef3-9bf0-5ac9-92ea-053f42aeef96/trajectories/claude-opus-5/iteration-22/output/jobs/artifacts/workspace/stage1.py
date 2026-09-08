"""Stage 1 -- the one-place lottery.

With a single place the task already pins almost everything the model needs:
code[a] + code[b] has to land on prototype code[a+b] whenever a+b < 10, which
forces the code to be linear in the digit; the fold has to move a carrying sum
back by one decade; and the pad position that reads the carry out has to see
one code step.  What one place cannot teach is the transparent class -- a
place with a+b == 9 only matters when there is an earlier place to propagate
from -- so stage 2 supplies that.

Members are trained in parallel on a leading ensemble axis; Adam is
elementwise so they are independent random restarts sharing one batch.
"""

import argparse
import time
import torch

import lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", type=int, default=512)
    ap.add_argument("--redraw", type=int, default=150)
    ap.add_argument("--out", type=str, default="ckpt/stage1.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    p = lab.init_params(args.E, seed=args.seed)
    for t in p.values():
        t.requires_grad_(True)

    # every one-place problem, as one full batch
    da = lab.PAIRS[:, 0:1].contiguous()
    db = lab.PAIRS[:, 1:2].contiguous()
    tgt = lab.answer(da, db)

    opt = torch.optim.AdamW(list(p.values()), lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.15)

    gen = torch.Generator(device=lab.DEV).manual_seed(args.seed + 991)

    def redraw_knees(mask):
        """A saturated clamp has no gradient at its thresholds, so the knees
        are the one coordinate gradient descent cannot move.  Restart them for
        members that have not solved the batch, drawing inside the range of
        token values that member's own code currently produces."""
        with torch.no_grad():
            code = lab.full_code(p)
            lo = 2.0 * code.min(dim=1).values
            hi = 2.0 * code.max(dim=1).values
            r = torch.rand(args.E, 2, device=lab.DEV, generator=gen)
            new = lo[:, None] + r * (hi - lo)[:, None]
            p["knee"].data = torch.where(mask[:, None], new, p["knee"].data)
            for st in opt.state.get(p["knee"], {}).values():
                if torch.is_tensor(st) and st.shape == p["knee"].shape:
                    st.mul_((~mask[:, None]).to(st.dtype))

    best_score = torch.full((args.E,), -1e9, device=lab.DEV)
    best = {k: v.detach().clone() for k, v in p.items()}
    t0 = time.time()
    for step in range(args.steps):
        if args.redraw and step % args.redraw == 0 and step < 0.7 * args.steps:
            with torch.no_grad():
                _, h = lab.loss_and_exact(p, da, db, tgt)
            redraw_knees(h < 100)
        opt.zero_grad(set_to_none=True)
        nll, hits = lab.loss_and_exact(p, da, db, tgt)
        nll.sum().backward()
        lab.per_member_clip(p, 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            score = hits.float() - 0.01 * nll.detach()
            better = score > best_score
            best_score = torch.where(better, score, best_score)
            for k in p:
                m = better.reshape(-1, *([1] * (p[k].dim() - 1)))
                best[k] = torch.where(m, p[k].detach(), best[k])
        if step % 500 == 0 or step == args.steps - 1:
            solved = int((hits == 100).sum())
            print(f"step {step:5d}  loss {nll.mean():.4f}  "
                  f"best_hits {int(hits.max())}/100  solved {solved}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    with torch.no_grad():
        nll, hits = lab.loss_and_exact(best, da, db, tgt)
        code = lab.full_code(best)
        step_sizes = code[:, 1:] - code[:, :-1]
        ramp = (step_sizes > 0).all(1) & (
            step_sizes.max(1).values < 1.5 * step_sizes.min(1).values)
        good = (hits == 100) & ramp
        print(f"exact members: {int((hits == 100).sum())}   "
              f"exact and ascending-linear code: {int(good.sum())}")

        rank = torch.where(good, -nll, torch.full_like(nll, 1e9))
        order = torch.argsort(rank)[: args.keep]
        order = order[good[order]]
        if len(order) == 0:
            raise SystemExit("no usable parents; widen the lottery")
        keep = lab.take(best, order)
        st = step_sizes[order[0]].mean()
        print(f"keeping {len(order)} parents; best member: code step {float(st):.4f}, "
              f"fold/step {float(best['fold'][order[0]] / st):.4f}, "
              f"carry_w/step {float(best['carry_w'][order[0]] / st):.4f}, "
              f"knee/step {[round(float(x / st), 3) for x in best['knee'][order[0]]]}")

    import os
    os.makedirs("ckpt", exist_ok=True)
    torch.save({k: v.cpu() for k, v in keep.items()}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
