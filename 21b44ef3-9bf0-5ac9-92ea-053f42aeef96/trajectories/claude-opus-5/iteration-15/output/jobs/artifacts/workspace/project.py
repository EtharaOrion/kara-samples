"""Collapse a two-channel parent onto one channel, then retrain.

The read-out compares the residual against the ten codes by squared distance,
so any component of the residual that every code shares cancels out of the
comparison.  When the ten learned codes happen to lie close to a line, almost
all of the block's work already happens along that line, and dropping the other
channel is a small perturbation -- but it *is* a perturbation, so the projected
member is treated as an initialisation and retrained, with noisy copies spread
over an ensemble axis so the retraining is itself a lottery.
"""
import argparse, os, time
import torch

import data, ens, reduce as red


def project(p2):
    """p2: a float64 C=2 member.  Returns a C=1 member and the fit residual."""
    code = p2["code"]                                     # [10,2]
    m = code.mean(0, keepdim=True)
    u_, s_, v_ = torch.linalg.svd(code - m, full_matrices=False)
    u = v_[0]                                             # principal direction
    w = v_[1]                                             # the discarded one
    par = code @ u                                        # [10]
    perp = code @ w                                       # [10]
    e0 = float(perp.mean())
    spread = float((perp - e0).abs().max())

    p1 = {k: v.clone() for k, v in p2.items()}
    p1["code"] = par[:, None]
    # the bank saw x = code[a] + code[b]; its perpendicular part is ~2*e0
    p1["bank_w"] = (p2["bank_w"] @ u)[:, None]
    p1["knee"] = p2["knee"] + (p2["bank_w"] @ w) * (2.0 * e0)
    p1["carry_w"] = (p2["carry_w"] @ u).reshape(1)
    p1["fold_w"] = (p2["fold_w"] @ u).reshape(1)
    p1["rb"] = (p2["rb"] @ u).reshape(1)
    return p1, spread, float(s_[1] / s_[0])


def spread_ensemble(p1, E, sigma, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    out = {}
    for k, v in p1.items():
        v = v.to(device=device, dtype=torch.float32)
        stack = v[None].repeat(E, *([1] * v.dim())).contiguous()
        if E > 1:
            noise = torch.randn(stack.shape, generator=g, device=device)
            scale = stack.abs().mean().clamp(min=1e-3) * sigma
            stack[1:] += noise[1:] * scale
        stack.requires_grad_(True)
        out[k] = stack
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--member", type=int, default=-1, help="-1 = scan the top")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--places", default="8,5,11,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    pr = {k: v.to(dev) for k, v in ck["params"].items()}
    acc, order = ck["acc"].to(dev), ck["order"].to(dev)
    cand = [int(order[i]) for i in range(len(order)) if float(acc[order[i]]) > 0.999]
    if args.member >= 0:
        cand = [args.member]
    cand = cand[:args.top]
    print(f"{len(cand)} parent members above 0.999: {cand}")

    places = [int(s) for s in args.places.split(",")]
    eg = torch.Generator(device=dev).manual_seed(777)
    evalset = [data.heldout(8192, n, eg, dev) for n in places]

    seeds = []
    for m in cand:
        p2 = ens.single(pr, m)
        p1, spread, ratio = project(p2)
        print(f"  member {m:5d}: code off-line spread {spread:.4f}, "
              f"singular ratio {ratio:.4f}")
        seeds.append((m, p1))

    per = max(1, args.E // len(seeds))
    stacked = {}
    tags = []
    for m, p1 in seeds:
        e = spread_ensemble(p1, per, args.sigma, dev, args.seed + m)
        tags += [m] * per
        for k, v in e.items():
            stacked.setdefault(k, []).append(v.detach())
    pr1 = {k: torch.cat(v, 0).requires_grad_(True) for k, v in stacked.items()}
    E = pr1["code"].shape[0]
    print(f"retraining {E} projected copies")

    opt = torch.optim.AdamW(list(pr1.values()), lr=args.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps,
                                                pct_start=0.1)
    g = torch.Generator(device=dev).manual_seed(args.seed + 4242)
    best_acc = torch.zeros(E, device=dev)
    best = {k: v.detach().clone() for k, v in pr1.items()}
    t0 = time.time()
    for step in range(args.steps):
        n = places[step % len(places)]
        ta, tb, tg, keep = data.sample(args.batch, n, g, dev)
        loss, _ = ens.loss_and_acc(pr1, ta, tb, tg, keep)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        with torch.no_grad():
            sq = torch.zeros(E, device=dev)
            for v in pr1.values():
                sq += (v.grad ** 2).flatten(1).sum(1)
            sc = (1.0 / sq.sqrt().clamp(min=1e-12)).clamp(max=1.0)
            for v in pr1.values():
                v.grad *= sc.view(-1, *([1] * (v.dim() - 1)))
        opt.step()
        sched.step()
        if (step + 1) % 250 == 0 or step + 1 == args.steps:
            with torch.no_grad():
                a = torch.stack([ens.exact_acc(pr1, *e)[0] for e in evalset]).min(0).values
                imp = a > best_acc
                best_acc = torch.where(imp, a, best_acc)
                for k, v in pr1.items():
                    best[k] = torch.where(imp.view(-1, *([1] * (v.dim() - 1))),
                                          v.detach(), best[k])
            print(f"step {step+1:6d} loss {loss.mean().item():.4f} "
                  f"best {best_acc.max().item():.6f} "
                  f"n>=0.999 {(best_acc>=0.999).sum().item():4d} "
                  f"{time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(dict(cfg=dict(C=1, U=ck["cfg"]["U"]),
                    params={k: v.cpu() for k, v in best.items()},
                    acc=best_acc.cpu(),
                    order=torch.argsort(best_acc, descending=True).cpu(),
                    tags=tags, args=vars(args)), args.out)
    print(f"saved {args.out}; {(best_acc>=0.999).sum().item()} members >= 0.999")


if __name__ == "__main__":
    main()
