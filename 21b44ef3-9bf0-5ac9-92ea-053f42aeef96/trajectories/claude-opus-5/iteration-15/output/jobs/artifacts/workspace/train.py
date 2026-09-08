"""Cold-train a wide ensemble of parent blocks and keep the members that work.

Usage:  python train.py --E 512 --steps 6000 --out ckpt/parent.pt
"""
import argparse, json, os, time
import torch

import data, ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=512)
    ap.add_argument("--C", type=int, default=1)
    ap.add_argument("--U", type=int, default=2)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--places", type=str, default="8,5,11,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--log", type=str, default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(C=args.C, U=args.U)
    places = [int(s) for s in args.places.split(",")]
    g = torch.Generator(device=dev).manual_seed(args.seed + 12345)
    pr = ens.init(cfg, args.E, dev, args.seed)

    opt = torch.optim.AdamW(list(pr.values()), lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.15)

    # per-member best snapshot, scored on held-out exact match
    best_acc = torch.zeros(args.E, device=dev)
    best = {k: v.detach().clone() for k, v in pr.items()}
    eg = torch.Generator(device=dev).manual_seed(999)
    evalset = [data.heldout(8192, n, eg, dev) for n in places]

    lines = []
    t0 = time.time()
    for step in range(args.steps):
        n = places[step % len(places)]
        ta, tb, tg, keep = data.sample(args.batch, n, g, dev)
        loss, _ = ens.loss_and_acc(pr, ta, tb, tg, keep)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        # clip each member independently so members stay independent
        with torch.no_grad():
            sq = torch.zeros(args.E, device=dev)
            for v in pr.values():
                sq += (v.grad ** 2).flatten(1).sum(1)
            scale = (args.clip / sq.sqrt().clamp(min=1e-12)).clamp(max=1.0)
            for v in pr.values():
                v.grad *= scale.view(-1, *([1] * (v.dim() - 1)))
        opt.step()
        sched.step()

        if (step + 1) % 250 == 0 or step + 1 == args.steps:
            with torch.no_grad():
                accs = []
                for (ta_, tb_, tg_, k_) in evalset:
                    a_, _ = ens.exact_acc(pr, ta_, tb_, tg_, k_)
                    accs.append(a_)
                acc = torch.stack(accs, 0).min(0).values
                imp = acc > best_acc
                if imp.any():
                    best_acc = torch.where(imp, acc, best_acc)
                    for k, v in pr.items():
                        m = imp.view(-1, *([1] * (v.dim() - 1)))
                        best[k] = torch.where(m, v.detach(), best[k])
            msg = (f"step {step+1:6d}  loss {loss.mean().item():.4f}  "
                   f"best {best_acc.max().item():.5f}  "
                   f"n>=0.999 {(best_acc>=0.999).sum().item():4d}  "
                   f"n==1.0 {(best_acc>=1.0).sum().item():4d}  "
                   f"{time.time()-t0:.0f}s")
            print(msg, flush=True)
            lines.append(msg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    order = torch.argsort(best_acc, descending=True)
    torch.save(dict(cfg=cfg, params={k: v.cpu() for k, v in best.items()},
                    acc=best_acc.cpu(), order=order.cpu(), args=vars(args)),
               args.out)
    print(f"saved {args.out}  top10 acc {best_acc[order[:10]].tolist()}")
    if args.log:
        with open(args.log, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
