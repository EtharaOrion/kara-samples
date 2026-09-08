"""Cold-train an ensemble of parents ("seed lottery") and keep the best members."""
import argparse
import os
import torch

import lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--U", type=int, default=6)
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ns", type=str, default="8,5,11,3")
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--tag", type=str, default="cold")
    ap.add_argument("--outdir", type=str, default="ck")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = lab.default_cfg(C=args.C, U=args.U)
    ns = tuple(int(x) for x in args.ns.split(","))
    print(f"cfg={cfg} params={lab.n_params(cfg)} E={args.E} steps={args.steps}", flush=True)

    ens = lab.Ens(cfg, args.E, seed=args.seed)
    lab.train(ens, steps=args.steps, B=args.B, lr=args.lr, ns=ns, seed=args.seed,
              log_every=2000, eval_every=5000, tag=args.tag)

    acc = lab.ens_acc(ens, n=8, batches=6, B=8192)          # structured, held out
    top = torch.argsort(acc, descending=True)[: max(args.keep, 8)]
    print("top structured acc:", [round(float(acc[i]), 5) for i in top[:8]], flush=True)
    rows = []
    for i in top.tolist():
        sd = ens.member_sd(i)
        m = lab.member_model(cfg, sd)
        a_u, _ = lab.model_acc(m, n=8, batches=4, B=16384, mix=(1.0, 0.0, 0.0))
        a_s, _ = lab.model_acc(m, n=8, batches=4, B=16384)
        rows.append((i, a_u, a_s, sd))
    rows.sort(key=lambda r: (min(r[1], r[2]), r[2]), reverse=True)
    for j, (i, a_u, a_s, sd) in enumerate(rows[: args.keep]):
        p = os.path.join(args.outdir, f"{args.tag}_{j}.pt")
        lab.save_ckpt(p, cfg, sd, note=f"member {i} uniform {a_u:.5f} struct {a_s:.5f}")
        print(f"saved {p}: member {i} uniform {a_u:.5f} struct {a_s:.5f}", flush=True)


if __name__ == "__main__":
    main()
