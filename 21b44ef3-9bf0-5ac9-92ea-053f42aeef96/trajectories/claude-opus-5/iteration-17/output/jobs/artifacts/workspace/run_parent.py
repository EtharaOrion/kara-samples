"""Production run of the two-phase parent training.

Phase 1 (n=1) learns the digit code / mod-10 fold / carry-generate detector.
Phase 2 warm-starts from phase-1 winners, re-inits the spare bank unit, and
learns carry ROUTING on 2/3/5/8 places with the transparency rate ramped in.
Evaluation is on held-out pairs at 8/5/11/3 places -- note 11 places is never
trained on, so exactness there is genuine length generalisation.
"""
import argparse
import time

import torch

import arch
import lab
import two_phase as tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E1", type=int, default=16384)
    ap.add_argument("--E2", type=int, default=8192)
    ap.add_argument("--p1_steps", type=int, default=3000)
    ap.add_argument("--p2_steps", type=int, default=5000)
    ap.add_argument("--lr2", type=float, default=0.03)
    ap.add_argument("--ramp", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="parent.pt")
    a = ap.parse_args()
    dev = "cuda"

    cfg = arch.default_cfg(C=2, U=2, act="clamp", fix_val_w=(0.0, 1.0))
    print("free scalars / member =", arch.n_free_values(cfg), flush=True)

    import os
    p1file = f"phase1_E{a.E1}_s{a.seed}.pt"
    if os.path.exists(p1file):
        d = torch.load(p1file, map_location=dev, weights_only=False)
        w1, b1, keys = d["params"], d["acc"].to(dev), arch.trainable_keys(cfg)
        w1 = {k: v.to(dev) for k, v in w1.items()}
        good = (b1 >= 0.9999).nonzero().flatten()
        print(f"phase1 (cached): {good.numel()}/{a.E1} exact on n=1", flush=True)
    else:
        t0 = time.time()
        b1, w1, keys = tp.phase1(a.E1, a.p1_steps, a.seed, cfg)
        good = (b1 >= 0.9999).nonzero().flatten()
        print(f"phase1: {good.numel()}/{a.E1} exact on n=1 "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save(dict(params={k: v.cpu() for k, v in w1.items()},
                        acc=b1.cpu(), cfg=cfg), p1file)

    rep = (a.E2 + good.numel() - 1) // good.numel()
    idx = good.repeat(rep)[:a.E2]
    base = {k: v[idx].contiguous() for k, v in w1.items()}
    corners = lab.make_eval([8, 5, 11, 3], dev, per_n=1024)

    p2 = tp.reinit_unit0(base, cfg, a.seed + 11)
    t0 = time.time()
    b2, w2 = tp.train(cfg, p2, keys, a.p2_steps, 512, a.lr2, a.seed + 3,
                      corners, [2, 3, 5, 8], pct=0.05, ramp=a.ramp, log="p2")
    print(f"phase2: max {b2.max().item():.5f} "
          f"#=1.0 {(b2>=1.0).sum().item()} "
          f"#>=.999 {(b2>=0.999).sum().item()} ({time.time()-t0:.0f}s)",
          flush=True)

    keep = (b2 >= 0.999).nonzero().flatten()
    torch.save(dict(params={k: v[keep].cpu() for k, v in w2.items()},
                    cfg=cfg, acc=b2[keep].cpu()), a.out)
    print(f"saved {a.out} with {keep.numel()} members", flush=True)


if __name__ == "__main__":
    main()
