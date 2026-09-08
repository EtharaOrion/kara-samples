"""Re-train one rung on its own architecture: a longer, gentler pass.

A rung that stops just above the bar makes a bad parent.  The next cut starts
from a point that is still moving, so the noise kick lands somewhere the
parent's own gradient was about to fix anyway, and the child spends its budget
re-learning what the parent had not finished.  This re-runs a config from its
own checkpoint with a small kick and a longer schedule, so whatever descends
from it starts from something settled.

Usage:  python polish.py <checkpoint> <tag> [steps] [sigma] [seed]
"""
import sys
import torch
import cascade


if __name__ == "__main__":
    ck = sys.argv[1]
    tag = sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 0.12
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 7
    cfg = torch.load(ck, map_location="cpu", weights_only=False)["cfg"]
    out, acc, n = cascade.rung(tag, cfg, steps, init=ck, seed=seed,
                               sigma=sigma, E=256, batch=1024, lr=0.008)
    print(f"{tag:10s} {n:5d} params  acc {acc:.5f}  -> {out}", flush=True)
