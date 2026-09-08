"""Phase-1 sweep scored by how many members find a linear code."""

import itertools
import subprocess
import sys

GRID = [
    dict(norm=n, ls=ls, lr=lr, batch=bs)
    for n, ls, lr, bs in itertools.product([0, 1], [1.0, 4.0], [0.02, 0.05], [256, 1024])
]

for cfg in GRID:
    ls = cfg["ls"] * (8.0 if cfg["norm"] else 1.0)
    tag = "p1_n%d_ls%g_lr%g_b%d" % (cfg["norm"], ls, cfg["lr"], cfg["batch"])
    cmd = [sys.executable, "train.py", "--ensemble", "4096", "--steps", "25000",
           "--lr", str(cfg["lr"]), "--metric", "l1", "--ls", str(ls),
           "--norm", str(cfg["norm"]), "--batch", str(cfg["batch"]),
           "--code_sigma", "3.0", "--exhaustive1", "0", "--places", "1",
           "--eval_places", "1", "--eval_every", "1250", "--out", tag + ".pt"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    last = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    print(f"{tag:28s} {last[-1] if last else r.stderr[-400:]}", flush=True)
