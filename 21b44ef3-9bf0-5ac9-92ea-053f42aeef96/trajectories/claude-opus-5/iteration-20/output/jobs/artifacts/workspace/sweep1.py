"""Small grid over phase-1 hyper-parameters, reported as members reaching 1.0."""

import itertools
import subprocess
import sys

GRID = list(itertools.product(
    ["l1", "l2"],          # training metric (inference is unchanged)
    [0.5, 1.0, 2.0, 4.0],  # loss temperature
    [1.0, 3.0, 6.0],       # code init sigma
))

for metric, ls, sigma in GRID:
    tag = f"s1_{metric}_{ls}_{sigma}"
    cmd = [sys.executable, "train.py", "--ensemble", "4096", "--steps", "2000",
           "--places", "1", "--eval_places", "1", "--eval_every", "2000",
           "--metric", metric, "--ls", str(ls), "--code_sigma", str(sigma),
           "--out", tag + ".pt"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    last = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    print(f"{metric:3s} ls={ls:<5} sigma={sigma:<4} -> {last[-1] if last else r.stderr[-300:]}",
          flush=True)
