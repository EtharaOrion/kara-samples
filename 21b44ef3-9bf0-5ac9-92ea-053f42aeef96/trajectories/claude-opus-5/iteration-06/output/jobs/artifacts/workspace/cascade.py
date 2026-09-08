"""Walk the shrinking ladder: each rung starts from the rung above it.

From-scratch training fails well before the sizes we want, so every rung is
initialised from the previous checkpoint (member 0 exact, the rest jittered) and
then retrained -- it still has to earn its accuracy on held-out pairs.  A rung
that clears the bar becomes the parent of the next one and, if it is the
smallest passing model so far, is written straight to the graded path.
"""

import argparse, json, os, subprocess, sys, time

import torch

from adapt import adapt
from model_src import default_cfg

# override -> what it buys, applied cumulatively from the parent down
LADDER = [
    ("A", {"C": 2, "U1": 4}),                                  # trim answer subspace + FFN
    ("B", {"C": 1}),                                           # one answer axis
    ("B2", {"code_fix": 1}),                                   # code[0] = 0 (forced, not a gauge)
    ("B3", {"code_fix": 2}),                                   # code[1] = 1 (scale gauge)
    ("C", {"f1_id_in": True, "f1_out": 2, "U1": 3}),           # confine the FFN
    ("D", {"kv_id": True, "wo_out": 0}),                       # keys/values are the FFN's axis
    ("E", {"q_proj": False}),                                  # query is a learned constant
    ("F", {"D": 2, "f1_out": 1}),                              # drop the provably dead axis
    ("G", {"share_q": True}),                                  # both heads ask the same question
    ("H", {"lam": [-10.0, -10.0]}),                            # fix the distance slope
    ("I", {"logit_scale": False}),                             # readout scale is a gauge
    ("J", {"wo_fix": 1}),                                      # head 0's write sets the scale
]

BAR = 0.995          # eval-set bar to accept a rung (graded bar is 0.99)


def run(cfg, init, out, tag, E, steps, sigma, lr, seed, log_every=1000):
    cmd = [sys.executable, "train.py", "--cfg", json.dumps(cfg), "--E", str(E),
           "--steps", str(steps), "--sigma", str(sigma), "--lr", str(lr),
           "--seed", str(seed), "--out", out, "--tag", tag,
           "--log_every", str(log_every)]
    if init:
        cmd += ["--init", init]
    with open(f"logs/{tag}.log", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    return torch.load(out, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="ck/p3.pt")
    ap.add_argument("--start", default="A")
    ap.add_argument("--stop", default="J")
    ap.add_argument("--E", type=int, default=192)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=0.006)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--bar", type=float, default=BAR)
    a = ap.parse_args()

    os.makedirs("ck", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    names = [n for n, _ in LADDER]
    lo, hi = names.index(a.start), names.index(a.stop)

    ck_path = a.parent
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = default_cfg(**ck["cfg"])
    print(f"parent {ck_path}: {ck['nparam']} params, acc {ck['acc']:.5f}", flush=True)

    for name, ov in LADDER[:hi + 1]:
        cfg = default_cfg(**{**cfg, **ov})
        if names.index(name) < lo:
            continue
        seed_ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        init = adapt(seed_ck, ov)
        init_path = f"ck/{name}_init.pt"
        torch.save(init, init_path)

        best = None
        for t in range(a.tries):
            tag = f"{name}{t}"
            sigma = a.sigma * (1 + t)
            steps = a.steps * (1 + t)
            t0 = time.time()
            got = run(cfg, init_path, f"ck/{tag}.pt", tag, a.E, steps, sigma,
                      a.lr, seed=100 + 7 * t, log_every=max(500, steps // 8))
            print(f"  {tag}: {got['nparam']} params  acc {got['acc']:.5f}  "
                  f"sigma {sigma:.2f} steps {steps}  {time.time()-t0:.0f}s", flush=True)
            if best is None or got["acc"] > best["acc"]:
                best = got
                torch.save(best, f"ck/{name}.pt")
            if best["acc"] >= a.bar:
                break

        print(f"[rung {name}] {best['nparam']} params  best acc {best['acc']:.5f}"
              f"  {'OK' if best['acc'] >= a.bar else 'BELOW BAR'}", flush=True)
        if best["acc"] < a.bar:
            print("  stopping: rung did not clear the bar", flush=True)
            return
        ck_path = f"ck/{name}.pt"


if __name__ == "__main__":
    main()
