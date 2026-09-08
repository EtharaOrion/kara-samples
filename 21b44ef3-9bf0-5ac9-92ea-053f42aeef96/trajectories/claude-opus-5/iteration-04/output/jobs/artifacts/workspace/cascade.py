"""Run a chain of shrinking configs, each warm-started from the one above it.

Every rung is a real training run: the parent is remapped onto the child's
architecture (member 0 of the ensemble is the parent exactly, the rest are the
parent plus noise) and then all E members train.  So every weight that ends up
in the submission was produced by gradient descent under the final
architecture, not carried over untouched from a larger one.
"""
import json, os, subprocess, sys, time
import torch
from model_src import Adder, count_params


def rung(tag, cfg, steps, E=256, batch=1024, lr=0.012, sigma=0.35, init=None,
         seed=0, temp=None, extra=()):
    out = f"/workspace/ck/{tag}.pt"
    # Without a learned logit scale the prototype gap is pinned near 1, so the
    # loss needs a constant sharpening to have any gradient (see train_ens).
    if temp is None:
        temp = 1.0 if cfg.get("logit_scale", True) else 8.0
    cmd = [sys.executable, "train_ens.py", "--cfg", json.dumps(cfg), "--E", str(E),
           "--steps", str(steps), "--batch", str(batch), "--lr", str(lr),
           "--seed", str(seed), "--out", out, "--tag", tag, "--temp", str(temp),
           "--eval_every", str(max(500, steps // 10))]
    if init:
        cmd += ["--init_from", init, "--sigma", str(sigma)]
    cmd += list(extra)
    with open(f"/workspace/logs/{tag}.log", "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    try:
        ck = torch.load(out, map_location="cpu", weights_only=False)
        return out, ck["acc"], ck["n_params"]
    except Exception as e:                                   # noqa: BLE001
        return None, -1.0, count_params(Adder(cfg))


RETRIES = ((0, None), (1, 0.6), (2, 1.0))


def chain(rungs, init=None, sigma=0.3, **kw):
    """rungs: list of (tag, cfg, steps).  Stops descending when a rung fails.

    A rung that misses the bar is retried with a fresh seed and a wider kick:
    how far the child has to be moved to leave the parent's basin depends on
    what the cut actually removed, and guessing it once is not reliable.
    """
    prev, res = init, []
    for tag, cfg, steps in rungs:
        best = (None, -1.0, 0)
        for seed, sg in RETRIES:
            t0 = time.time()
            sg = sigma if sg is None else sg
            ck, acc, n = rung(f"{tag}_s{seed}", cfg, steps, init=prev,
                              seed=seed, sigma=sg, **kw)
            print(f"{tag:10s} {n:5d} params  seed {seed} sigma {sg:.2f}  "
                  f"acc {acc:.5f}  ({time.time()-t0:.0f}s)", flush=True)
            if acc > best[1]:
                best = (ck, acc, n)
            if best[1] >= 0.995:
                break
        ck, acc, n = best
        res.append((tag, n, acc))
        if ck is None or acc < 0.99:
            print(f"  -> stop: {tag} did not reach 0.99", flush=True)
            break
        prev = ck
    return res
