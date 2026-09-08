"""Zero-shot cost of each gauge cut, on the real task.

The cuts are meant to be re-parameterisations: the child has one number fewer
but computes the same function, and the retraining that follows is there to let
it settle, not to rebuild it.  So a cut that lands below 1.0000 here is either
not exact or not absorbed, and that is worth knowing before a rung spends an
hour finding out.

Cuts accumulate, exactly as cascade_gauge applies them, but every rung is
remapped straight from the same parent -- no retraining in between -- so the
number shown is what the child inherits, not what it can reach.

Usage:  python scratch/zeroshot.py <parent-checkpoint>
"""
import sys

import torch

import cascade_gauge
import data
import warm
from ladder import LEAN, LEAN1, n
from tryfix import acc

if __name__ == "__main__":
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    pcfg, pp = ck["cfg"], ck["params"]
    fam = LEAN1 if pcfg["d"] == 2 else LEAN
    du = data.eval_set(4096, "cpu", 1234, data.UNIFORM)
    dh = data.eval_set(4096, "cpu", 5678, data.HARD)
    print(f"parent   {n(pcfg):3d}p  u {acc(pcfg, pp, du):.4f}  h {acc(pcfg, pp, dh):.4f}")
    accum = {}
    for tag, cut in cascade_gauge.CUTS:
        accum = dict(accum, **cut)
        c = fam(pcfg["f1"], pcfg["f2"], **accum)
        cp = warm.remap(pcfg, pp, c)
        print(f"{tag:8s} {n(c):3d}p  u {acc(c, cp, du):.4f}  h {acc(c, cp, dh):.4f}",
              flush=True)
