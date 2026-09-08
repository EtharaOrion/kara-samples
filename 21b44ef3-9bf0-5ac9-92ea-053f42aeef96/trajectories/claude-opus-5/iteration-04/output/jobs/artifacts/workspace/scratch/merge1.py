"""The d=3 -> d=2 merge, scored on the real task before any retraining.

Dropping to two residual axes makes one feature serve as both the attention's
key and its value, so f1 writes three numbers instead of six.  The merge is not
a gauge move -- the value's contribution to the score is real, and the key's
contribution to the answer is real -- so what matters is how much of the
parent's function survives it, which is what this prints.

Usage:  python scratch/merge1.py <LEAN-checkpoint>
"""
import sys

import torch

import data
import warm
from ladder import LEAN1, n
from tryfix import acc

if __name__ == "__main__":
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    pcfg, pp = ck["cfg"], ck["params"]
    gauge = {k: pcfg[k] for k in ("strict", "self_bias", "code_fix") if k in pcfg}
    du = data.eval_set(4096, "cpu", 1234, data.UNIFORM)
    dh = data.eval_set(4096, "cpu", 5678, data.HARD)
    print(f"parent {n(pcfg):3d}p  u {acc(pcfg, pp, du):.4f}  h {acc(pcfg, pp, dh):.4f}")
    c = LEAN1(pcfg["f1"], pcfg["f2"], **gauge)
    cp = warm.remap(pcfg, pp, c)
    print(f"lean1  {n(c):3d}p  u {acc(c, cp, du):.4f}  h {acc(c, cp, dh):.4f}")
    print("  " + "  ".join(f"{k} {[round(float(x), 4) for x in v.reshape(-1)]}"
                           for k, v in sorted(cp.items()) if v.numel() <= 4))
