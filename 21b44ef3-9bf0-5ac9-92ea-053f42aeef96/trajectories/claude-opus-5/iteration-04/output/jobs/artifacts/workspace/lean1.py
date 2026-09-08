"""Move the block from three residual axes to two, then shrink again.

In the three-axis layout f1 writes two features -- a key that marks "this place
is not a nine" and a value that marks "this place carries out" -- so its output
matrix has two columns.  Two axes make one feature do both jobs and the matrix
loses a column per unit, at the cost of giving the attention's output scale
back.  At three units that is six parameters for four.

The merge is not a gauge move: the value's share of the attention score is
real, and so is the key's share of the answer.  Both are only tolerable in a
narrow band -- the value must not outrank one step of recency, the key must not
drift by a code step once the output scale has multiplied it up -- and a parent
that never had to keep its key flat generally lands outside that band.  So the
merge happens at the *widest* rung available, where the extra units give the
child room to flatten the key itself, and the descent follows afterwards.

Usage:  python lean1.py <LEAN-checkpoint> [steps]
"""
import sys

import torch

import cascade
from ladder import LEAN1, n

# Merge first, wide, then walk back down.  The first entry is deliberately
# wider in f1 than the parent: the merged feature has to be flat where the
# parent's key never had to be, which is one more knee to place.
STEPS = [(5, 3), (4, 3), (3, 3), (3, 2)]
KEEP = ("strict", "self_bias", "code_fix")


if __name__ == "__main__":
    init = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 24000
    cfg = torch.load(init, map_location="cpu", weights_only=False)["cfg"]
    gauge = {k: cfg[k] for k in KEEP if k in cfg}
    rs = [(f"o{a}{b}", LEAN1(a, b, **gauge), steps) for a, b in STEPS]
    for t, c, s in rs:
        print(f"{t:8s} {n(c):5d} {s:6d} steps", flush=True)
    cascade.chain(rs, init=init, E=256, batch=1024, lr=0.012, sigma=0.5)
