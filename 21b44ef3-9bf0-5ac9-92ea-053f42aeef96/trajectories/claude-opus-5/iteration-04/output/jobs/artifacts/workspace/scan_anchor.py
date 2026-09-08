"""From-scratch anchors for the cascade, with the readout fixed to axis 0.

Tying the prototypes to the embedding costs the readout its affine freedom:
a readout matrix would scale the query side while the prototypes, being the
same tensor as the embedding, kept their own scale.  So the whole family reads
the answer straight off axis 0, and these runs check that the wide end of the
ladder still trains that way -- and how far down it keeps training.
"""
import sys
import scan
from ladder import CFG, n

JOBS = [
    ("an_8", CFG(8, 16, 16)),
    ("an_5", CFG(5, 12, 10)),
    ("an_4", CFG(4, 10, 8)),
    ("an_3", CFG(3, 9, 7)),
]

if __name__ == "__main__":
    for t, c in JOBS:
        print(f"{t:6s} {n(c):5d}", flush=True)
    par = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for tag, line in scan.run(JOBS, E=256, steps=25000, par=par,
                              extra=["--eval_every", "2500"]):
        print(line, flush=True)
