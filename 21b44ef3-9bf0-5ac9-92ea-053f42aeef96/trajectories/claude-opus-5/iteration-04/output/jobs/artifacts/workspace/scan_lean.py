"""From-scratch trainability of the confined (lean) layouts.

The cascade needs to know where the cliff is: which of these still find the
solution from a random start, so we only have to warm-start below that point.
"""
import sys
import scan
from ladder import LEAN, LEAN1, n

JOBS = [
    ("lz1_86", LEAN1(8, 6)),
    ("lz1_64", LEAN1(6, 4)),
    ("lz1_43", LEAN1(4, 3)),
    ("lz1_32", LEAN1(3, 2)),
    ("lz3_65", LEAN(6, 5)),
    ("lz3_32", LEAN(3, 2)),
]

if __name__ == "__main__":
    for t, c in JOBS:
        print(f"{t:8s} {n(c):4d}", flush=True)
    par = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for tag, line in scan.run(JOBS, E=512, steps=30000, par=par,
                              extra=["--eval_every", "2000"]):
        print(line, flush=True)
