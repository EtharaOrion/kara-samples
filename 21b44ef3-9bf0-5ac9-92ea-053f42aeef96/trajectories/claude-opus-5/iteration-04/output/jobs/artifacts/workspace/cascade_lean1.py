"""A second descent, into the two-axis layout: key and value share a feature.

A single scalar phi does both jobs at once.  Three ReLUs can shape it so that
phi(t) = 0 for t <= 8, phi(9) = -delta and phi(t) = e for t >= 10 -- the tail is
flat because the three output weights sum to zero -- and then one axis carries
both the "this place is not a nine" signal the score needs and the "this place
carries out" signal the value needs.  That halves what the first FFN has to
write, which is where the saving is.

Usage:  python cascade_lean1.py <d=3-checkpoint> [first-rung-index]
"""
import sys
import cascade
from ladder import LEAN1, n


def rungs():
    out = []
    for a, b in [(9, 7), (7, 5), (5, 4), (4, 3), (3, 3), (3, 2)]:
        out.append((f"o_{a}{b}", LEAN1(a, b), 14000))
    return out


if __name__ == "__main__":
    init = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rs = rungs()[start:]
    for t, c, s in rs:
        print(f"{t:8s} {n(c):5d} {s:6d} steps", flush=True)
    cascade.chain(rs, init=init, E=256, batch=1024, lr=0.012, sigma=0.3)
