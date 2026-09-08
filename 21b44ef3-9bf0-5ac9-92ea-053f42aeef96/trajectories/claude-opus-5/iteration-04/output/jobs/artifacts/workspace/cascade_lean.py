"""The parameter cascade: shrink one architectural element at a time.

Each rung retrains from the rung above it.  The first cuts confine the block
inputs and outputs to named residual axes; the cut into LEAN is pure gauge (the
attention's three scalar scales, the constant write on the answer axis and the
logit scale are all absorbed exactly into the code table); the rest shrink the
two FFNs, which is where the real capacity comes out.

Usage:  python cascade_lean.py <parent-checkpoint> [first-rung-index]
"""
import sys
import cascade
from ladder import CFG, LEAN, LEAN1, n

C3 = dict(d=3, f1=9, f2=7)


def rungs():
    d = CFG(3, 9, 7, sep_qk=False)
    e = CFG(3, 9, 7, sep_qk=False, f2_in=(0, 1), f2_out=(0, 1))
    f = CFG(3, 9, 7, sep_qk=False, f2_in=(0, 1), f2_out=(0, 1),
            f1_in=(0, 1), f1_out=(1, 3), f1_ob=False)
    g = CFG(3, 9, 7, sep_qk=False, f2_in=(0, 1), f2_out=(0, 1),
            f1_in=(0, 1), f1_out=(1, 3), f1_ob=False,
            qk_in=(1, 2), v_in=(2, 3), o_out=(0, 1))
    out = [("cs_d", d, 10000), ("cs_e", e, 12000), ("cs_f", f, 12000),
           ("cs_g", g, 12000), ("cs_h", LEAN(9, 7), 12000)]
    # (3, 2) is the floor: the carry feature needs a knee at 9 with a flat tail
    # on either side, which is three ReLUs, and the mod-10 fold needs two.
    for a, b in [(7, 5), (5, 4), (4, 3), (3, 3), (3, 2)]:
        out.append((f"cs_{a}{b}", LEAN(a, b), 14000))
    return out


if __name__ == "__main__":
    init = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rs = rungs()[start:]
    for t, c, s in rs:
        print(f"{t:8s} {n(c):5d} {s:6d} steps", flush=True)
    cascade.chain(rs, init=init, E=256, batch=1024, lr=0.012, sigma=0.3)
