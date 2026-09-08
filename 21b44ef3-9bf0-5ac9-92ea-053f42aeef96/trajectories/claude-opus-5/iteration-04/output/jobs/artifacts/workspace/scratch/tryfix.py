"""What a gauge cut costs before any retraining, measured on the real task.

warm.check compares child against parent on uniformly random tokens, which is
not the distribution the model is graded on (both operands carry eight digits
and positions 0 and 9 are sinks).  This scores the remapped child directly on
eval_set, so a cut that quietly softens the softmax shows up as lost digits
rather than as a near-perfect agreement score.

Usage:  python scratch/tryfix.py <parent-checkpoint> [values...]
"""
import sys

import torch

import data
import warm
from ladder import LEAN, LEAN1, n


def acc(cfg, params, dig):
    m, miss = warm.load(cfg, params)
    assert not miss, miss
    m.eval()
    with torch.no_grad():
        p = m(data.tokens(dig))[:, 1:, :].argmax(-1)
    y = data.targets(dig)
    return float((p == y).all(-1).float().mean())


if __name__ == "__main__":
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    pcfg, pp = ck["cfg"], ck["params"]
    vals = [float(v) for v in sys.argv[2:]] or [-1, -2, -3, -4, -6, -8]
    fam = LEAN1 if pcfg["d"] == 2 else LEAN
    gauge = {k: pcfg[k] for k in warm.__dict__.get("KEEP", ())} if False else {}
    for k in ("strict", "self_bias", "code_fix", "f1_id_in", "f2_id_in"):
        if k in pcfg:
            gauge[k] = pcfg[k]
    du = data.eval_set(4096, "cpu", 1234, data.UNIFORM)
    dh = data.eval_set(4096, "cpu", 5678, data.HARD)
    print(f"parent {n(pcfg):3d}p  u {acc(pcfg, pp, du):.4f}  h {acc(pcfg, pp, dh):.4f}")
    for v in vals:
        c = fam(pcfg["f1"], pcfg["f2"], **dict(gauge, alibi_fix=v))
        cp = warm.remap(pcfg, pp, c)
        print(f"alibi_fix {v:6.2f}  {n(c):3d}p  u {acc(c, cp, du):.4f} "
              f" h {acc(c, cp, dh):.4f}  agree {warm.check(pcfg, pp, c, cp)[0]:.4f}")
