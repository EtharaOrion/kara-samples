"""Shrink the two FFNs from wherever the parent already is.

The structural cuts (fewer ReLUs) and the gauge cuts (fewer coordinates) are
independent, so this takes whatever config a checkpoint happens to carry --
gauge cuts included -- and walks only the (f1, f2) counts down from it.  That
lets the descent restart from the best parent available rather than from the
rung the ladder happened to reach first.

The floor is (3, 2): the carry key needs a knee at 9 with a flat tail on both
sides, which is three ReLUs, and the mod-10 fold needs two.

Usage:  python descend.py <checkpoint> [steps]
"""
import sys
import torch
import cascade
from ladder import LEAN, LEAN1, n

STEPS = [(a, b) for a, b in [(7, 5), (5, 4), (4, 3), (3, 3), (3, 2)]]
KEEP = ("strict", "self_bias", "alibi_fix", "code_fix", "f1_id_in", "f2_id_in")


def sig(cfg):
    """A short name for which gauge cuts a parent is already carrying, so two
    descents that differ only in those do not overwrite each other."""
    s = "".join(c for c, k in (("s", cfg.get("strict")),
                               ("a", cfg.get("alibi_fix") is not None),
                               ("1", cfg.get("f1_id_in")),
                               ("2", cfg.get("f2_id_in"))) if k)
    return s + "c" * cfg.get("code_fix", 0)


def rungs(cfg, steps):
    fam, pre = (LEAN1, "o") if cfg["d"] == 2 else (LEAN, "")
    gauge = {k: cfg[k] for k in KEEP if k in cfg}
    pre += "d" + sig(cfg) + "_"
    out = []
    for a, b in STEPS:
        if (a, b) >= (cfg["f1"], cfg["f2"]):
            continue
        out.append((f"{pre}{a}{b}", fam(a, b, **gauge), steps))
    return out


if __name__ == "__main__":
    init = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 24000
    cfg = torch.load(init, map_location="cpu", weights_only=False)["cfg"]
    rs = rungs(cfg, steps)
    for t, c, s in rs:
        print(f"{t:8s} {n(c):5d} {s:6d} steps", flush=True)
    cascade.chain(rs, init=init, E=256, batch=1024, lr=0.012, sigma=0.3)
