"""Take a unit out of an FFN by retiring it, then descend onto the smaller bank.

The cascade stops at four ReLUs in the pre-attention FFN.  The trained bank's
four units sit at knees that together make a ramp plus two kinks, and no three
of them can be refitted to that -- a three-unit solution exists (knees at 8, 9
and 10, output weights summing to zero, giving both the "this place is a nine"
key and the "this place carries out" value) but it is a different basin, and a
child dropped into it from the four-unit parent lands nowhere near it.

So retire a unit instead of removing one.  Its output row is scaled to nothing
over the first two thirds of a run while the accuracy term keeps pulling the
rest of the bank into shape, and the ensemble tries every unit at once -- member
e retires unit e mod f.  What comes out has a row of exact zeros, and the rung
below it is then an ordinary cut.

Usage:  python prune.py <checkpoint> <f1|f2> [steps] [sigma]
"""
import sys
import torch
import cascade
from ladder import LEAN, LEAN1, n
from descend import KEEP, sig


if __name__ == "__main__":
    ck = sys.argv[1]
    ffn = sys.argv[2] if len(sys.argv) > 2 else "f1"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 24000
    sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    cfg = torch.load(ck, map_location="cpu", weights_only=False)["cfg"]
    f1, f2 = cfg["f1"], cfg["f2"]
    fam = LEAN1 if cfg["d"] == 2 else LEAN
    gauge = {k: cfg[k] for k in KEEP if k in cfg}
    tag = f"pr{sig(cfg)}_{ffn}{f1}{f2}"

    out, acc, np_ = cascade.rung(tag, cfg, steps, init=ck, seed=3, sigma=sigma,
                                 E=256, batch=1024, lr=0.01,
                                 extra=["--prune_ffn", ffn])
    print(f"{tag:12s} {np_:5d} params  acc {acc:.5f}  -> {out}", flush=True)
    if out is None or acc < 0.99:
        sys.exit(f"{tag} did not survive retiring a {ffn} unit")

    child = fam(f1 - (ffn == "f1"), f2 - (ffn == "f2"), **gauge)
    cascade.chain([(f"{tag}_cut", child, steps)], init=out, E=256, batch=1024,
                  lr=0.012, sigma=0.3)
