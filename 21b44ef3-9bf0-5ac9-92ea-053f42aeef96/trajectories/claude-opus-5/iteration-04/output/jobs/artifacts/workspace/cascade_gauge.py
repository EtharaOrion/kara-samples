"""The tail of the cascade: cuts that remove a coordinate rather than a feature.

By LEAN(3,2) the block has nothing structural left to give up -- three ReLUs is
the floor for the carry feature (a knee at 9 needs a flat tail on both sides,
which is three) and two is the floor for the mod-10 fold.  What is still there
is arithmetic that the parameterisation is paying for twice:

    strict      the diagonal of the attention mask.  A place's own digits say
                nothing about the carry arriving at it, so the learned scalar
                that pushes the diagonal down is doing a constant's job.
    alibi_fix   the recency slope.  Only its size relative to the content score
                matters, and the FFN that writes the key axis already owns that.
                The constant has to be a round number no smaller than the slopes
                training settles on: pinning it low does not rescale the score,
                it divides it, and the child inherits its parent's attention at
                a higher temperature -- a blurrier version of the same argmax.
    code_fix 1  the unit of the answer axis: rescaling the axis and the code
                together changes nothing, so digit 1 may as well sit at 1.
    f1/f2_id_in a ReLU reading one axis has one weight too many -- (w, b, o) and
                (cw, cb, o/c) are the same unit -- so it can read the axis as it
                stands.
    code_fix 2  the origin.  This one is *not* free (the constant write that
                would move it went into absorbing f2_ob), so digit 0 at 0 is a
                normalisation the child has to retrain under, not an identity.

Every rung still retrains, so nothing here is carried over untouched.

Usage:  python cascade_gauge.py <LEAN(3,2)-checkpoint> [first-rung-index]
"""
import sys
import cascade
from ladder import LEAN, LEAN1, n

CUTS = [("strict", dict(strict=True, self_bias=False)),
        ("alifix", dict(alibi_fix=-4.0)),
        ("cfix1", dict(code_fix=1)),
        ("f1id", dict(f1_id_in=True)),
        ("f2id", dict(f2_id_in=True)),
        ("cfix2", dict(code_fix=2))]


def rungs(fam=LEAN, f1=3, f2=2, steps=16000, cuts=CUTS, pre=""):
    acc, out = {}, []
    for tag, cut in cuts:
        acc = dict(acc, **cut)
        out.append((f"{pre}{tag}", fam(f1, f2, **acc), steps))
    return out


if __name__ == "__main__":
    import torch
    init = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    cfg = torch.load(init, map_location="cpu", weights_only=False)["cfg"]
    # A gauge cut removes a coordinate, not a feature, so it commutes with the
    # structural descent: the chain can hang off whatever rung has actually
    # landed.  Take the family and both ReLU counts from the parent.
    fam, pre = (LEAN1, "o") if cfg["d"] == 2 else (LEAN, "")
    tail = f"{cfg['f1']}{cfg['f2']}"
    rs = rungs(fam, cfg["f1"], cfg["f2"], pre=f"{pre}g{tail}_")[start:]
    for t, c, s in rs:
        print(f"{t:8s} {n(c):5d} {s:6d} steps", flush=True)
    cascade.chain(rs, init=init, E=256, batch=1024, lr=0.012, sigma=0.3)
