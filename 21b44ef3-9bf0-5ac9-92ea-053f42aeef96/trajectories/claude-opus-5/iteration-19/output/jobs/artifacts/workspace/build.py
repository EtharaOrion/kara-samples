"""Emit /workspace/submission.py.

The graded file is the shipped source of `model_src.py` copied verbatim, plus the trained
weights written out as plain Python float literals.  Nothing else: it imports only torch,
contains no training code and no data generator.

    python build.py --ckpt ckpt/r1.pt --index 0
"""
import argparse
import os

import torch

WORK = os.path.dirname(os.path.abspath(__file__))
BEGIN = "# ---- BEGIN SHIPPED SOURCE ----"
END = "# ---- END SHIPPED SOURCE ----"

HEADER = '''"""Minimal transformer for 8-digit addition.

12 learned parameters: a 10-entry digit code (entry 0 pinned at the origin, entries 1-9
learned), two clamp-bank thresholds, and the mod-10 write-back.  See DigitPairAdder for
what the block computes.  Weights below were produced by training in this workspace
(train.py); this file holds the model and its inference path only.
"""
'''


def shipped_source():
    txt = open(os.path.join(WORK, "model_src.py")).read()
    body = txt.split(BEGIN)[1].split(END)[0]
    assert "import" in body
    return body.strip("\n")


def fmt(t):
    v = [repr(float(x)) for x in t.reshape(-1).tolist()]
    return "[" + ", ".join(v) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(WORK, "submission.py"))
    args = ap.parse_args()

    ck = torch.load(os.path.join(WORK, args.ckpt) if not os.path.isabs(args.ckpt)
                    else args.ckpt, map_location="cpu")
    p = ck["params"]
    w = {k: p[k][args.index] for k in ("code_free", "carry_w", "knee", "fold")}

    src = shipped_source()
    weights = ("_WEIGHTS = {\n"
               + "".join(f"    {k!r}: {fmt(v)},\n" for k, v in w.items())
               + "}\n")
    text = HEADER + "\n" + src + "\n\n\n" + weights

    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {args.out} ({len(text)} bytes)")
    for k, v in w.items():
        print(f"  {k}: {[round(float(x), 6) for x in v.tolist()]}")


if __name__ == "__main__":
    main()
