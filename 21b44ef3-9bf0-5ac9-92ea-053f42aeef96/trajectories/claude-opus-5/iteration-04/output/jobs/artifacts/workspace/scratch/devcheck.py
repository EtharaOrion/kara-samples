"""Does the shipped model answer the same way on someone else's hardware?

The readout is a squared distance between numbers about 0.126 apart and the
tightest decision it ever makes sits 0.037 of that from the boundary, so it is
worth checking that the answer does not depend on where the arithmetic runs:
CPU against CUDA, and CUDA with TF32 matmuls enabled (10 mantissa bits) against
full float32.

Usage:  python scratch/devcheck.py [n]
"""
import sys

import torch

import data
import submission
from margin import patterns


@torch.no_grad()
def preds(dev, tf32, dig):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    m, _ = submission.build_model()
    m = m.to(dev)
    out = []
    for i in range(0, dig.shape[0], 65536):
        x = data.tokens(dig[i:i + 65536].to(dev))
        out.append(m(x)[:, 1:, :].argmax(-1).cpu())
    return torch.cat(out)


def main(n):
    dig = torch.cat([data.eval_set(n, "cpu", 31, data.UNIFORM),
                     data.eval_set(n, "cpu", 41, data.HARD),
                     patterns()])
    tgt = data.targets(dig)
    base = preds("cpu", False, dig)
    print(f"n={dig.shape[0]}")
    print(f"cpu   float32          exact-match {float((base == tgt).all(-1).float().mean()):.6f}")
    if not torch.cuda.is_available():
        print("no cuda on this box; cpu is the reference")
        return
    for tf32 in (False, True):
        p = preds("cuda", tf32, dig)
        acc = float((p == tgt).all(-1).float().mean())
        dis = int((p != base).any(-1).sum())
        print(f"cuda  {'tf32   ' if tf32 else 'float32'}         "
              f"exact-match {acc:.6f}  disagreements with cpu {dis}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200000)
