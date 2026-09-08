"""Find the tightest decision the shipped model ever has to make.

`margin.py` reports a worst-case boundary distance that is the same on every
sample set, which says it is structural rather than luck.  This locates it: it
sweeps each carry structure many times over with fresh digit pairs, so the
non-affine wobble in the learned code table (the prototype steps differ by
about 1.4% end to end) gets exercised against every carry pattern, and prints
the input that comes closest to flipping.

Usage:  python scratch/worst.py [repeats-per-pattern]
"""
import sys

import torch

import data
import submission
from margin import patterns


def structured(rep, seed):
    """`rep` independent digit-pair draws for each of the 3**8 structures."""
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(rep):
        for code in range(3 ** 8):
            da, db, c = [], [], code
            for i in range(8):
                k, c = c % 3, c // 3
                lo = 1 if i == 7 else 0
                if k == 0:
                    x = rng.randint(lo, 9 - lo); y = 9 - x
                elif k == 1:
                    s = rng.randint(max(10, 2 * lo), 18)
                    x = rng.randint(max(lo, s - 9), min(9, s - lo)); y = s - x
                else:
                    s = rng.randint(2 * lo, 8)
                    x = rng.randint(lo, min(9, s - lo)); y = s - x
                da.append(x); db.append(y)
            out.append([da, db])
    return torch.tensor(out).permute(0, 2, 1)


@torch.no_grad()
def main(rep):
    m, _ = submission.build_model()
    code = m._code()
    step = float(code.sort()[0].diff().abs().max())

    best = (1e9, None, None)
    wrong = 0
    total = 0
    for chunk in range(rep):
        dig = structured(1, 1000 + chunk)
        x = data.tokens(dig)
        lg = m(x)[:, 1:, :]
        tgt = data.targets(dig)
        pred = lg.argmax(-1)
        wrong += int((pred != tgt).any(-1).sum())
        total += dig.shape[0]
        top2 = lg.topk(2, -1).values
        room = (top2[..., 0] - top2[..., 1]) / (2 * step)
        v, idx = room.reshape(-1).min(0)
        if float(v) < best[0]:
            i, p = divmod(int(idx), room.shape[1])
            best = (float(v), dig[i].clone(), p)
        print(f"  sweep {chunk+1}/{rep}  n={total}  wrong={wrong}  "
              f"worst so far {best[0]:.5f} steps", flush=True)

    v, dig, p = best
    a = sum(int(dig[i, 0]) * 10 ** i for i in range(8))
    b = sum(int(dig[i, 1]) * 10 ** i for i in range(8))
    print(f"\ntightest: {a} + {b} = {a + b}, at output place {p}")
    print(f"  boundary distance {v:.5f} prototype steps "
          f"= {v * step:.6f} in answer-axis units")
    print(f"  model says {submission.add(m, a, b)}  (correct: {a + b})")

    # what float32 actually costs, measured the same way
    lg32 = m(data.tokens(dig[None]))[:, 1:, :]
    lg64 = m.double()(data.tokens(dig[None]))[:, 1:, :]
    err = float((lg32.double() - lg64).abs().max()) / (2 * step)
    print(f"  float32 noise on the same quantity {err:.2e} steps "
          f"-> {v / max(err, 1e-12):.0f}x headroom")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
