"""Grader-style check of /workspace/submission.py (import, param count, accuracy)."""

import argparse
import random
import sys
import time

import torch


def main(n, path):
    sys.path.insert(0, path)
    import submission

    model, meta = submission.build_model()
    nparam = sum(p.numel() for p in model.parameters())
    assert isinstance(model, torch.nn.Module)
    print("params:", nparam, "| metadata:", meta)

    rng = random.Random(1234)
    cases = [(0, 0), (0, 10 ** 14 - 1), (10 ** 14 - 1, 10 ** 14 - 1), (1, 10 ** 14 - 1),
             (10 ** 14 - 1, 1), (99999999999999, 99999999999999)]
    for k in range(15):
        cases += [(10 ** k - 1, 1), (1, 10 ** k - 1), (10 ** k, 10 ** k), (5 * 10 ** k, 5 * 10 ** k)]
    for _ in range(n):
        cases.append((rng.randint(0, 10 ** 14 - 1), rng.randint(0, 10 ** 14 - 1)))
    for _ in range(n // 4):  # varied lengths
        cases.append((rng.randint(0, 10 ** rng.randint(1, 14) - 1),
                      rng.randint(0, 10 ** rng.randint(1, 14) - 1)))
    for _ in range(n // 4):  # carry-chain stress
        a = rng.randint(0, 10 ** 14 - 1)
        cases.append((a, 10 ** 14 - 1 - a))

    t0 = time.time()
    bad = []
    for a, b in cases:
        got = submission.add(model, a, b)
        if got != a + b:
            bad.append((a, b, got, a + b))
    acc = 1 - len(bad) / len(cases)
    print(f"checked {len(cases)} cases in {time.time()-t0:.1f}s -> accuracy {acc:.6f}")
    if bad:
        print("failures (first 10):")
        for row in bad[:10]:
            print("  ", row)
    return acc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=2000)
    p.add_argument("--path", default="/workspace")
    a = p.parse_args()
    main(a.n, a.path)
