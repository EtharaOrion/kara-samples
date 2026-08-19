"""Grader-style check: import ONLY submission.py and test add() on unseen pairs."""

import importlib.util
import random
import sys
import time

spec = importlib.util.spec_from_file_location('submission', '/workspace/submission.py')
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

MAX = 99_999_999_999_999


def main(n=20000, seed=0):
    model, meta = sub.build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print('metadata:', meta)
    print('independently counted parameters:', n_params)

    rng = random.Random(seed)
    cases = [(0, 0), (0, MAX), (MAX, MAX), (MAX, 1), (1, MAX), (0, 1), (9, 1),
             (99999, 1), (5_000_000_000_000, 5_000_000_000_000),
             (12345678901234, 98765432109876), (99999999999999, 99999999999998)]
    # long carry chains
    for k in range(1, 15):
        x = int('9' * k)
        cases += [(x, 1), (x, x), (MAX - x, x + 1)]
    # random unseen pairs
    for _ in range(n):
        cases.append((rng.randint(0, MAX), rng.randint(0, MAX)))
    # random pairs with mixed magnitudes
    for _ in range(n // 2):
        cases.append((rng.randint(0, 10 ** rng.randint(1, 14)),
                      rng.randint(0, 10 ** rng.randint(1, 14))))
    # adversarial: many 9-sum positions
    for _ in range(n // 2):
        da = [rng.randint(0, 9) for _ in range(14)]
        db = [(9 - d) if rng.random() < 0.85 else rng.randint(0, 9) for d in da]
        a = sum(d * 10 ** i for i, d in enumerate(da))
        b = sum(d * 10 ** i for i, d in enumerate(db))
        cases.append((a, b))

    bad = []
    t0 = time.time()
    for a, b in cases:
        if sub.add(model, a, b) != a + b:
            bad.append((a, b))
    dt = time.time() - t0
    print(f'{len(cases)-len(bad)}/{len(cases)} correct '
          f'({100*(1-len(bad)/len(cases)):.4f}%), {dt:.1f}s '
          f'({1000*dt/len(cases):.2f} ms/call)')
    if bad:
        print('failures (first 10):', bad[:10])
    return len(bad)


if __name__ == '__main__':
    sys.exit(1 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000) else 0)
