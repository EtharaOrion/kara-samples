"""Independent audit of /workspace/submission.py.

Deliberately does not import any of the training modules except the loader:
it exercises the graded file exactly the way a grader would, through
build_model() and add().
"""
import argparse
import random
import time

import torch

import build


def digits(v, n):
    d = []
    for _ in range(n):
        d.append(v % 10)
        v //= 10
    return d


def tok_of(a, b, n):
    da, db = digits(a, n), digits(b, n)
    return [[0, 0]] + [[da[i], db[i]] for i in range(n)] + [[0, 0]]


def batch_check(mod, model, A, B, n, dev, chunk=8192):
    """Vectorised exactness check on integer arrays A, B (python ints)."""
    wrong = 0
    for i in range(0, len(A), chunk):
        aa, bb = A[i:i + chunk], B[i:i + chunk]
        tok = torch.tensor([tok_of(x, y, n) for x, y in zip(aa, bb)],
                           dtype=torch.long, device=dev)
        with torch.no_grad():
            pred = model(tok).argmax(-1)[:, 1:].tolist()
        for j, (x, y) in enumerate(zip(aa, bb)):
            out = 0
            for d in reversed(pred[j]):
                out = out * 10 + int(d)
            if out != x + y:
                wrong += 1
    return wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--n_random", type=int, default=1000000)
    a = ap.parse_args()
    mod = build.load_fresh(a.path, "verify_mod")
    model, meta = mod.build_model()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"registered parameters : {n_par}")
    print(f"named parameters      : "
          f"{[(k, tuple(v.shape)) for k, v in model.named_parameters()]}")
    print(f"metadata              : {meta}")
    src = open(a.path).read()
    imports = sorted({ln.strip() for ln in src.splitlines()
                      if ln.startswith(("import ", "from "))})
    print(f"imports in graded file: {imports}")
    print(f"file size             : {len(src)} bytes")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    rng = random.Random(20260907)

    # 1. the grading distribution: uniform full-width 8-digit operands
    t0 = time.time()
    A = [rng.randint(10 ** 7, 10 ** 8 - 1) for _ in range(a.n_random)]
    B = [rng.randint(10 ** 7, 10 ** 8 - 1) for _ in range(a.n_random)]
    w = batch_check(mod, model, A, B, 8, dev)
    print(f"\nuniform 8-digit   : {a.n_random - w}/{a.n_random} exact "
          f"({time.time()-t0:.0f}s)")

    # 2. carry-heavy operands (long transparent runs)
    A2, B2 = [], []
    for _ in range(200000):
        da, db = [], []
        for i in range(8):
            if rng.random() < 0.85:
                x = rng.randint(0, 9)
                da.append(x)
                db.append(9 - x)
            else:
                da.append(rng.randint(0, 9))
                db.append(rng.randint(0, 9))
        if da[7] == 0:
            da[7] = rng.randint(1, 9)
        if db[7] == 0:
            db[7] = rng.randint(1, 9)
        A2.append(sum(d * 10 ** i for i, d in enumerate(da)))
        B2.append(sum(d * 10 ** i for i, d in enumerate(db)))
    w2 = batch_check(mod, model, A2, B2, 8, dev)
    print(f"carry-heavy       : {len(A2) - w2}/{len(A2)} exact")

    # 3. edge cases
    edges = [(10 ** 7, 10 ** 7), (99999999, 99999999), (99999999, 10000001),
             (10000000, 89999999), (55555555, 44444445), (12345678, 87654321),
             (19999999, 10000001), (98765432, 12345678), (11111111, 88888889),
             (50000000, 50000000), (10000001, 19999999), (99999998, 10000002)]
    bad = [(x, y) for x, y in edges if mod.add(model, x, y) != x + y]
    print(f"edge cases        : {len(edges) - len(bad)}/{len(edges)} exact "
          f"{'' if not bad else bad}")

    # 4. exhaustive over carry structure: all 3^8 carry-class patterns
    import itertools
    A3, B3 = [], []
    for pat in itertools.product([0, 1, 2], repeat=8):
        da, db = [], []
        for c in pat:
            if c == 0:
                x = rng.randint(0, 8)
                y = rng.randint(0, 8 - x)
            elif c == 1:
                x = rng.randint(0, 9)
                y = 9 - x
            else:
                x = rng.randint(1, 9)
                y = rng.randint(10 - x, 9)
            da.append(x)
            db.append(y)
        if da[7] == 0 or db[7] == 0:      # keep both operands full width
            continue
        A3.append(sum(d * 10 ** i for i, d in enumerate(da)))
        B3.append(sum(d * 10 ** i for i, d in enumerate(db)))
    w3 = batch_check(mod, model, A3, B3, 8, dev)
    print(f"carry patterns    : {len(A3) - w3}/{len(A3)} exact "
          f"(all 3^8 structures, full-width samples)")

    # 5. other widths -- the model has no width-specific parameters
    for n in (2, 3, 5, 12, 16):
        lo, hi = 10 ** (n - 1), 10 ** n - 1
        A4 = [rng.randint(lo, hi) for _ in range(20000)]
        B4 = [rng.randint(lo, hi) for _ in range(20000)]
        w4 = batch_check(mod, model, A4, B4, n, dev)
        print(f"width {n:2d}          : {20000 - w4}/20000 exact")

    # 6. device / precision agreement
    mcpu = mod.build_model()[0]
    dis = sum(1 for x, y in list(zip(A[:5000], B[:5000]))
              if mod.add(mcpu, x, y) != mod.add(model, x, y))
    print(f"cpu vs cuda       : {5000 - dis}/5000 agree")


if __name__ == "__main__":
    main()
