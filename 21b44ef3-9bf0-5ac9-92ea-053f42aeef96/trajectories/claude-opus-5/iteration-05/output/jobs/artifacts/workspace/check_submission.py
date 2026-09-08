"""Smoke-test /workspace/submission.py: params, metadata, and N random pairs."""
import argparse
import importlib.util
import random
import sys


def load(path='/workspace/submission.py'):
    spec = importlib.util.spec_from_file_location('sub', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='/workspace/submission.py')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    m = load(args.path)
    model, meta = m.build_model()
    n = sum(p.numel() for p in model.parameters())
    rng = random.Random(args.seed)
    bad = 0
    for _ in range(args.n):
        a = rng.randint(10 ** 7, 10 ** 8 - 1)
        b = rng.randint(10 ** 7, 10 ** 8 - 1)
        if m.add(model, a, b) != a + b:
            bad += 1
    print(f'params {n}  errors {bad}/{args.n}  meta_params {meta.get("parameters")}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
