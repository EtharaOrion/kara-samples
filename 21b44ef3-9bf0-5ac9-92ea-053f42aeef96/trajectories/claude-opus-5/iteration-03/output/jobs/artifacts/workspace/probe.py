"""Ask the shipped model which of its scalars it could do without.

Sets each parameter to zero, one at a time, and measures exact-match accuracy.
A scalar whose removal costs nothing is a scalar that should not be counted --
either it is genuinely unused, or the architecture has a redundancy that a
rewrite could quotient out.  Every reduction taken so far was found this way
and then justified independently; the probe locates candidates, it does not
license removing them.

Zeroing is tested on two distributions, because a scalar can be idle on uniform
operands and essential on carry chains: `p` is the probability that a place is
transparent (a+b == 9), so p=0.85 makes long chains common.
"""
import argparse
import importlib.util

import torch

from verify import batched_acc, enriched_pairs, load, rand_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--p", type=float, default=0.85)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(a.path).build_model()[0].to(dev)
    named = list(model.named_parameters())
    total = sum(p.numel() for _, p in named)

    g = torch.Generator(device=dev).manual_seed(1234)
    au, bu = rand_pairs(a.n, dev, g)
    ac, bc = enriched_pairs(a.n, dev, g, a.p)
    ev = lambda: (batched_acc(model, au, bu), batched_acc(model, ac, bc))

    base_u, base_c = ev()
    print(f"{a.path}: {total} parameters")
    print(f"baseline  uniform {base_u:.6f}   p={a.p} {base_c:.6f}\n")

    rows = []
    for name, p in named:
        flat = p.data.view(-1)
        for i in range(flat.numel()):
            old = float(flat[i])
            if old == 0.0:
                continue
            flat[i] = 0.0
            u, c = ev()
            flat[i] = old
            rows.append((min(u, c), name, i, old, u, c))

    rows.sort(reverse=True)
    print(f"{'parameter':>12} {'value':>10} {'zeroed: uniform':>16} "
          f"{'p=' + str(a.p):>10}")
    for worst, name, i, old, u, c in rows:
        flag = "  <-- survives zeroing" if worst > 0.99 else ""
        print(f"{name + '[' + str(i) + ']':>12} {old:10.4f} {u:16.6f} "
              f"{c:10.6f}{flag}")
    free = [r for r in rows if r[0] > 0.99]
    print(f"\n{len(free)} of {total} scalars survive individual zeroing at 99%.")


if __name__ == "__main__":
    main()
