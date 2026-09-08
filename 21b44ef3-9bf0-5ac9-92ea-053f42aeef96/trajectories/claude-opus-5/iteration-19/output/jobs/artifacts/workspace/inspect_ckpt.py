"""Print what the top members of a checkpoint actually learned.

The code table should come out close to a straight line -- `code[a] + code[b]` has to land
on `code[a+b]` for the read-out to work at all -- so the residual gives the step and the
worst deviation from it, alongside the two thresholds and the two write weights.
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    p = ck["params"]
    for i in range(min(a.k, p["code_free"].shape[0])):
        c = torch.cat([torch.tensor([0.0, 1.0]), p["code_free"][i]]).double()
        step = float(c[1] - c[0])
        lin = float((c - torch.arange(10, dtype=torch.float64) * step).abs().max())
        print(f"[{i}] acc={float(ck['acc'][i]):.4f} margin={float(ck['margin'][i]):+.3f}")
        print(f"    code    {[round(float(x), 4) for x in c]}")
        print(f"    knee    {[round(float(x), 4) for x in p['knee'][i].tolist()]}"
              f"   carry_w {float(p['carry_w'][i]):+.4f}   fold {float(p['fold'][i]):+.4f}")
        print(f"    step {step:.4f}  worst deviation from a straight line {lin:.4f}"
              f"  fold/step {float(p['fold'][i]) / step:+.3f}"
              f"  carry_w/step {float(p['carry_w'][i]) / step:+.3f}")


if __name__ == "__main__":
    main()
