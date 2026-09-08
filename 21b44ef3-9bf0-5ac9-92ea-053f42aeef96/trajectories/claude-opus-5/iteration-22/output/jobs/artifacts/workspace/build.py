"""Emit /workspace/submission.py = model_src.py with trained weights inlined.

The weights go in as plain Python float literals via repr(), which round-trips
float64 exactly, so the shipped file imports nothing but torch.
"""

import argparse
import torch

import lab


def literal(x):
    if isinstance(x, list):
        return "[" + ", ".join(literal(v) for v in x) + "]"
    return repr(float(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="ckpt/stage3.pt")
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--out", type=str, default="submission.py")
    args = ap.parse_args()

    ck = torch.load(args.src)
    i = args.member
    code = torch.cat([torch.zeros(1), ck["code_free"][i]])       # code[0..9]
    weights = {
        "code_free": code[2:].tolist(),                          # code[2..9]
        "carry_w": float(ck["carry_w"][i]),
        "knee": ck["knee"][i].tolist(),
        "fold": float(ck["fold"][i]),
    }
    assert abs(code[0]) == 0.0 and abs(float(code[1]) - 1.0) < 1e-6, code[:2]

    body = "\n".join(f'    "{k}": {literal(v)},' for k, v in weights.items())
    src = open("model_src.py").read()
    assert "_WEIGHTS = None\n" in src
    src = src.replace("_WEIGHTS = None\n", "_WEIGHTS = {\n" + body + "\n}\n", 1)
    open(args.out, "w").write(src)

    print(f"wrote {args.out} from {args.src} member {i}")
    print("  code    ", [round(v, 6) for v in code.tolist()])
    print("  carry_w ", round(weights["carry_w"], 6),
          " knee", [round(v, 6) for v in weights["knee"]],
          " fold", round(weights["fold"], 6))


if __name__ == "__main__":
    main()
