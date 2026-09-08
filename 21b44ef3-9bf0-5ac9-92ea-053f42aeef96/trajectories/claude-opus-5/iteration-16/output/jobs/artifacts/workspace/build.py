"""Emit /workspace/submission.py = template source (verbatim) + weight literals.

The graded file imports only torch, contains no training or data-generation
code, and stores weights as plain Python float literals.
"""

import argparse
import torch


def fmt(x):
    """Nested Python list literal of a tensor, using repr() so the float32
    values round-trip exactly."""
    if x.dim() == 0:
        return repr(float(x))
    return "[" + ", ".join(fmt(v) for v in x) + "]"


def emit(template, weights, out, extra=None):
    src = open(template).read()
    body = ["{"]
    for k, v in weights.items():
        if torch.is_tensor(v):
            body.append(f"    {k!r}: {fmt(v.detach().float().cpu())},")
        else:
            body.append(f"    {k!r}: {v!r},")
    body.append("}")
    lit = "\n".join(body)
    marker = "_W = {}  # WEIGHTS"
    assert marker in src, "template is missing the weight marker"
    src = src.replace(marker, "_W = " + lit)
    with open(out, "w") as f:
        f.write(src)
    print(f"wrote {out} ({len(src)} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--template", default="tpl_generic.py")
    ap.add_argument("--out", default="/workspace/submission.py")
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()

    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    p = d["params"]
    w = {k: v[a.index] for k, v in p.items()}
    w["C"] = d["C"]; w["U"] = d["U"]
    emit(a.template, w, a.out)


if __name__ == "__main__":
    main()
