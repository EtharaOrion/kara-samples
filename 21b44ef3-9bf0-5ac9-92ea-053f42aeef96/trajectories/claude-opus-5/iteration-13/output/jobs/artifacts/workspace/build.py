"""Emit /workspace/submission.py = verbatim model source + weights as plain float literals.

The graded file imports only `torch` / `torch.nn`. No serialisation formats, no
encodings, no training code, no data generation.

The reduced model drops four of core.SPEC's tensors from the forward pass entirely
because the gauge normalisation has driven them to their identity values:
    rb = 0, vb = 0   the residual stream and the value head have no constant offset
    q  = 1, ls = 1   query/key scale and read-out temperature gauges
Everything else is emitted, split into learned parameters and fixed buffers.
"""

import argparse, os
import torch

IDENTITY = {"rb": 0.0, "vb": 0.0, "q": 1.0, "ls": 1.0}

HEAD = '''"""Minimal transformer that adds two 8-digit numbers.

The weight literals at the bottom of this file are the trained values: a parent
model was trained from scratch by /workspace/train.py, then rewritten into these
coordinates by exact, function-preserving changes of variable (/workspace/reduce.py)
and checked over the entire input domain by /workspace/certify.py. `add()` returns
the argmax decoding of a single forward pass of the returned module -- nothing about
the answer is computed outside the model.

{summary}
"""

'''


def fmt(x):
    if isinstance(x, list):
        return "[" + ", ".join(fmt(v) for v in x) + "]"
    return repr(float(x))


def split_values(p):
    """core.SPEC tensors (leading ensemble axis of 1) -> the module's value dict."""
    g = {k: p[k][0].detach().to(torch.float64).cpu() for k in p}
    for k, want in IDENTITY.items():
        got = float(g[k])
        if abs(got - want) > 1e-12:
            raise ValueError(f"{k}={got!r} is not at its identity value {want}; "
                             f"the model is not in the reduced coordinates")
    vals = {
        # learned
        "code_free": g["code"][1:].tolist(),          # 9
        "bb": g["bb"].tolist(),                       # 2
        "fold": float(g["e"][1]),                     # 1
        # fixed
        "code_zero": [float(g["code"][0])],
        "carry_w": float(g["e"][0]),
        "bw": g["bw"].tolist(),
        "kw": g["kw"].tolist(),
        "vw": g["vw"].tolist(),
        "lam": g["lam"].tolist(),
    }
    if vals["code_zero"] != [0.0] or vals["carry_w"] != 1.0:
        raise ValueError("code[0] must be 0 and e[0] must be 1 in the reduced coordinates")
    return vals


ORDER = ["code_free", "bb", "fold", "code_zero", "carry_w", "bw", "kw", "vw", "lam"]
LEARNED = {"code_free", "bb", "fold"}


def emit(p, path, summary=""):
    vals = split_values(p)
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_src.py")).read()
    body = ["_VALUES = {"]
    for k in ORDER:
        tag = "learned" if k in LEARNED else "fixed"
        body.append("    %r: %s,%s" % (k, fmt(vals[k]), "" if tag == "fixed" else ""))
    body.append("}")
    text = HEAD.format(summary=summary) + src.rstrip() + "\n\n\n" + "\n".join(body) + "\n"
    with open(path, "w") as f:
        f.write(text)
    n = sum(len(vals[k]) if isinstance(vals[k], list) else 1 for k in LEARNED)
    return text, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--out", default="/workspace/submission.py")
    ap.add_argument("--summary", default="")
    args = ap.parse_args()
    st = torch.load(args.ckpt, map_location="cpu")
    p = {k: v[args.member:args.member + 1] for k, v in st["p"].items()}
    _, n = emit(p, args.out, args.summary)
    print(f"wrote {args.out}: {n} learned parameters")


if __name__ == "__main__":
    main()
