"""Emit /workspace/submission.py: trained weights as plain float literals plus
the shipped model source, spliced verbatim out of model_src.py.

The graded file imports nothing but `torch`.
"""

import argparse
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BEGIN = "# ---- BEGIN SHIPPED ----"
END = "# ---- END SHIPPED ----"

HEADER = '''"""Minimal transformer that adds two 8-digit integers.

One block, a single scalar residual channel, 12 learned parameters.  Each token
is one decimal place carrying the digit pair (a_p, b_p), least significant
first, embedded as code[a_p] + code[b_p] from a learned 10-entry table that is
also the read-out prototype set.  A two-unit gate bank turns that scalar into
one key/value stream; a strictly-causal head reads the carry into the place and
an inclusively-causal head reads the carry out of it.  The whole sum is
produced by a single forward pass.

The weights below were trained by /workspace/train.py.
"""

import torch

'''


def shipped_source():
    src = open(os.path.join(HERE, "model_src.py")).read()
    return src[src.index(BEGIN) + len(BEGIN):src.index(END)].strip("\n")


def fmt(x):
    return repr(float(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--member", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "submission.py"))
    a = ap.parse_args()

    p = torch.load(a.ckpt, map_location="cpu")["params"]
    i = a.member
    lines = [HEADER, "_WEIGHTS = {"]
    lines.append("    \"code_free\": [%s]," % ", ".join(fmt(v) for v in p["code_free"][i]))
    lines.append("    \"carry_w\": %s," % fmt(p["carry_w"][i]))
    lines.append("    \"knee\": [%s]," % ", ".join(fmt(v) for v in p["knee"][i]))
    lines.append("    \"fold\": %s," % fmt(p["fold"][i]))
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append(shipped_source())
    lines.append("")
    open(a.out, "w").write("\n".join(lines))
    print("wrote", a.out, "from", a.ckpt, "member", i)


if __name__ == "__main__":
    main()
