"""Emit /workspace/submission.py from a trained checkpoint.

The graded file gets the model source verbatim from adder.py plus the trained
weights as plain Python float literals.  Its only import is torch.
"""

import argparse
import re
import torch

HEADER = '''"""Minimal transformer for exact 8-digit decimal addition.

12 learned parameters.  See build_model() for what each of them is.
The weights below were produced by gradient descent on synthetic addition
problems (trainer: train.py, not imported here); this file contains the model
and its inference path only.
"""

import torch
import torch.nn as nn

'''

FOOTER = '''

_WEIGHTS = {{
    "code_free": {code_free},
    "bank_bias": {bank_bias},
    "fold": {fold},
}}


def build_model():
    """Return (model, metadata).  All 12 learned floats are nn.Parameters."""
    model = DigitPairAdder()
    with torch.no_grad():
        model.code_free.copy_(torch.tensor(_WEIGHTS["code_free"], dtype=torch.float32))
        model.bank_bias.copy_(torch.tensor(_WEIGHTS["bank_bias"], dtype=torch.float32))
        model.fold.copy_(torch.tensor(_WEIGHTS["fold"], dtype=torch.float32))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    metadata = {{
        "name": "DigitPairAdder",
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "parameter_breakdown": {{
            "code_free": "9 - the learned digit code, tied embedding / read-out prototypes",
            "bank_bias": "2 - the two clamp knees that split a+b into <=8 / ==9 / >=10",
            "fold": "1 - the write weight of the carry-out head (the mod-10 fold)",
        }},
        "architecture": "1 block: 2-unit clamp bank -> 2 masked attention heads "
                        "(strictly causal, inclusively causal) sharing one key "
                        "and one value stream -> tied prototype read-out",
        "sequence": "P = places + 2 tokens, least-significant place first, "
                    "(0,0) pads at both ends; every answer digit comes from "
                    "one forward pass",
        "input_range": [10000000, 99999999],
        "trained": "from random initialisation on synthetic addition problems",
    }}
    return model, metadata
'''


def extract_source(path="adder.py"):
    src = open(path).read()
    body = src.split("MODEL_SRC_BEGIN = None", 1)[1].split("# ---------------------------------------------------------------------------\n# END MODEL SOURCE", 1)[0]
    return body.strip("\n")


def fmt(t):
    if t.dim() == 0:
        return repr(float(t))
    return "[" + ", ".join(repr(float(v)) for v in t.reshape(-1)) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--out", default="submission.py")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    p = ck["params"] if "params" in ck else ck
    i = args.member
    code_free = p["code_free"][i].double()
    bank_bias = p["bank_bias"][i].double()
    fold = p["fold"][i].double()

    body = extract_source()
    text = HEADER + body + "\n" + FOOTER.format(
        code_free=fmt(code_free), bank_bias=fmt(bank_bias), fold=fmt(fold))
    with open(args.out, "w") as f:
        f.write(text)

    # the shipped file must import nothing but torch
    imports = sorted(set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", text, re.M)))
    print(f"wrote {args.out}; imports={imports}")
    print(f"code={[round(v,6) for v in code_free.tolist()]}")
    print(f"bank_bias={[round(v,6) for v in bank_bias.tolist()]} fold={float(fold):.6f}")


if __name__ == "__main__":
    main()
