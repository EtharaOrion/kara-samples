"""Emit /workspace/submission.py from a trained checkpoint.

Copies the model source out of model_def.py verbatim and inlines the trained
weights, so the graded file contains the model and its inference path only.
"""

import argparse
import os

import torch

from model_def import TinyAdder

HERE = os.path.dirname(os.path.abspath(__file__))

HEADER = '''"""Minimal transformer that adds two integers in [0, 99_999_999_999_999].

A single-layer, single-head decoder-only transformer, trained from scratch on
sampled operand pairs (training code is not part of this file).  The operands
are tokenised into digit slots, least-significant first, and the model emits
all 15 sum digits in one forward pass: its attention head implements a learned
carry lookahead, letting each digit slot read the carry out of the nearest
earlier slot whose digit pair does not merely propagate one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

'''

FOOTER = '''

def build_model():
    """Return the trained model and a metadata dict."""
    model = TinyAdder({cfg})
    for name, p in model.named_parameters():
        w = torch.tensor(WEIGHTS[name], dtype=torch.float32).reshape(p.shape)
        p.data.copy_(w)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    meta = {{
        "name": "tiny-carry-lookahead-adder",
        "parameters": n,
        "architecture": "1-layer 1-head decoder-only transformer, d_model={d_model}, d_ff={d_ff}",
        "max_operand": 10 ** 14 - 1,
        "input": "digit-pair slots, least-significant first",
        "output": "15 sum digits, one forward pass",
    }}
    return model, meta


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two integers in [0, 99_999_999_999_999] via one forward pass."""
    dev = next(model.parameters()).device
    da = torch.tensor([[0] + [(int(a) // 10 ** i) % 10 for i in range(SEQ - 1)]], device=dev)
    db = torch.tensor([[0] + [(int(b) // 10 ** i) % 10 for i in range(SEQ - 1)]], device=dev)
    digits = model(da, db).argmax(-1)[0, 1:].tolist()
    return sum(d * 10 ** i for i, d in enumerate(digits))
'''


def fmt(t):
    t = t.detach().cpu().float()
    if t.dim() == 0:
        return repr(float(t))
    if t.dim() == 1:
        return "[" + ", ".join(repr(float(v)) for v in t) + "]"
    return "[" + ", ".join(fmt(r) for r in t) + "]"


def emit(ckpt_path, out_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = TinyAdder(d_model=cfg["d_model"], d_ff=cfg["d_ff"],
                      head=cfg["head"],
                      mlp_full_out=cfg["mlp_out"] == "full",
                      pair_mode=cfg.get("pair_mode", "table"),
                      d_feat=cfg.get("d_feat", 3))
    model.load_state_dict(ck["state"], strict=False)

    src = open(os.path.join(HERE, "model_def.py")).read()
    body = src[src.index("NDIG = 15"):].rstrip() + "\n"

    lines = ["WEIGHTS = {"]
    for name, p in model.named_parameters():
        lines.append("    %r: %s," % (name, fmt(p)))
    lines.append("}\n")

    kw = "d_model=%d, d_ff=%d, head=%r, mlp_full_out=%r, pair_mode=%r, d_feat=%d" % (
        cfg["d_model"], cfg["d_ff"], cfg["head"], cfg["mlp_out"] == "full",
        cfg.get("pair_mode", "table"), cfg.get("d_feat", 3))
    text = (HEADER + body + "\n" + "\n".join(lines) +
            FOOTER.format(cfg=kw, d_model=cfg["d_model"], d_ff=cfg["d_ff"]))
    with open(out_path, "w") as f:
        f.write(text)
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote {out_path} from {ckpt_path}: {n} params")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--out", default=os.path.join(HERE, "submission.py"))
    emit(*vars(p.parse_args()).values())
