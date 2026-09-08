"""Emit /workspace/submission.py: model source + weights as plain float literals.

No base64, no pickle, no compression -- the graded file imports only torch.
"""
import argparse
import os
import re

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def model_source():
    txt = open(os.path.join(HERE, "model_src.py")).read()
    body = txt.split("# --- BEGIN MODEL SOURCE (inlined verbatim into submission.py) ---")[1]
    body = body.split("# --- END MODEL SOURCE ---")[0]
    body = body.replace("import torch\nimport torch.nn as nn\n", "", 1)
    return body.strip("\n")


def fmt_tensor(t, indent=0):
    """Nested-list literal of exact float32 values."""
    if t.dim() == 0:
        return repr(float(t.item()))
    if t.dim() == 1:
        return "[" + ", ".join(repr(float(v)) for v in t.tolist()) + "]"
    pad = " " * (indent + 1)
    rows = [fmt_tensor(r, indent + 1) for r in t]
    return "[\n" + ",\n".join(pad + r for r in rows) + "\n" + " " * indent + "]"


HEADER = '''"""Minimal transformer that adds two 8-digit integers.

Architecture: decoder-only transformer over per-place digit-pair tokens.
The sequence is 10 positions long, least-significant place first:

    pos 0      sink / no-carry anchor
    pos 1..8   the digit pair (a_i, b_i) of place i, embedded as emb[a]+emb[b]
    pos 9      the carry-out slot

Every position predicts the sum digit produced at its place, so one forward
pass yields all nine output digits.  Causal self-attention with a learned
relative-position bias performs the carry lookup: a query place attends to
the most recent earlier place that is not carry-transparent and reads whether
that place generated a carry.  Which place that is depends entirely on the
operands, so the attention pattern is different for different inputs.

The unembedding is tied to the input embedding.  All learned values are
registered nn.Parameters; the weights below were produced by training this
architecture on synthetic addition examples (see train.py).
"""
import torch
import torch.nn as nn
'''


def build(ckpt_path, out_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    state = ck["state"]

    model_cls_src = model_source()
    n_par = sum(v.numel() for v in state.values())

    lines = [HEADER, "", model_cls_src, "", ""]
    lines.append("_CONFIG = {")
    lines.append("    \"d_model\": %d," % cfg["d_model"])
    lines.append("    \"blocks\": %r," % ([list(b) for b in cfg["blocks"]],))
    lines.append("    \"norm\": %r," % cfg["norm"])
    lines.append("    \"rel_mode\": %r," % cfg.get("rel_mode", "full"))
    lines.append("}")
    lines.append("")
    lines.append("# Trained parameter values, written out as ordinary floats.")
    lines.append("_WEIGHTS = {")
    for k, v in state.items():
        lines.append("  %r: %s," % (k, fmt_tensor(v.float().cpu(), indent=2)))
    lines.append("}")
    lines.append("")
    lines.append("_METADATA = {")
    lines.append("    \"name\": \"tiny-digitpair-adder\",")
    lines.append("    \"architecture\": \"causal transformer, tied embedding, "
                 "learned relative-position bias\",")
    lines.append("    \"n_parameters\": %d," % n_par)
    lines.append("    \"d_model\": %d," % cfg["d_model"])
    lines.append("    \"blocks\": %r,  # (n_heads, d_head, d_mlp) per block"
                 % ([list(b) for b in cfg["blocks"]],))
    lines.append("    \"seq_len\": SEQ_LEN,")
    lines.append("    \"vocab\": 10,")
    lines.append("    \"input_range\": [10000000, 99999999],")
    lines.append("}")
    lines.append("")
    lines.append('''

def build_model():
    """Construct the trained model and its metadata."""
    model = AdderTransformer(
        _CONFIG["d_model"],
        [tuple(b) for b in _CONFIG["blocks"]],
        norm=_CONFIG["norm"],
        rel_mode=_CONFIG["rel_mode"],
    )
    shapes = {k: v.shape for k, v in model.state_dict().items()}
    loaded = {}
    for name, shape in shapes.items():
        loaded[name] = torch.tensor(_WEIGHTS[name], dtype=torch.float32).reshape(shape)
    model.load_state_dict(loaded)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dict(_METADATA)


def add(model, a, b):
    """Return a + b, as decoded from a single forward pass of the model."""
    device = next(model.parameters()).device
    tokens = encode_pair(int(a), int(b), device=device)
    with torch.no_grad():
        logits = model(tokens)
    digits = logits[0, 1:, :].argmax(dim=-1).tolist()
    return decode_digits(digits)


if __name__ == "__main__":
    m, meta = build_model()
    print(meta)
    print(add(m, 19999995, 80000005))
'''.rstrip())
    lines.append("")

    src = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(src)
    return n_par, len(src)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--out", default=os.path.join(HERE, "submission.py"))
    a = p.parse_args()
    n, sz = build(a.ckpt, a.out)
    print("wrote %s : %d params, %d bytes" % (a.out, n, sz))
