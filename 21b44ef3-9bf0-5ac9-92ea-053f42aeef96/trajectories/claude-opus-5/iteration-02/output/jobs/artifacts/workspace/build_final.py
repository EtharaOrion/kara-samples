"""Emit /workspace/submission.py from a trained checkpoint.

The graded file contains the model source, the trained weights written out as
ordinary Python floats, build_model() and add().  Nothing else: it imports
only torch.
"""
import argparse
import ast
import os

import torch

import final_model
from final_model import convert

HERE = os.path.dirname(os.path.abspath(__file__))


def model_source():
    txt = open(os.path.join(HERE, "final_model.py")).read()
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


HEADER = '''"""A minimal transformer that adds two 8-digit integers.

The model is a single Macaron-style transformer block -- feed-forward, causal
self-attention, feed-forward -- over a sequence of per-place digit-pair tokens.
The sequence is 10 positions long, least-significant place first:

    pos 0      a sink that anchors "no carry from below"
    pos 1..8   digit place i, embedded as emb[a_i] + emb[b_i]
    pos 9      the carry-out slot

Every position predicts the sum digit produced at its own place, so a single
forward pass yields all nine output digits.

The self-attention is what performs carry propagation.  The first
feed-forward marks each place as carry-generating (a_i + b_i >= 10) or
carry-transparent (a_i + b_i == 9); the attention head then makes each place
attend to the nearest earlier place that is *not* transparent and read off
whether that place generated a carry.  Which position that is depends entirely
on the operands -- the head jumps over runs of 9s of whatever length the input
happens to contain -- so the attention pattern is genuinely input-dependent.
Replacing it with its average over inputs destroys the model's accuracy.

The unembedding is tied to the input embedding, normalisation is
parameter-free RMS, and every learned value is a registered nn.Parameter.  The
weights below are the result of training this architecture on synthetic
addition examples; the training code is not part of this file.
"""
import torch
import torch.nn as nn
'''

FOOTER = '''

def build_model():
    """Construct the trained model and return it with its metadata."""
    model = AdderTransformer(
        _ARCH["d_model"],
        _ARCH["d_ff_in"],
        _ARCH["n_heads"],
        _ARCH["d_head"],
        _ARCH["d_ff_out"],
    )
    target = model.state_dict()
    loaded = {}
    for name, ref in target.items():
        loaded[name] = torch.tensor(_WEIGHTS[name], dtype=torch.float32).reshape(ref.shape)
    model.load_state_dict(loaded)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dict(_METADATA)


def add(model, a, b):
    """Return a + b, decoded from a single forward pass of the model."""
    device = next(model.parameters()).device
    tokens = encode_pair(int(a), int(b), device=device)
    with torch.no_grad():
        logits = model(tokens)
    digits = logits[0, 1:, :].argmax(dim=-1).tolist()
    return decode_digits(digits)


if __name__ == "__main__":
    model, metadata = build_model()
    print(metadata)
    for x, y in [(19999995, 80000005), (12345678, 87654321), (10000000, 10000000)]:
        print(x, "+", y, "=", add(model, x, y))
'''


def build(ckpt_path, out_path, extra_meta=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, arch = convert(ck)
    state = model.state_dict()
    n_par = sum(v.numel() for v in state.values())

    out = [HEADER, "", model_source(), "", ""]
    out.append("_ARCH = {")
    for k in ("d_model", "d_ff_in", "n_heads", "d_head", "d_ff_out"):
        out.append("    %r: %d," % (k, arch[k]))
    out.append("}")
    out.append("")
    out.append("# The trained parameter values, written out as ordinary floats.")
    out.append("_WEIGHTS = {")
    for k, v in state.items():
        out.append("  %r: %s," % (k, fmt_tensor(v.float().cpu(), indent=2)))
    out.append("}")
    out.append("")
    out.append("_METADATA = {")
    out.append("    \"name\": \"macaron-digitpair-adder\",")
    out.append("    \"architecture\": \"1 block: feed-forward -> causal self-attention "
               "-> feed-forward; tied embedding; parameter-free RMS norm\",")
    out.append("    \"n_parameters\": %d," % n_par)
    out.append("    \"n_attention_layers\": 1,")
    for k in ("d_model", "d_ff_in", "n_heads", "d_head", "d_ff_out"):
        out.append("    %r: %d," % (k, arch[k]))
    out.append("    \"seq_len\": SEQ_LEN,")
    out.append("    \"vocab\": 10,")
    out.append("    \"tokenisation\": \"one token per decimal place, LSB first; "
               "emb[a_i] + emb[b_i]\",")
    out.append("    \"input_range\": [10000000, 99999999],")
    for k, v in (extra_meta or {}).items():
        out.append("    %r: %r," % (k, v))
    out.append("}")
    out.append(FOOTER.rstrip())
    out.append("")

    src = "\n".join(out)
    with open(out_path, "w") as f:
        f.write(src)

    # sanity: the emitted file must import nothing but torch
    tree = ast.parse(src)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {n.name.split(".")[0] for n in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert mods <= {"torch"}, "unexpected imports: %s" % sorted(mods)
    return n_par, len(src), model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--out", default=os.path.join(HERE, "submission.py"))
    a = p.parse_args()
    n, sz, _ = build(a.ckpt, a.out)
    print("wrote %s : %d params, %d bytes" % (a.out, n, sz))
