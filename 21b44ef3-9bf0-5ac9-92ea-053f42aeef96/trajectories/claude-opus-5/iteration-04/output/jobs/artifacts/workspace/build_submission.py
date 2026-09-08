"""Emit /workspace/submission.py: model source + learned weights as float literals.

The graded file contains only the architecture and the inference path; it
imports nothing but torch.  Weights are written as plain Python floats (repr
round-trips exactly through float64), no encoding step of any kind.
"""
import argparse, json, re, sys
import torch

import model_src
from model_src import Adder, count_params

HEADER = '''"""Minimal transformer for exact 8-digit decimal addition.

One Macaron block -- FFN, single-head causal self-attention with a learned
relative-position bias, FFN -- over per-place digit-pair tokens.

Token layout (length 10, least-significant place first):

    pos 0 : (0, 0)                  a virtual place with digit sum 0; it neither
                                    generates nor propagates a carry, so it is
                                    the natural terminator for the carry lookahead
    pos i : (a[i-1], b[i-1])        places 0..7 of the two operands
    pos 9 : (0, 0)                  a virtual place 8, which holds the final carry

Position i predicts the output digit of the place it holds, so positions 1..9
emit all 9 digits of the sum in a single forward pass.  Attention is what
carries the carry: each position looks back for the most recent place that does
not propagate (a + b != 9) and reads off whether that place generates
(a + b >= 10).  Those keys and values are computed from the token contents, so
the attention pattern is a function of the input, not a fixed template.

The whole vocabulary is one learned table of ten numbers.  It is the input
embedding -- a token embeds as the sum of its two digits' entries -- and it is
also the set of output prototypes: a position predicts the digit whose entry
its residual stream ends up nearest to.  Training is free to put those ten
numbers anywhere; what it converges on is an evenly spaced ramp, which is what
makes summing the two embeddings mean anything at all.

All weights below were produced by gradient training (see train_ens.py).
"""
'''

FOOTER = '''

_CFG = {cfg}

_W = {weights}


def build_model():
    """Returns (model, metadata)."""
    model = Adder(_CFG)
    sd = model.state_dict()
    for k, v in _W.items():
        t = torch.tensor(v, dtype=torch.float32).reshape(sd[k].shape)
        sd[k] = t
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    meta = {{
        "n_params": sum(p.numel() for p in model.parameters()),
        "d_model": _CFG["d"],
        "n_layers": 1,
        "n_heads": 1,
        "d_head": 1,
        "seq_len": _CFG["T"],
        "vocab": 10,
        "task": "8-digit decimal addition",
        "description": "single Macaron transformer block; positions 1..9 emit "
                       "the 9 sum digits in one forward pass",
    }}
    return model, meta


def _digits(n):
    return [(n // 10 ** i) % 10 for i in range(8)]


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two 8-digit integers, read off one forward pass."""
    da, db = _digits(int(a)), _digits(int(b))
    seq = [[0, 0]] + [[da[i], db[i]] for i in range(8)] + [[0, 0]]
    x = torch.tensor([seq], dtype=torch.long)
    logits = model(x)[0, 1:, :]
    out = 0
    for i, d in enumerate(logits.argmax(-1).tolist()):
        out += d * 10 ** i
    return out


@torch.no_grad()
def add_batch(model, pairs):
    """Vectorised form of `add` for a list of (a, b) pairs."""
    seq = []
    for a, b in pairs:
        da, db = _digits(int(a)), _digits(int(b))
        seq.append([[0, 0]] + [[da[i], db[i]] for i in range(8)] + [[0, 0]])
    x = torch.tensor(seq, dtype=torch.long)
    dig = model(x)[:, 1:, :].argmax(-1)
    pw = torch.tensor([10 ** i for i in range(9)], dtype=torch.long)
    return (dig * pw).sum(-1).tolist()
'''


def model_source():
    src = open(model_src.__file__).read()
    return src.split("# --- BEGIN MODEL ---")[1].split("# --- END MODEL ---")[0].strip()


def fmt(t):
    t = t.detach().cpu().to(torch.float32)
    flat = [repr(float(x)) for x in t.reshape(-1).tolist()]
    return "[" + ", ".join(flat) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="/workspace/submission.py")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, params = ck["cfg"], ck["params"]

    ref = Adder(cfg)
    ref.load_state_dict({k: v.float() for k, v in params.items()})
    ref.eval()

    body = ("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n\n"
            + model_source().split("import torch.nn.functional as F", 1)[1].strip())
    wtxt = "{\n" + "".join(
        f"    {k!r}: {fmt(v)},\n" for k, v in params.items()) + "}"
    text = HEADER + "\n" + body + FOOTER.format(cfg=repr(cfg), weights=wtxt)
    open(a.out, "w").write(text)

    # ---- round-trip check: reloaded file must reproduce the checkpoint bit for bit
    sys.path.insert(0, "/workspace")
    import importlib.util
    spec = importlib.util.spec_from_file_location("sub_check", a.out)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m2, meta = mod.build_model()
    g = torch.Generator().manual_seed(7)
    x = torch.randint(0, 10, (512, cfg["T"], 2), generator=g)
    with torch.no_grad():
        d = (ref(x) - m2(x)).abs().max().item()
    n = count_params(m2)
    assert d == 0.0, f"round-trip mismatch {d}"
    assert n == meta["n_params"] == count_params(ref)
    imports = sorted(set(re.findall(r"^(?:import|from)\s+([\w.]+)", text, re.M)))
    print(f"wrote {a.out}: {n} params, round-trip exact, "
          f"{len(text)} bytes, imports={imports}")


if __name__ == "__main__":
    main()
