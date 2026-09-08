"""Emit /workspace/submission.py from a trained checkpoint.

The graded file gets: the model class source (copied verbatim from
model_src.py), the trained weights as plain Python float literals, build_model()
and add().  Nothing else -- it imports only torch.
"""
import argparse
import json
import os

import torch

import cfgutil

HEADER = '''"""Minimal transformer that adds two 8-digit integers.

Trained from scratch on synthetic addition (see train.py / ens.py in the same
workspace); the weights below are the trained parameters, inlined as literals.
Every answer returned by add() comes from a forward pass of this model.
"""
'''

TAIL = '''

_SHAPES = {shapes!r}

_CFG = {cfg!r}


def build_model():
    """Return (model, metadata).  The model is ready for inference."""
    model = AdderTransformer(**_CFG)
    state = {{}}
    for name, shape in _SHAPES.items():
        flat = torch.tensor(_flatten(_WEIGHTS[name]), dtype=torch.float32)
        state[name] = flat.reshape(shape)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    meta = {{
        "name": "tiny-adder-transformer",
        "architecture": "1 macaron transformer block (FFN -> 1-head causal "
                        "self-attention -> FFN) over per-place digit-pair tokens",
        "n_parameters": n_params,
        "d_model": _CFG["d_model"],
        "n_layers": 1,
        "n_heads": 1,
        "d_ff_in": _CFG["d_ff_in"],
        "d_ff_out": _CFG["d_ff_out"],
        "vocab_size": 10,
        "seq_len": _CFG["n_pos"],
        "tokenization": "one token per decimal place, LSB first; token "
                        "embedding = emb[a_i] + emb[b_i]; tied unembedding",
        "digits": 8,
        "held_out_exact_match": {acc!r},
    }}
    return model, meta


def _flatten(x):
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_flatten(v))
        return out
    return [x]


def _digits(n, k=8):
    return [(n // (10 ** i)) % 10 for i in range(k)]


@torch.no_grad()
def add(model, a, b):
    """Return a + b for 8-digit operands, computed by a forward pass."""
    a, b = int(a), int(b)
    k = max(8, len(str(max(a, b))))
    dev = next(model.parameters()).device
    ad = torch.tensor([_digits(a, k)], dtype=torch.long, device=dev)
    bd = torch.tensor([_digits(b, k)], dtype=torch.long, device=dev)
    logits = model(ad, bd)
    pred = logits[0, 1:, :].argmax(-1).tolist()
    return sum(d * (10 ** i) for i, d in enumerate(pred))


if __name__ == "__main__":
    m, info = build_model()
    print(info["n_parameters"], "parameters")
    print("12345678 + 87654321 =", add(m, 12345678, 87654321))
'''


def fmt(t):
    t = t.detach().cpu().to(torch.float32)
    if t.dim() == 0:
        return repr(float(t))
    if t.dim() == 1:
        return "[" + ", ".join(repr(float(v)) for v in t) + "]"
    return "[" + ",\n ".join(fmt(r) for r in t) + "]"


def build(ckpt_path, member, out_path, acc=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    act = cfg.pop("act", "relu")
    state = ck["members"][member]
    if acc is None:
        acc = round(float(ck["acc"][member]), 6)

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "model_src.py")).read()

    shapes = {k: list(v.shape) for k, v in state.items()}
    body = ["_WEIGHTS = {"]
    for k, v in state.items():
        body.append(f"    {k!r}: " + fmt(v).replace("\n", "\n" + " " * 8) + ",")
    body.append("}")
    weights = "\n".join(body)

    full_cfg = dict(cfgutil.kwargs(cfg), n_pos=10, vocab=10, act=act)
    text = (HEADER + src.rstrip() + "\n\n\n" + weights + "\n"
            + TAIL.format(shapes=shapes, cfg=full_cfg, acc=acc))
    with open(out_path, "w") as f:
        f.write(text)
    n = cfgutil.count(cfg)
    print(f"wrote {out_path}: {n} params, member {member}, acc {acc}, "
          f"{os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--member", type=int, default=None)
    ap.add_argument("--out", default="/workspace/submission.py")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m = args.member
    if m is None:
        m = max(ck["acc"], key=lambda k: ck["acc"][k])
    build(args.ckpt, m, args.out)
