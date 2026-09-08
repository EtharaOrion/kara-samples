"""Emit /workspace/submission.py from a trained checkpoint.

The graded file gets: the model class (copied verbatim from model_src.py), the
config, the trained weights as plain float literals, build_model() and add().
It imports nothing but torch.
"""

import argparse, os, re, sys

import torch

from model_src import DigitPairAdder, default_cfg

HEADER = '''"""Minimal transformer that adds two 8-digit numbers.

One Macaron-style block (FFN -> multi-head self-attention -> readout) over ten
digit-pair tokens.  {NP} trained parameters; the weights below are the ones
produced by training (see train.py / cascade.py, which are not imported here).

Mechanism, for the reader: the embedding row of a digit doubles as the answer
prototype for that digit, so a place's token code[a] + code[b] already sits at
the prototype of (a + b) when there is no carry.  The FFN turns each place into
a key that is extreme exactly when a + b == 9 -- the places that pass a carry
along -- so the attention softmax skips those and lands on the nearest place
that actually decides the carry.  One head looks strictly before the current
place (the carry coming in) and one includes it (the carry going out, which is
what tells the place to wrap past ten); their two writes move the residual onto
the prototype of the answer digit.
"""

import torch
import torch.nn as nn

'''

FOOTER = '''

_CFG = {CFG}

_W = {W}


def build_model():
    """Return (model, metadata).  Weights are the trained ones, inlined above."""
    model = DigitPairAdder(_CFG)
    own = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in _W.items():
            p = own[name]
            p.copy_(torch.tensor(value, dtype=torch.float32).reshape(p.shape))
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    meta = {
        "name": "digit-pair adder",
        "task": "exact addition of two 8-digit integers",
        "n_params": n,
        "num_parameters": n,
        "parameters": n,
        "architecture": "1-block transformer: FFN -> {H}-head self-attention, tied embedding/readout",
        "n_layers": 1,
        "n_heads": {H},
        "d_model": {D},
        "vocab_size": 10,
        "seq_len": {P},
        "digits": 8,
        "trained": True,
    }
    return model, meta


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two 8-digit operands, read off one forward pass."""
    da = torch.tensor([[0] + [(int(a) // 10 ** i) % 10 for i in range(8)] + [0]])
    db = torch.tensor([[0] + [(int(b) // 10 ** i) % 10 for i in range(8)] + [0]])
    digits = model(da, db).argmax(-1)[0, 1:].tolist()
    return sum(d * 10 ** i for i, d in enumerate(digits))
'''


def class_source(path="model_src.py"):
    src = open(path).read()
    i = src.index("class DigitPairAdder(nn.Module):")
    j = len(src)
    for m in re.finditer(r"^(?:class |def |# ---)", src[i + 10:], re.M):
        j = i + 10 + m.start()
        break
    return src[i:j].rstrip() + "\n"


def fmt(x):
    if isinstance(x, list):
        return "[" + ", ".join(fmt(v) for v in x) + "]"
    return repr(float(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ck")
    ap.add_argument("--out", default="/workspace/submission.py")
    a = ap.parse_args()
    ck = torch.load(a.ck, map_location="cpu", weights_only=False)
    cfg = default_cfg(**ck["cfg"])
    model = DigitPairAdder(cfg)
    sd = ck["sd"]
    own = dict(model.named_parameters())
    assert set(sd) == set(own), (set(sd) ^ set(own))
    n = sum(p.numel() for p in own.values())

    w = "{\n" + "".join(
        f"    {k!r}: {fmt(sd[k].reshape(own[k].shape).tolist())},\n" for k in sorted(sd)
    ) + "}"
    cfg_txt = "{\n" + "".join(f"    {k!r}: {cfg[k]!r},\n" for k in sorted(cfg)) + "}"

    text = (HEADER.replace("{NP}", str(n)) + class_source()
            + FOOTER.replace("{CFG}", cfg_txt).replace("{W}", w)
            .replace("{H}", str(cfg["H"])).replace("{D}", str(cfg["D"]))
            .replace("{P}", str(cfg["P"])))
    open(a.out, "w").write(text)
    print(f"wrote {a.out}: {n} parameters, {len(text)} bytes")


if __name__ == "__main__":
    main()
