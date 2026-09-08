"""Emit /workspace/submission.py: model_src.py's class text + trained weights.

The graded file must contain the model and its inference path only, and import
nothing but torch, so the weights go in as plain Python float literals.
"""
import argparse, json, re
import torch
import lib
from check import load

HEAD = '''"""Exact 8-digit addition from a {NP}-parameter transformer.

The model is a single transformer block over LSB-first digit-pair tokens: token
i embeds ``code[a_i] + code[b_i]`` from one shared 10-entry code table (also
used as the output prototypes), and the block runs FFN -> strictly-causal
single-head self-attention -> FFN over a {C}-dimensional residual stream.

The FFN before attention turns the local digit sum a_i+b_i into a key that is
sharply lowest exactly where a place is carry-transparent, so each position's
attention lands on the nearest earlier place that either generates or absorbs a
carry and reads that place's carry off the same axis; the FFN after attention
folds the result mod 10.  Attention is therefore doing the long-range work, and
the map it computes is a function of the digits it is given.

All weights below were produced by training (see train.py / README.md); the
answer is the argmax decode of a single forward pass.
"""
import torch
import torch.nn as nn

'''

TAIL = '''

def build_model():
    """Return (model, metadata).  The weights are the trained ones above."""
    model = DigitPairAdder(_CFG)
    sd = model.state_dict()
    for k, v in _W.items():
        sd[k] = torch.tensor(v, dtype=torch.float32).reshape(sd[k].shape)
    model.load_state_dict(sd)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    meta = {
        "name": "digit-pair-adder",
        "param_count": n,
        "architecture": "1 transformer block (FFN -> strictly-causal 1-head self-attention -> FFN)",
        "residual_width": _CFG["C"],
        "tokens": "one token per decimal place, LSB first, embedding code[a_i]+code[b_i]",
        "readout": "nearest of the 10 tied code prototypes, at every position, in one forward pass",
        "trained_on": "synthetic digit pairs; held-out accuracy is measured on a hashed 1-in-16 split",
        "digits": "operands of any width; graded range is 8 digits",
    }
    return model, meta


def add(model, a, b):
    """Exact sum of two non-negative integers, decoded from one forward pass."""
    a, b = int(a), int(b)
    n = max(len(str(a)), len(str(b)))
    da = [0] + [(a // 10 ** i) % 10 for i in range(n)] + [0]
    db = [0] + [(b // 10 ** i) % 10 for i in range(n)] + [0]
    dev = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor([da], device=dev), torch.tensor([db], device=dev))
    d = logits.argmax(-1)[0].tolist()
    return sum(d[i] * 10 ** (i - 1) for i in range(1, n + 2))
'''


def fmt(t):
    t = t.detach().cpu().to(torch.float32)
    if t.dim() == 0:
        return repr(float(t))
    if t.dim() == 1:
        return "[" + ", ".join(repr(float(v)) for v in t) + "]"
    return "[\n" + ",\n".join("        " + fmt(r) for r in t) + ",\n    ]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--out", default="/workspace/submission.py")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = lib.default_cfg(**ck["cfg"])
    pm = {n: t[args.member] for n, t in ck["params"].items()}
    model = load(cfg, pm)
    npar = sum(p.numel() for p in model.parameters())
    assert npar == lib.n_params(cfg), (npar, lib.n_params(cfg))

    src = open("/workspace/model_src.py").read()
    cls = src[src.index("class DigitPairAdder"):].rstrip() + "\n"

    body = [HEAD.replace("{NP}", str(npar)).replace("{C}", str(cfg["C"]))]
    body.append("_CFG = {" + ", ".join(f"{k!r}: {cfg[k]!r}" for k in sorted(cfg)) + "}\n\n")
    w = ["_W = {"]
    for n, t in model.state_dict().items():
        if n in dict(model.named_parameters()):
            w.append(f"    {n!r}: " + fmt(t) + ",")
    w.append("}\n\n")
    body.append("\n".join(w) + "\n")
    body.append(cls)
    body.append(TAIL)
    txt = "".join(body)
    txt = re.sub(r"\n{4,}", "\n\n\n", txt)
    open(args.out, "w").write(txt)
    print(f"wrote {args.out}: {npar} params, cfg={json.dumps(cfg)}")
    print("scores in ckpt:", [round(s, 5) for s in ck.get("scores", [])][:8])


if __name__ == "__main__":
    main()
