"""Emit /workspace/submission.py: the model class, trained weights as plain
float literals, and the inference path.  Nothing but torch is imported."""
import argparse
import json
import torch

from model_src import DigitAdder, default_cfg

API = '''

# ---------------------------------------------------------------------------
# Weights below were produced by gradient descent in train.py (see the trainer
# alongside this file); nothing here is hand-set.
# ---------------------------------------------------------------------------

_CFG = {cfg}

_W = {weights}


def _t(x):
    return torch.tensor(x, dtype=torch.float32)


def build_model():
    model = DigitAdder(_CFG)
    sd = model.state_dict()
    for k, v in _W.items():
        sd[k] = _t(v).reshape(sd[k].shape)
    model.load_state_dict(sd)
    model.eval()
    meta = {meta}
    meta["parameters"] = sum(p.numel() for p in model.parameters())
    return model, meta


def _digits(x, n):
    d = []
    for _ in range(n):
        d.append(x % 10)
        x //= 10
    return d


@torch.no_grad()
def add(model, a, b):
    """Sum of a and b, read off one forward pass of the model."""
    n = max(len(str(int(a))), len(str(int(b))))
    da = [0] + _digits(int(a), n) + [0]
    db = [0] + _digits(int(b), n) + [0]
    ta = torch.tensor([da], dtype=torch.long)
    tb = torch.tensor([db], dtype=torch.long)
    pred = model(ta, tb)[0, 0, 1:, :].argmax(-1).tolist()
    out = 0
    for i, d in enumerate(pred):
        out += int(d) * 10 ** i
    return out
'''


def to_list(t):
    if t.dim() == 0:
        return float(t)
    return [to_list(x) for x in t]


def fmt(x, ind=0):
    if isinstance(x, float):
        return repr(x)
    inner = ", ".join(fmt(v, ind + 1) for v in x)
    if len(inner) < 88:
        return "[" + inner + "]"
    pad = " " * (ind * 4 + 4)
    return "[\n" + ",\n".join(pad + fmt(v, ind + 1) for v in x) + "\n" + " " * (ind * 4) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--out", default="/workspace/submission.py")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = default_cfg(ck["cfg"])
    cfg["E"] = 1
    model = DigitAdder(cfg)
    sd = model.state_dict()
    w = {}
    for k in dict(model.named_parameters()):
        v = ck["state"][k][args.member: args.member + 1].to(torch.float32)
        sd[k] = v.reshape(sd[k].shape)
        w[k] = to_list(v.reshape(sd[k].shape))
    model.load_state_dict(sd)
    n_par = sum(p.numel() for p in model.parameters())

    heads = ("strictly-causal head" if cfg["heads"] == 1 else
             "two heads (strictly-causal and inclusively-causal) sharing "
             "one set of keys and values")
    act = {"relu": "ReLU", "clamp": "clamp(.,0,1)"}.get(cfg["act"], cfg["act"])
    meta = {
        "name": "digit-pair single-block transformer",
        "task": "decimal addition, graded at 8 digits",
        "architecture": f"one block: {act} projection -> self-attention with "
                        f"{heads}"
                        + (" -> ReLU fold" if cfg["fold"] else "")
                        + ", tied digit code used as both embedding and read-out",
        "residual_width": cfg["C"],
        "bank_units": cfg["U"],
        "heads": cfg["heads"],
        "key_value_shared": bool(cfg["kv_share"]),
        "sequence": "LSB-first, one token per decimal place, (0,0) pad at each end",
        "activation": act,
        "learned_mechanism": "the key is flat, notched at a place sum of 9 and "
                             "flat again above it, so each head attends to the "
                             "nearest earlier place that is not carry-"
                             "transparent; the two masks read the carry in and "
                             "the carry out of the current place",
        "trained_by": "train.py (AdamW, ensemble seed lottery, warm-started "
                      "shrink ladder)",
    }

    src = open("model_src.py").read().rstrip() + "\n"
    body = API.format(
        cfg=repr(cfg),
        weights="{\n" + ",\n".join(f"    {k!r}: {fmt(v, 1)}" for k, v in w.items()) + ",\n}",
        meta="{\n" + "".join(f"        {k!r}: {v!r},\n" for k, v in meta.items()) + "    }",
    )
    with open(args.out, "w") as f:
        f.write(src + body)
    print(f"wrote {args.out}  parameters {n_par}")


if __name__ == "__main__":
    main()
