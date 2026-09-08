"""Emit /workspace/submission.py from a trained checkpoint.

The graded file is adder.py verbatim with the two placeholder assignments
replaced by the architecture description and the trained weights, written as
plain Python float literals (``repr`` round-trips float32 exactly).  Only
``torch`` is imported.
"""
import argparse
import importlib.util
import os
import sys

import torch

import lab


def fmt(t):
    if t.dim() == 0:
        return repr(float(t))
    return "[" + ", ".join(fmt(x) for x in t) + "]"


def build(ckpt, out="/workspace/submission.py"):
    cfg, sd, note = lab.load_ckpt(ckpt)
    m = lab.member_model(cfg, sd, dev="cpu")
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "adder.py")).read()
    cfg_lit = "_CFG = " + repr(dict(cfg))
    w_lit = "_W = {\n" + "".join(
        f"    {k!r}: {fmt(v.detach())},\n" for k, v in m.named_parameters()) + "}"
    assert src.count("_CFG = None") == 1 and src.count("_W = None") == 1
    src = src.replace("_CFG = None", cfg_lit).replace("_W = None", w_lit)
    tmp = out[:-3] + "_tmp.py"
    with open(tmp, "w") as f:
        f.write(src)

    # the emitted file must reproduce the checkpoint exactly
    spec = importlib.util.spec_from_file_location("_sub_check", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sm, meta = mod.build_model()
    g = torch.Generator(device="cpu").manual_seed(7)
    ad, bd, tgt, _ = lab.sample(4096, 8, g, dev="cpu")
    d = (sm(ad, bd) - m(ad, bd)).abs().max().item()
    assert d == 0.0, f"emitted file differs from checkpoint by {d}"
    n = sum(p.numel() for p in sm.parameters())
    assert n == meta["n_params"] == sum(p.numel() for p in m.parameters())
    os.replace(tmp, out)
    print(f"wrote {out}: {n} parameters, {os.path.getsize(out)} bytes  ({note})")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="/workspace/submission.py")
    a = ap.parse_args()
    build(a.ckpt, a.out)
