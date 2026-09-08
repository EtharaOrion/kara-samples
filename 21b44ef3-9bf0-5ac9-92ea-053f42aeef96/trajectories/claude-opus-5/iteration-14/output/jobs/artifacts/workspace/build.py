"""Emit /workspace/submission.py from a trained single-member weight dict.

The graded file is the ship_tmpl.py source with the weights inlined as plain Python
float literals -- it imports nothing but torch/torch.nn, contains no training code
and no data generation.
"""
import os
import torch

import ship_tmpl

ALL_KEYS = ["code", "code0", "Bw", "bb", "kw", "vw", "vb",
            "q", "lam", "w1", "w2", "rb", "ls"]


def to_lit(t):
    t = t.detach().to(torch.float64).cpu()
    if t.dim() == 0:
        return repr(float(t))
    return "[" + ", ".join(to_lit(x) for x in t) + "]"


def fill_defaults(p, cfg):
    """Return a complete weight dict for the shipped template (adds the pieces the
    parent may not carry, as explicit zeros/identities)."""
    C, U = cfg["C"], cfg["U"]
    dev = p["code"].device
    q = dict(p)
    if "code0" not in q:
        q["code0"] = torch.zeros(1, C, device=dev) if cfg["code0_fixed"] else None
    if "vb" not in q:
        q["vb"] = torch.zeros((), device=dev)
    if "rb" not in q:
        q["rb"] = torch.zeros(C, device=dev)
    assert q["code"].shape == (9, C) and q["code0"].shape == (1, C)
    assert q["Bw"].shape == (C, U) and q["bb"].shape == (U,)
    assert q["kw"].shape == (U,) and q["vw"].shape == (U,)
    assert q["w1"].shape == (C,) and q["w2"].shape == (C,) and q["rb"].shape == (C,)
    for k in ("q", "lam", "ls", "vb"):
        assert q[k].dim() == 0, (k, q[k].shape)
    return q


def emit(p, cfg, param_keys, path="submission.py", doc=""):
    q = fill_defaults(p, cfg)
    const_keys = [k for k in ALL_KEYS if k not in param_keys]
    lines = ["_PARAMS = {"]
    for k in ALL_KEYS:
        if k in param_keys:
            lines.append(f"    {k!r}: {to_lit(q[k])},")
    lines.append("}")
    lines.append("")
    lines.append("_CONST = {")
    for k in const_keys:
        lines.append(f"    {k!r}: {to_lit(q[k])},")
    lines.append("}")
    src = ship_tmpl.HEADER.replace("@WEIGHTS@", "\n".join(lines)).replace("@DOC@", doc)
    with open(path, "w") as f:
        f.write(src)
    nparam = sum(int(q[k].numel()) for k in param_keys)
    return nparam, len(src)


def load_member(ck_path, idx=0, device="cpu"):
    st = torch.load(ck_path, map_location=device)
    p = {k: v[idx].to(device) for k, v in st["p"].items()}
    return p, st["cfg"], st


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out", default="submission.py")
    ap.add_argument("--params", default="")   # comma list; default = everything learned
    ap.add_argument("--doc", default="")
    a = ap.parse_args()
    p, cfg, st = load_member(a.ckpt, a.idx)
    keys = a.params.split(",") if a.params else [k for k in ALL_KEYS if k in p]
    n, sz = emit(p, cfg, keys, a.out, a.doc)
    print(f"wrote {a.out}: {n} parameters, {sz} bytes")
