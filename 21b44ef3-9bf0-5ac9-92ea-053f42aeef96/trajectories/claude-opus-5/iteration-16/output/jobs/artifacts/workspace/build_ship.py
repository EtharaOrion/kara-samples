"""Emit the 12-parameter submission from a stage-3 checkpoint.

Checks that the member really is in the shipped pose (every constant already
equal to the architectural value) before extracting the twelve free numbers, so
nothing can silently ship with a fitted value hiding in a buffer.
"""

import argparse
import torch

import arch
import build
import stage3

CONSTANTS = {
    "Wb": lambda v: torch.allclose(v, torch.full_like(v, stage3.BANK_W)),
    "kw": lambda v: torch.allclose(v, torch.tensor([-stage3.KEY_W, stage3.KEY_W],
                                                   device=v.device)),
    "vw": lambda v: torch.allclose(v, torch.tensor([0.0, 1.0], device=v.device)),
    "vb": lambda v: torch.allclose(v, torch.zeros_like(v)),
    "uA": lambda v: torch.allclose(v, torch.full_like(v, stage3.CARRY_W)),
    "q": lambda v: torch.allclose(v, torch.full_like(v, stage3.Q)),
    "lam": lambda v: torch.allclose(v, torch.full_like(v, stage3.LAM)),
    "ls": lambda v: torch.allclose(v, torch.full_like(v, stage3.LS)),
}


def extract(ckpt, index):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    p = {k: v[index:index + 1] for k, v in d["params"].items()}
    for k, ok in CONSTANTS.items():
        assert ok(p[k][0]), f"{k} is not at its architectural constant: {p[k][0]}"
    assert abs(float(p["code"][0, 0, 0])) < 1e-12, "code[0] is not pinned to 0"
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/ship.pt")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="/workspace/submission.py")
    a = ap.parse_args()

    p = extract(a.ckpt, a.index)
    w = {
        "code": p["code"][0, 1:, 0],     # digits 1..9; digit 0 pinned at 0
        "knee": p["bb"][0],
        "fold": p["uB"][0, 0],
    }
    build.emit("tpl_ship.py", w, a.out)

    # the emitted module must reproduce the training-time forward exactly
    import importlib.util
    spec = importlib.util.spec_from_file_location("shipped", a.out)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    model, meta = mod.build_model()
    n_par = sum(t.numel() for t in model.parameters())
    assert n_par == 12, f"expected 12 parameters, got {n_par}"

    import data
    g = torch.Generator(device="cpu"); g.manual_seed(99)
    da, db = data.batch(512, 8, "cpu", g)
    dap, dbp = data.pad(da, db)
    ref = arch.forward({k: v.cpu() for k, v in p.items()}, dap, dbp)[0]
    got = model(dap, dbp)
    print(f"parameters: {n_par}   max |logit difference| vs training model: "
          f"{float((ref - got).abs().max()):.3e}")
    print("metadata:", meta)


if __name__ == "__main__":
    main()
