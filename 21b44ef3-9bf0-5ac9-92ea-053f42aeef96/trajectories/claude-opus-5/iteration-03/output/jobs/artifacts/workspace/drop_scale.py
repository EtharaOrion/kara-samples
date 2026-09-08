"""Strip parameters that provably cannot change the model's predictions.

`logit_scale` multiplies every logit at a position by one positive scalar, so
it rescales the softmax temperature but never moves the arg max.  Removing it
therefore gives a strictly smaller model with bit-identical predictions --
this script rewrites a checkpoint with the scalar dropped and checks that the
two models agree on every digit of a large sample before the new checkpoint is
used.
"""
import argparse

import torch

from cfgutil import count, mk
from data import sample, targets


def strip(ckpt_in, ckpt_out, member, n_check=200000, seed=7):
    ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    state = {k: v.clone() for k, v in ck["members"][member].items()}

    scale = float(state["logit_scale"])
    assert scale > 0, f"logit_scale must be positive to drop it, got {scale}"

    old = mk(cfg, state)
    new_cfg = dict(cfg, out_scale=False)
    new_state = {k: v for k, v in state.items() if k != "logit_scale"}
    new = mk(new_cfg, new_state)

    g = torch.Generator().manual_seed(seed)
    agree, correct, seen = 0, 0, 0
    with torch.no_grad():
        for _ in range(max(1, n_check // 20000)):
            a, b = sample(20000, "cpu", g)
            po = old(a, b)[:, 1:, :].argmax(-1)
            pn = new(a, b)[:, 1:, :].argmax(-1)
            t = targets(a, b)
            agree += int((po == pn).all(-1).sum())
            correct += int((pn == t).all(-1).sum())
            seen += a.shape[0]
    print(f"[strip] logit_scale={scale:.4f} dropped")
    print(f"[strip] identical predictions on {agree}/{seen} samples")
    print(f"[strip] exact-match accuracy of stripped model: {correct / seen:.6f}")
    assert agree == seen, "predictions changed -- refusing to write"

    n = count(new_cfg)
    assert sum(p.numel() for p in new.parameters()) == n
    torch.save({"cfg": new_cfg, "members": [new_state],
                "acc": [ck["acc"][member]],
                "meta": dict(ck.get("meta", {}), stripped_from=ckpt_in)},
               ckpt_out)
    print(f"[strip] wrote {ckpt_out} with {n} parameters")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--n_check", type=int, default=200000)
    a = ap.parse_args()
    strip(a.ckpt, a.out, a.member, a.n_check)
