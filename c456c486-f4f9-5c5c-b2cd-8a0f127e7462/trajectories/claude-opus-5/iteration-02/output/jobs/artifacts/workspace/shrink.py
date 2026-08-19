"""Shrink a trained TinyAdder into a narrower one, keeping the useful weights.

Channels 0 and 1 of the residual stream carry the digit code and the pair
feature; any further channels are scratch that the attention head writes into,
so narrowing keeps the leading channels.  Feed-forward units are ranked by the
size of their contribution and the weakest are dropped.  The result is a
starting point for fine-tuning, not a finished model.
"""

import argparse
import os

import torch

from model_def import TinyAdder

HERE = os.path.dirname(os.path.abspath(__file__))


def build(cfg, **over):
    c = dict(d_model=cfg["d_model"], d_ff=cfg["d_ff"], head=cfg["head"],
             mlp_full_out=cfg["mlp_out"] == "full", pair_mode=cfg.get("pair_mode", "table"),
             d_feat=cfg.get("d_feat", 3))
    c.update(over)
    return TinyAdder(**c)


def shrink(ckpt_path, out_path, d_model, d_ff, head=None, mlp_out=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    src = build(cfg)
    src.load_state_dict(ck["state"], strict=False)

    new_cfg = dict(cfg)
    new_cfg["d_model"] = d_model
    new_cfg["d_ff"] = d_ff
    if head:
        new_cfg["head"] = head
    if mlp_out:
        new_cfg["mlp_out"] = mlp_out
    dst = build(new_cfg)

    dm = min(src.d_model, d_model)
    # rank feed-forward units by how much they move the output
    imp = src.w2.detach().abs().sum(1) * src.w1.detach().abs().sum(0)
    keep = torch.argsort(imp, descending=True)[:d_ff]
    keep = keep[torch.argsort(keep)]
    kf = min(len(keep), d_ff)

    with torch.no_grad():
        dst.digit_code.copy_(src.digit_code)
        if hasattr(dst, "pair_code") and hasattr(src, "pair_code"):
            dst.pair_code.copy_(src.pair_code)
        for n in ("wq", "wk", "wv"):
            getattr(dst, n)[:dm].copy_(getattr(src, n)[:dm])
        for n in ("bq", "bk", "bv", "slope"):
            getattr(dst, n).copy_(getattr(src, n))
        dst.wo[0, :dm].copy_(src.wo[0, :dm])
        dst.w1[:dm, :kf].copy_(src.w1[:dm][:, keep[:kf]])
        dst.b1[:kf].copy_(src.b1[keep[:kf]])
        do = min(dst.d_out, src.d_out)
        dst.w2[:kf, :do].copy_(src.w2[keep[:kf]][:, :do])
        dst.b2[:do].copy_(src.b2[:do])
        if dst.head == src.head:
            if dst.head == "linear":
                dst.wout[:dm].copy_(src.wout[:dm])
                dst.bout.copy_(src.bout)
            elif dst.head == "dist":
                dst.centre.copy_(src.centre)
                dst.tau.copy_(src.tau)
            else:
                dst.tau.copy_(src.tau)

    n = sum(p.numel() for p in dst.parameters())
    torch.save({"state": dst.state_dict(), "config": new_cfg, "params": n,
                "uniform": -1.0, "stress": -1.0}, out_path)
    print(f"shrank {ckpt_path} ({sum(p.numel() for p in src.parameters())} params) -> "
          f"{out_path} (d_model={d_model} d_ff={d_ff}, {n} params)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--out", required=True)
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--d_ff", type=int, required=True)
    p.add_argument("--head", default=None)
    p.add_argument("--mlp_out", default=None)
    a = p.parse_args()
    shrink(a.ckpt, a.out, a.d_model, a.d_ff, a.head, a.mlp_out)
