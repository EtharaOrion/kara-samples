"""Confine both FFNs' stream interface to the digit plane, for free.

After `gauge.py` the embedding occupies the first `emb_rank` coordinates of the
residual stream and the tied readout reads only those.  Two consequences follow,
and each deletes parameters without changing a single prediction:

  * **The pre-attention FFN cannot see the other coordinates.**  It is the first
    thing in the block, so the stream it reads is the embedding and nothing
    else, and the embedding is identically zero outside the plane.  The
    corresponding rows of `w_in1` multiply exact zeros.

  * **The post-attention FFN cannot usefully write them.**  Whatever it puts
    outside the plane reaches the logits only through the RMSNorm denominator,
    which scales all ten class logits at a position by one positive factor.
    That rescales the softmax temperature, exactly as `logit_scale` did, and so
    cannot move the arg max.

The coordinates outside the plane are *not* dead: the pre-attention FFN writes
them and attention and the post-attention FFN read them -- that is the channel
carrying the carry-generate flag.  Only the two ends deleted here are vacuous.

The first claim is exact, so it is checked exactly (the discarded activations
must be zero).  The second changes the logits by a positive scale, so it is
checked by direction: normalising each position's logit vector must reproduce
the original bit for bit, which holds only if the two differ by a positive
factor.
"""
import argparse

import torch

from cfgutil import count, mk
from data import sample, targets


def confine(ckpt_in, ckpt_out, member=0, n_check=200_000, seed=17):
    ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    state = {k: v.detach().clone().double() for k, v in
             ck["members"][member].items()}
    r, d = cfg.get("emb_rank", 0), cfg["d_model"]
    assert r and cfg.get("emb_fixed_up"), "run gauge.py first"
    assert not cfg.get("plane_io"), "already confined"
    assert r < d, "nothing outside the plane to confine"

    old_m = mk(cfg, {k: v.float() for k, v in state.items()}).double()
    old_m.load_state_dict(state)

    # Claim 1, checked exactly: at the pre-attention FFN's input the stream is
    # the embedding alone, which is zero outside the plane.
    g = torch.Generator().manual_seed(seed)
    a, b = sample(20000, "cpu", g)
    with torch.no_grad():
        outside = old_m._n(old_m.embed(a, b))[..., r:].abs().max().item()
    print(f"[confine] stream outside the digit plane at the first FFN: "
          f"max |value| = {outside:g}")
    assert outside == 0.0, "the pre-attention FFN does see those coordinates"

    dropped = {"w_in1": state["w_in1"][r:].flatten().tolist(),
               "w_out2": state["w_out2"][:, r:].flatten().tolist()}
    new = dict(state)
    new["w_in1"] = state["w_in1"][:r]
    new["w_out2"] = state["w_out2"][:, :r]
    if "b_out2" in state:
        dropped["b_out2"] = state["b_out2"][r:].tolist()
        new["b_out2"] = state["b_out2"][:r]
    print("[confine] discarding "
          + ", ".join(f"{k}{[round(x, 3) for x in v]}"
                      for k, v in dropped.items()))

    new_cfg = dict(cfg, plane_io=True)
    new_m = mk(new_cfg, {k: v.float() for k, v in new.items()}).double()
    new_m.load_state_dict(new)

    g = torch.Generator().manual_seed(seed + 1)
    agree = correct = seen = 0
    dirdev, smin, smax = 0.0, float("inf"), 0.0
    with torch.no_grad():
        for _ in range(max(1, n_check // 20000)):
            a, b = sample(20000, "cpu", g)
            lo, ln = old_m(a, b), new_m(a, b)
            # Claim 2: ln = s * lo for some s > 0 at each position.  Equivalent
            # to the two logit vectors having the same direction, which is
            # scale-free and so numerically stable.
            no = lo / lo.norm(dim=-1, keepdim=True)
            nn_ = ln / ln.norm(dim=-1, keepdim=True)
            dirdev = max(dirdev, (no - nn_).abs().max().item())
            s = ln.norm(dim=-1) / lo.norm(dim=-1)
            smin, smax = min(smin, float(s.min())), max(smax, float(s.max()))
            po, pn = lo[:, 1:, :].argmax(-1), ln[:, 1:, :].argmax(-1)
            agree += int((po == pn).all(-1).sum())
            correct += int((pn == targets(a, b)).all(-1).sum())
            seen += a.shape[0]
    print(f"[confine] logit direction unchanged to {dirdev:.3e} over {seen} "
          f"samples")
    print(f"[confine] the induced rescaling stays in [{smin:.4f}, {smax:.4f}] "
          f"-- positive, so the arg max cannot move")
    print(f"[confine] identical predictions on {agree}/{seen} samples")
    print(f"[confine] exact-match accuracy of confined model: "
          f"{correct / seen:.6f}")
    assert dirdev < 1e-10, "logit direction changed -- refusing to write"
    assert smin > 0, "rescaling is not positive -- refusing to write"
    assert agree == seen, "predictions changed -- refusing to write"

    f32 = {k: v.float() for k, v in new.items()}
    om, nm = mk(cfg, {k: v.float() for k, v in state.items()}), mk(new_cfg, f32)
    f32_agree = f32_seen = 0
    g32 = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        for _ in range(max(1, n_check // 20000)):
            a, b = sample(20000, "cpu", g32)
            f32_agree += int((om(a, b)[:, 1:, :].argmax(-1)
                              == nm(a, b)[:, 1:, :].argmax(-1)).all(-1).sum())
            f32_seen += a.shape[0]
    print(f"[confine] float32 arg max agrees on {f32_agree}/{f32_seen} samples")
    assert f32_agree == f32_seen, "float32 predictions changed"

    n = count(new_cfg)
    assert sum(p.numel() for p in nm.parameters()) == n
    torch.save({"cfg": new_cfg, "members": [f32], "acc": [ck["acc"][member]],
                "meta": dict(ck.get("meta", {}), confined_from=ckpt_in)},
               ckpt_out)
    print(f"[confine] wrote {ckpt_out} with {n} parameters "
          f"({count(cfg) - n} removed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--n_check", type=int, default=200000)
    a = ap.parse_args()
    confine(a.ckpt, a.out, a.member, a.n_check)
