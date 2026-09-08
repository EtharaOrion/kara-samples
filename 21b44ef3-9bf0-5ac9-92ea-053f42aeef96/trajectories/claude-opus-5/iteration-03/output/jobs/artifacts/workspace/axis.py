"""Remove the pre-attention FFN's read matrix, for free.

`gauge.py` used the O(d) gauge on the residual stream to pin `emb_up` to
[I | 0], but it did not consume the whole group.  The stabiliser of [I | 0] is
still O(r) x O(d-r): rotating *within* the digit plane leaves the fixed
up-projection unchanged, because [I | 0] diag(R, I) = [R | 0] and the digit
table `emb_lo` simply rotates with it.  That leftover O(r) is a genuine
redundancy in the parameter vector, and it is exactly big enough to canonicalise
one in-plane direction.

With `d_ff_in == 1` the pre-attention FFN reads a single in-plane direction,
`w_in1`.  Two steps remove it entirely:

  1. **Rotate.**  Pick R in O(r) with R' w_in1 = (t, 0, ..., 0)', t = ||w_in1||
     > 0, and apply the stream gauge diag(R, I).  Now the FFN reads the plane's
     first axis and nothing else.

  2. **Absorb the scale.**  relu is positively homogeneous, so for t > 0

         relu(t*z + b) w_in2  =  relu(z + b/t) (t*w_in2),

     and the surviving factor t can be moved into `b_in1` and `w_in2`.  The
     read matrix is then the constant 1 and is not stored at all.

Neither step changes the function, and neither hides a value: the rotation is
absorbed into the digit table, the scale into a bias and a write matrix.  What
is left is an architecture -- the pre-attention FFN's pre-activation *is* the
first residual coordinate -- which `train.py --ffn_in_axis` can train from
scratch, and does.

As in gauge.py the identity is checked in float64, where it should hold to
rounding, with a separate float32 arg-max gate for the model that ships.
"""
import argparse

import torch

from cfgutil import count, mk
from data import sample, targets

_READS = ("w_q", "w_k", "w_v", "w_out1")     # (d, *)  -> Q' W
_WRITES = ("w_in2", "b_in2", "w_o")          # (*, d)  -> W Q


def fix(ckpt_in, ckpt_out, member=0, n_check=200_000, seed=23):
    ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    state = {k: v.detach().clone().double() for k, v in
             ck["members"][member].items()}
    r, d = cfg.get("emb_rank", 0), cfg["d_model"]
    assert r and cfg.get("emb_fixed_up"), "run gauge.py first"
    assert not cfg.get("ffn_in_axis"), "already on-axis"
    assert cfg["d_ff_in"] == 1, "only a single pre-attention unit has one axis"
    dio = r if cfg.get("plane_io") else d
    assert dio >= 2, "nothing to rotate"

    w = state["w_in1"][:, 0]                       # the in-plane read direction
    t = w[:r].norm()
    assert dio == r or w[r:].abs().max() == 0, "read direction leaves the plane"
    assert t > 1e-9, f"read direction is degenerate (norm {float(t):g})"

    # R in O(r) taking w/||w|| to e_0.  Householder, then a sign flip so that
    # det > 0 is not required and the first column is exactly +w/||w||.
    u = (w[:r] / t).clone()
    e0 = torch.zeros(r, dtype=u.dtype)
    e0[0] = 1.0
    v = u - e0
    R = (torch.eye(r, dtype=u.dtype) if v.norm() < 1e-12
         else torch.eye(r, dtype=u.dtype) - 2 * torch.outer(v, v) / v.dot(v))
    # R is symmetric orthogonal with R u = e_0, hence R' w = (t, 0, ..., 0)'.
    chk = (R @ w[:r] - t * e0).abs().max().item()
    print(f"[axis] read direction {[round(float(x), 4) for x in w[:r]]} "
          f"-> ({float(t):.4f}, 0)   residual {chk:.2e}")
    assert chk < 1e-9

    Q = torch.eye(d, dtype=u.dtype)
    Q[:r, :r] = R.t()                              # stream gauge diag(R', I)

    new = {k: v.clone() for k, v in state.items()}
    new.pop("w_in1")
    new["emb_lo"] = state["emb_lo"] @ R.t()
    for k in _READS:
        if k in new:
            new[k] = Q.t() @ state[k]
    for k in _WRITES:
        if k in new:
            new[k] = state[k] @ Q
    # In-plane weights see only the R block.
    for k in ("w_out2", "b_out2"):
        if k in new:
            blk = R.t() if new[k].shape[-1] == r else Q
            new[k] = state[k] @ blk
    # Step 2: absorb t.  `w_in2` writes to the stream, so it has already been
    # rotated above; the scale multiplies that, it does not replace it.
    new["b_in1"] = state["b_in1"] / t
    new["w_in2"] = new["w_in2"] * t

    new_cfg = dict(cfg, ffn_in_axis=True)
    old_d = mk(cfg, {k: v.float() for k, v in state.items()}).double()
    old_d.load_state_dict(state)
    new_d = mk(new_cfg, {k: v.float() for k, v in new.items()}).double()
    new_d.load_state_dict(new)

    tbl = ((old_d._emb() @ Q) - new_d._emb()).abs().max().item()
    print(f"[axis] digit table reproduced (up to the rotation) to {tbl:.2e}")
    assert tbl < 1e-9, "the rotation does not reproduce the digit table"

    g = torch.Generator().manual_seed(seed)
    agree = correct = seen = 0
    dev = mag = 0.0
    with torch.no_grad():
        for _ in range(max(1, n_check // 20000)):
            a, b = sample(20000, "cpu", g)
            lo, ln = old_d(a, b), new_d(a, b)
            dev = max(dev, (lo - ln).abs().max().item())
            mag = max(mag, lo.abs().max().item())
            po, pn = lo[:, 1:, :].argmax(-1), ln[:, 1:, :].argmax(-1)
            agree += int((po == pn).all(-1).sum())
            correct += int((pn == targets(a, b)).all(-1).sum())
            seen += a.shape[0]
    rel = dev / max(mag, 1e-12)
    print(f"[axis] max |logit| deviation over {seen} samples (float64): "
          f"{dev:.3e} ({rel:.2e} relative to a peak logit of {mag:.4g})")
    print(f"[axis] identical predictions on {agree}/{seen} samples")
    print(f"[axis] exact-match accuracy of the on-axis model: "
          f"{correct / seen:.6f}")
    assert rel < 1e-10, "logits changed -- refusing to write"
    assert agree == seen, "predictions changed -- refusing to write"

    f32_old = {k: v.float() for k, v in state.items()}
    f32_new = {k: v.float() for k, v in new.items()}
    om, nm = mk(cfg, f32_old), mk(new_cfg, f32_new)
    f32_agree = f32_seen = 0
    g32 = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for _ in range(max(1, n_check // 20000)):
            a, b = sample(20000, "cpu", g32)
            f32_agree += int((om(a, b)[:, 1:, :].argmax(-1)
                              == nm(a, b)[:, 1:, :].argmax(-1)).all(-1).sum())
            f32_seen += a.shape[0]
    print(f"[axis] float32 arg max agrees on {f32_agree}/{f32_seen} samples")
    assert f32_agree == f32_seen, "float32 predictions changed"

    n = count(new_cfg)
    assert sum(p.numel() for p in nm.parameters()) == n
    torch.save({"cfg": new_cfg, "members": [f32_new], "acc": [ck["acc"][member]],
                "meta": dict(ck.get("meta", {}), on_axis_from=ckpt_in)},
               ckpt_out)
    print(f"[axis] wrote {ckpt_out} with {n} parameters "
          f"({count(cfg) - n} removed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--n_check", type=int, default=200000)
    a = ap.parse_args()
    fix(a.ckpt, a.out, a.member, a.n_check)
