"""Gauge-fix the factorised embedding, removing `emb_up` for free.

With `emb_rank=r` the digit table is `emb = emb_lo @ emb_up`, and `emb_up`
(r x d) is r*d learned floats.  None of them carry information, because the
block has two exact symmetries:

  * **GL(r) on the factorisation.**  `(emb_lo M, M^-1 emb_up)` gives the same
    table for any invertible r x r matrix M.

  * **O(d) on the residual stream.**  RMSNorm is equivariant under orthogonal
    maps, since ||xQ|| = ||x||, and every other operation in the block is
    either a matrix multiply (absorb Q into the weight) or attention over the
    stream (commutes).  Concretely, sending

        emb -> emb Q,  w_in1 -> Q' w_in1,  w_in2 -> w_in2 Q,  b_in2 -> b_in2 Q,
        w_q -> Q' w_q, w_k -> Q' w_k,  w_v -> Q' w_v,  w_o -> w_o Q,
        w_out1 -> Q' w_out1,  w_out2 -> w_out2 Q,  b_out2 -> b_out2 Q

    leaves every logit unchanged: the tied readout contributes
    rms(x)Q (emb Q)' = rms(x) Q Q' emb' = rms(x) emb'.

Together these act transitively on rank-r up-projections, so *any* trained
`emb_up` can be mapped to the constant injection [I | 0] with no change to the
model's function.  This script performs that map and refuses to write unless
the two models agree on every digit of a large sample -- the same standard
used for the `logit_scale` strip.

The result is a model whose embedding writes into the first r coordinates of
the residual stream and reads back out of them, with r*d fewer parameters.
"""
import argparse

import torch

from cfgutil import count, mk
from data import sample, targets

# Weights that read the residual stream (left-multiplied by Q') and weights
# that write to it (right-multiplied by Q).
_READS = ("w_in1", "w_q", "w_k", "w_v", "w_out1")
_WRITES = ("w_in2", "b_in2", "w_o", "w_out2", "b_out2")


def fix(ckpt_in, ckpt_out, member=0, n_check=200_000, seed=11):
    ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    state = {k: v.detach().clone().double() for k, v in
             ck["members"][member].items()}
    r, d = cfg["emb_rank"], cfg["d_model"]
    assert r, "checkpoint has no factorised embedding"
    assert not cfg.get("emb_fixed_up"), "already gauge-fixed"

    # Q: an orthogonal map sending the row space of emb_up onto the first r
    # coordinates, so emb_up @ Q = [B | 0] with B invertible.
    up = state["emb_up"]
    _, s, vh = torch.linalg.svd(up, full_matrices=True)
    assert s[r - 1] > 1e-8 * s[0], f"emb_up is rank-deficient: {s.tolist()}"
    Q = vh.t()
    upQ = up @ Q
    tail = upQ[:, r:].abs().max().item()
    assert tail < 1e-9, f"row space not isolated (residual {tail:g})"
    B = upQ[:, :r]

    new = {k: v.clone() for k, v in state.items()}
    new.pop("emb_up")
    new["emb_lo"] = state["emb_lo"] @ B          # absorbs M = B^-1
    for k in _READS:
        if k in new:
            new[k] = Q.t() @ state[k]
    for k in _WRITES:
        if k in new:
            new[k] = state[k] @ Q

    f32_state = {k: v.float() for k, v in state.items()}
    f32_new = {k: v.float() for k, v in new.items()}
    old_m = mk(cfg, f32_state)
    new_cfg = dict(cfg, emb_fixed_up=True)
    new_m = mk(new_cfg, f32_new)

    # The new table is the old one rotated by Q, which is exactly what the
    # rest of the network has been rotated to expect.
    tbl_err = ((old_m._emb().double() @ Q.float().double())
               - new_m._emb().double()).abs().max().item()
    print(f"[gauge] singular values of emb_up: "
          f"{[round(float(x), 4) for x in s]}")
    print(f"[gauge] digit table reproduced (up to Q) to {tbl_err:.2e}")
    assert tbl_err < 1e-5, "gauge map does not reproduce the digit table"

    # The gauge leaves every logit invariant, not merely the arg max, so the
    # sharp test is on the logits themselves -- run in double precision, since
    # in float32 the two models take genuinely different arithmetic paths
    # through the same function and disagree at the usual 1e-7 relative level.
    old_d, new_d = old_m.double(), new_m.double()
    old_d.load_state_dict(state)
    new_d.load_state_dict(new)

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
    print(f"[gauge] max |logit| deviation over {seen} samples (float64): "
          f"{dev:.3e} ({rel:.2e} relative to a peak logit of {mag:.4g})")
    print(f"[gauge] identical predictions on {agree}/{seen} samples")
    assert rel < 1e-10, "logits changed -- refusing to write"
    print(f"[gauge] exact-match accuracy of gauge-fixed model: "
          f"{correct / seen:.6f}")
    assert agree == seen, "predictions changed -- refusing to write"

    # ...and the float32 model that actually ships must still agree exactly on
    # the arg max, which is all `add` consumes.
    f32_agree = f32_seen = 0
    g32 = torch.Generator().manual_seed(seed + 1)
    om, nm = mk(cfg, f32_state), mk(new_cfg, f32_new)
    with torch.no_grad():
        for _ in range(max(1, n_check // 20000)):
            a, b = sample(20000, "cpu", g32)
            po = om(a, b)[:, 1:, :].argmax(-1)
            pn = nm(a, b)[:, 1:, :].argmax(-1)
            f32_agree += int((po == pn).all(-1).sum())
            f32_seen += a.shape[0]
    print(f"[gauge] float32 arg max agrees on {f32_agree}/{f32_seen} samples")
    assert f32_agree == f32_seen, "float32 predictions changed"

    n = count(new_cfg)
    assert sum(p.numel() for p in nm.parameters()) == n
    torch.save({"cfg": new_cfg,
                "members": [f32_new],
                "acc": [ck["acc"][member]],
                "meta": dict(ck.get("meta", {}), gauge_fixed_from=ckpt_in)},
               ckpt_out)
    print(f"[gauge] wrote {ckpt_out} with {n} parameters "
          f"({count(cfg) - n} removed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--n_check", type=int, default=200000)
    a = ap.parse_args()
    fix(a.ckpt, a.out, a.member, a.n_check)
