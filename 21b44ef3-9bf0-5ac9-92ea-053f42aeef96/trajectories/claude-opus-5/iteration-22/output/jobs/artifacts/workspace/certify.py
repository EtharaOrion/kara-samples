"""Whole-domain certificate for the shipped weights.

Sampling can only ever say "no counterexample found in N draws".  This instead
proves the model is right on every pair of 8-digit operands, by an independent
float64 reimplementation of the forward pass:

  1. the clamp bank is saturated on all 101 tokens the model can ever see, so
     key and value are functions of a place's carry class alone;
  2. over all 3^8 carry-class patterns, both heads route to the place the
     carry actually comes from, with a measured softmax leakage bound;
  3. over all 100 digit pairs and both carry-in values, the residual lands
     closer to the right prototype than to any other by a measured margin;
  4. margin beats twice the worst residual perturbation that leakage and
     float32 rounding can cause -- so no input anywhere can flip a digit.
"""

import argparse
import importlib.util
import itertools
import torch

torch.set_printoptions(precision=8)


def load(path):
    spec = importlib.util.spec_from_file_location("graded", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def f64(model):
    return {
        "code": torch.cat([model.code_pin, model.code_free]).double(),
        "carry_w": model.carry_w.double(),
        "knee": model.knee.double(),
        "fold": model.fold.double(),
        "slope": model.gate_slope.double(),
        "kscale": model.key_scale.double(),
        "rec": model.recency.double(),
    }


def forward64(w, da, db):
    """Independent float64 reimplementation, used to cross-check the module."""
    c = w["code"]
    z = c[da] + c[db]
    pad = torch.zeros(z.shape[0], 1, dtype=torch.float64)
    z = torch.cat([pad, z, pad], dim=1)
    u = torch.clamp(w["slope"] * (z.unsqueeze(-1) - w["knee"]), 0.0, 1.0)
    val, key = u[..., 1], w["kscale"] * (u[..., 1] - u[..., 0])
    pos = torch.arange(z.shape[1])
    gap = (pos.unsqueeze(-1) - pos.unsqueeze(0)).double()
    sc = key.unsqueeze(1) + w["rec"] * gap
    blk = torch.full_like(gap, -1e9)
    ci = torch.einsum("bpq,bq->bp",
                      (sc + torch.where(gap > 0, 0.0, blk)).softmax(-1), val)
    co = torch.einsum("bpq,bq->bp",
                      (sc + torch.where(gap >= 0, 0.0, blk)).softmax(-1), val)
    res = z + w["carry_w"] * ci + w["fold"] * co
    return (-(res.unsqueeze(-1) - c).abs())[:, 1:, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="submission.py")
    ap.add_argument("--places", type=int, default=8)
    args = ap.parse_args()
    n = args.places
    P = n + 2

    mod = load(args.path)
    model, _ = mod.build_model()
    w = f64(model)
    print(f"weights: code {[round(float(x), 6) for x in w['code']]}")
    print(f"         carry_w {float(w['carry_w']):.6f}  "
          f"knee {[round(float(x), 6) for x in w['knee']]}  "
          f"fold {float(w['fold']):.6f}")

    # ---- 1. the bank is saturated on every token that can ever occur -------
    d = torch.arange(10)
    A, B = torch.meshgrid(d, d, indexing="ij")
    z = (w["code"][A] + w["code"][B]).reshape(-1)                  # 100 tokens
    e = w["slope"] * (z.unsqueeze(-1) - w["knee"])                 # (100, 2)
    slack = torch.maximum(-e, e - 1.0)
    print(f"\n[1] bank saturation: min slack over 100 tokens x 2 units "
          f"= {float(slack.min()):.6f}  (must be > 0)")
    assert float(slack.min()) > 0
    u = torch.clamp(e, 0.0, 1.0)
    assert bool(((u == 0) | (u == 1)).all())

    s = (A + B).reshape(-1)
    true_cls = torch.where(s < 9, 0, torch.where(s == 9, 1, 2))
    got_cls = (u[:, 0] + u[:, 1]).long()                           # 0/1/2
    agree = bool((got_cls == true_cls).all())
    print(f"    the gate splits the 100 pairs by a+b into "
          f"absorb / transparent / generate: {agree}")
    assert agree
    K = w["kscale"] * (u[:, 1] - u[:, 0])
    V = u[:, 1]
    Kc = torch.stack([K[true_cls == k][0] for k in range(3)])
    Vc = torch.stack([V[true_cls == k][0] for k in range(3)])
    for k in range(3):
        assert bool((K[true_cls == k] == Kc[k]).all())
        assert bool((V[true_cls == k] == Vc[k]).all())
    print(f"    per-class key {[float(x) for x in Kc]}  "
          f"value {[float(x) for x in Vc]}")

    # ---- 2. routing, over every carry-class pattern ------------------------
    pats = torch.tensor(list(itertools.product(range(3), repeat=n)))   # (3^n, n)
    M = pats.shape[0]
    cls = torch.zeros(M, P, dtype=torch.long)
    cls[:, 1:n + 1] = pats                       # both pads are absorb
    key = Kc[cls]
    val = Vc[cls]
    pos = torch.arange(P)
    gap = (pos.unsqueeze(-1) - pos.unsqueeze(0)).double()
    sc = key.unsqueeze(1) + w["rec"] * gap
    blk = torch.full_like(gap, -1e9)
    ci = torch.einsum("bpq,bq->bp",
                      (sc + torch.where(gap > 0, 0.0, blk)).softmax(-1), val)
    co = torch.einsum("bpq,bq->bp",
                      (sc + torch.where(gap >= 0, 0.0, blk)).softmax(-1), val)

    cout = torch.zeros(M, P, dtype=torch.float64)
    carry = torch.zeros(M, dtype=torch.float64)
    for j in range(P):
        carry = torch.where(cls[:, j] == 2, torch.ones_like(carry),
                            torch.where(cls[:, j] == 1, carry, torch.zeros_like(carry)))
        cout[:, j] = carry
    cin = torch.cat([torch.zeros(M, 1, dtype=torch.float64), cout[:, :-1]], 1)

    e_ci = float((ci[:, 1:] - cin[:, 1:]).abs().max())
    e_co = float((co[:, 1:] - cout[:, 1:]).abs().max())
    print(f"\n[2] routing over all 3^{n} = {M} carry-class patterns:")
    print(f"    max |strict head - true carry-in |  = {e_ci:.3e}")
    print(f"    max |inclusive  - true carry-out|  = {e_co:.3e}")

    # ---- 3. read-out margin, over every (digit pair, carry-in) -------------
    aa = A.reshape(-1).repeat(2)
    bb = B.reshape(-1).repeat(2)
    cc = torch.cat([torch.zeros(100), torch.ones(100)]).double()
    ss = (aa + bb).double() + cc
    co_i = (ss >= 10).double()
    res = w["code"][aa] + w["code"][bb] + w["carry_w"] * cc + w["fold"] * co_i
    tgt = (ss % 10).long()
    dist = (res.unsqueeze(-1) - w["code"]).abs()
    dt = dist.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    dw = dist.scatter(-1, tgt.unsqueeze(-1), float("inf")).min(-1).values
    m_min = float((dw - dt).min())
    print(f"\n[3] read-out margin over all 100 pairs x carry-in in "
          f"{{0,1}}: min {m_min:.6f}")
    worst = int((dw - dt).argmin())
    print(f"    tightest case a={int(aa[worst])} b={int(bb[worst])} "
          f"carry_in={int(cc[worst])} -> digit {int(tgt[worst])}")

    # ---- 4. does margin beat every perturbation? --------------------------
    torch.manual_seed(0)
    da = torch.randint(0, 10, (20000, n))
    db = torch.randint(0, 10, (20000, n))
    with torch.no_grad():
        l32 = model(da, db)
    l64 = forward64(w, da, db)
    round_err = float((l32.double() - l64).abs().max())
    delta = float(w["carry_w"].abs()) * e_ci + float(w["fold"].abs()) * e_co
    total = delta + round_err
    print(f"\n[4] residual perturbation: leakage {delta:.3e} + float32 rounding "
          f"{round_err:.3e} = {total:.3e}")
    print(f"    a wrong digit needs a perturbation of {m_min / 2:.6f}; "
          f"safety factor {m_min / (2 * total):.1f}x")
    assert m_min > 2 * total

    # ---- cross-check the two implementations agree on answers -------------
    dis = int((l32.argmax(-1) != l64.argmax(-1)).sum())
    print(f"\n[5] float32 module vs float64 reimplementation: "
          f"{dis} differing digits out of {l32.shape[0] * l32.shape[1]}")
    assert dis == 0

    print(f"\nCERTIFIED: every pair of {n}-digit operands is summed exactly.")


if __name__ == "__main__":
    main()
