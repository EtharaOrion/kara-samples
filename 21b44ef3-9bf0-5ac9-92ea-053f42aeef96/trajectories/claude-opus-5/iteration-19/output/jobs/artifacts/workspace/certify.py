"""Exhaustive certificate for /workspace/submission.py over the whole 8-digit domain.

Checking 10^16 operand pairs one at a time is impossible, but the block's behaviour
factorises, so the whole domain can be covered exactly:

  * the residual value at a place depends only on that place's digit pair -- 100 cases;
  * the clamp bank is saturated, so the key and the value at a place depend only on which
    of the three carry classes {absorb, transparent, generate} the place falls in;
  * therefore the attention -- and with it the carry into and out of every place -- depends
    only on the *pattern* of classes across the eight places, of which there are 3^8;
  * and the read-out at a place depends only on that place's digit pair and its carry.

So enumerating 3^8 class patterns x 100 digit pairs x every place covers every one of the
8.1 x 10^15 in-range operand pairs, with nothing sampled.  Saturation is checked rather
than assumed: if any clamp came out part-way the factorisation would not hold and the
script says so instead of certifying.

Everything here runs in float64 off the shipped module's own parameters.  The replicated
block is cross-checked against `model(tok)` before it is trusted.

    python certify.py
"""
import argparse
import importlib.util
import itertools
import os
import sys

import torch

WORK = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(WORK, "submission.py")
TOL = 1e-6


def load(path):
    spec = importlib.util.spec_from_file_location("graded_submission", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["graded_submission"] = m
    spec.loader.exec_module(m)
    return m


def attention(key, lam, n_pos):
    """The two masked maps the block builds, in float64.  key: (..., P)."""
    idx = torch.arange(n_pos)
    rel = lam * (idx[:, None] - idx[None, :]).double()
    logit = key.unsqueeze(-2) + rel
    after = idx[:, None] > idx[None, :]
    aoa = idx[:, None] >= idx[None, :]
    strict = after | ((idx[:, None] == 0) & (idx[None, :] == 0))
    a_s = torch.softmax(torch.where(strict, logit, torch.tensor(-1e30, dtype=torch.float64)), -1)
    a_i = torch.softmax(torch.where(aoa, logit, torch.tensor(-1e30, dtype=torch.float64)), -1)
    return a_s, a_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=SUB)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    n = args.n
    p_len = n + 2

    mod = load(args.path)
    model, _ = mod.build_model()
    with torch.no_grad():
        code = model.codes().double()
        carry_w = model.carry_w.double().reshape(()).clone()
        fold = model.fold.double().reshape(()).clone()
        knee = model.knee.double().clone()
        bank_w = model.bank_w.double().clone()
        key_w = model.key_w.double().clone()
        lam = model.lam.double().clone()

    fails = []
    print("code   ", [round(float(c), 6) for c in code])
    print("carry_w", round(float(carry_w), 6), " fold", round(float(fold), 6),
          " knee", [round(float(k), 6) for k in knee])

    # ---------------------------------------------------------- 1. per-place quantities
    d = torch.arange(10)
    xp = code[:, None] + code[None, :]                       # (10, 10) residual value
    u = torch.clamp(bank_w * (xp[..., None] - knee), 0.0, 1.0)
    slack = torch.minimum(u, 1.0 - u).max()
    print(f"\nclamp saturation: worst distance from a hard 0/1 is {float(slack):.3e}")
    if float(slack) > TOL:
        fails.append(f"clamp bank not saturated (slack {float(slack):.3e})")

    key_pair = key_w * (u[..., 1] - u[..., 0])
    val_pair = u[..., 1]
    got_cls = (u[..., 0].round() + u[..., 1].round()).long()  # 0 absorb 1 transp 2 gen
    s = d[:, None] + d[None, :]
    want_cls = torch.where(s <= 8, 0, torch.where(s == 9, 1, 2))
    bad_cls = (got_cls != want_cls).sum().item()
    print(f"carry classes: {100 - bad_cls}/100 digit pairs put in the right class "
          f"(absorb / transparent / generate)")
    if bad_cls:
        ii, jj = (got_cls != want_cls).nonzero(as_tuple=True)
        fails.append(f"{bad_cls} digit pairs mis-classified, e.g. "
                     f"({int(ii[0])},{int(jj[0])})")

    # with the bank saturated the key and value are constant within a class
    key_cls = torch.zeros(3, dtype=torch.float64)
    val_cls = torch.zeros(3, dtype=torch.float64)
    for c in range(3):
        m = got_cls == c
        key_cls[c], val_cls[c] = key_pair[m][0], val_pair[m][0]
        spread = max(float((key_pair[m] - key_cls[c]).abs().max()),
                     float((val_pair[m] - val_cls[c]).abs().max()))
        if spread > TOL:
            fails.append(f"class {c} key/value not constant (spread {spread:.3e})")
    print(f"per-class key {[round(float(k), 3) for k in key_cls]}  "
          f"value {[round(float(v), 3) for v in val_cls]}")

    # ---------------------------------------------------------- 2. every class pattern
    pats = torch.tensor(list(itertools.product(range(3), repeat=n)))    # (3^n, n)
    npat = pats.shape[0]
    cls_seq = torch.zeros(npat, p_len, dtype=torch.long)                # pads are absorb
    cls_seq[:, 1:n + 1] = pats
    a_s, a_i = attention(key_cls[cls_seq], lam, p_len)
    v = val_cls[cls_seq]
    cin = (a_s * v.unsqueeze(-2)).sum(-1)                               # (3^n, P)
    cout = (a_i * v.unsqueeze(-2)).sum(-1)

    # the carries the task actually calls for, rippled straight off the class pattern
    true_cin = torch.zeros(npat, p_len, dtype=torch.float64)
    c = torch.zeros(npat, dtype=torch.long)
    for k in range(n):
        true_cin[:, k + 1] = c.double()
        c = torch.where(pats[:, k] == 2, torch.ones_like(c),
                        torch.where(pats[:, k] == 1, c, torch.zeros_like(c)))
    true_cin[:, n + 1] = c.double()
    err_in = (cin - true_cin)[:, 1:].abs().max()
    true_cout = torch.zeros_like(true_cin)
    true_cout[:, 1:n + 1] = torch.where(
        pats == 2, 1.0, torch.where(pats == 1, true_cin[:, 1:n + 1], 0.0))
    err_out = (cout - true_cout)[:, 1:].abs().max()
    print(f"\nattention over all 3^{n} = {npat} class patterns:")
    print(f"  carry-in  worst error vs the true carry: {float(err_in):.3e}")
    print(f"  carry-out worst error vs the true carry: {float(err_out):.3e}")
    if float(err_in) > 1e-3 or float(err_out) > 1e-3:
        fails.append(f"attention resolves carries to only {max(float(err_in), float(err_out)):.3e}")

    # ---------------------------------------------------------- 3. every read-out
    off = carry_w * cin + fold * cout                                   # (3^n, P)
    worst = float("inf")
    n_bad = 0
    for i in range(1, n + 1):
        y = xp[None, :, :] + off[:, i][:, None, None]                   # (3^n, 10, 10)
        want = (s[None] + true_cin[:, i].long()[:, None, None]) % 10
        dist = (y[..., None] - code).abs()                              # (3^n,10,10,10)
        pred = dist.argmin(-1)
        live = got_cls[None] == cls_seq[:, i][:, None, None]            # class must match
        n_bad += int(((pred != want) & live).sum())
        dt = dist.gather(-1, want.unsqueeze(-1)).squeeze(-1)
        do = dist.scatter(-1, want.unsqueeze(-1), 1e30).amin(-1)
        gm = torch.where(live, do - dt, torch.full_like(dt, float("inf")))
        worst = min(worst, float(gm.min()))
    # the carry slot: the pad digit pair, offset by whatever reached it
    y = xp[0, 0] + off[:, n + 1]
    want = true_cin[:, n + 1].long()
    dist = (y[:, None] - code).abs()
    n_bad += int((dist.argmin(-1) != want).sum())
    gm = (dist.scatter(-1, want[:, None], 1e30).amin(-1)
          - dist.gather(-1, want[:, None]).squeeze(-1))
    worst = min(worst, float(gm.min()))

    total = npat * (n * 100 + 1)
    print(f"\nread-out over every (class pattern, place, digit pair): "
          f"{total - n_bad}/{total} correct")
    print(f"  worst read-out margin anywhere in the domain: {worst:.4f} code units "
          f"(a full step is {float(code[1] - code[0]):.4f})")
    if n_bad:
        fails.append(f"{n_bad} read-outs wrong")

    # ---------------------------------------------------------- 4. trust the replica
    g = torch.Generator().manual_seed(11)
    tok = torch.randint(0, 10, (256, p_len, 2), generator=g)
    tok[:, 0] = 0
    tok[:, n + 1] = 0
    with torch.no_grad():
        ref = model(tok).double()
    kk = key_pair[tok[..., 0], tok[..., 1]]
    vv = val_pair[tok[..., 0], tok[..., 1]]
    ra, rb = attention(kk, lam, p_len)
    yy = (xp[tok[..., 0], tok[..., 1]]
          + carry_w * (ra * vv.unsqueeze(-2)).sum(-1)
          + fold * (rb * vv.unsqueeze(-2)).sum(-1))
    rep = -(yy.unsqueeze(-1) - code) ** 2
    dd = float((rep - ref).abs().max())
    print(f"\nreplica vs model(tok) on 256 sequences: max logit difference {dd:.3e}")
    if dd > 1e-3:
        fails.append(f"replica disagrees with the module ({dd:.3e})")

    print("\n" + "=" * 72)
    if fails:
        print("NOT CERTIFIED:")
        for f in fails:
            print("  -", f)
        return 1
    print(f"CERTIFIED: exact on all {10 ** n} x {10 ** n} operand pairs of width {n}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
