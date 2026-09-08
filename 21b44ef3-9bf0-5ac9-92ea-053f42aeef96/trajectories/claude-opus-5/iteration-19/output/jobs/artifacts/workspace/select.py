"""Rank the members of a checkpoint by how cleanly they implement the mechanism.

Held-out accuracy saturates -- thousands of members score 1.0 on the yardstick -- so it
cannot pick a winner on its own.  What separates them is how much room they leave:

  * `slack`  -- how far the clamp bank is from hard 0/1 over all 100 digit pairs.  A
    member with slack 0 has a key and a value that depend only on the place's carry class,
    which is what makes an exhaustive certificate over the whole domain possible at all.
  * `class`  -- whether the two thresholds cut the 100 pairs into absorb / transparent /
    generate exactly where addition does.
  * `margin` -- the worst distance, in code units, between the answer axis and the wrong
    read-out prototype, over every carry class pattern and every digit pair.

Everything is computed in float64 straight from the stored weights.

    python select.py ckpt/phase2.pt --k 12
"""
import argparse
import itertools

import torch

BANK_W, KEY_W, LAM = 8.0, 400.0, -12.0


def screen(p, n=8):
    """Cheap per-member quantities over all 100 digit pairs.  Returns dicts of (E,)."""
    e = p["code_free"].shape[0]
    code = torch.cat([torch.zeros(e, 1), torch.ones(e, 1), p["code_free"]], 1).double()
    xp = code[:, :, None] + code[:, None, :]                      # (E, 10, 10)
    knee = p["knee"].double()
    u = torch.clamp(BANK_W * (xp[..., None] - knee[:, None, None, :]), 0.0, 1.0)
    slack = torch.minimum(u, 1.0 - u).amax(dim=(1, 2, 3))         # (E,)
    got = (u[..., 0].round() + u[..., 1].round()).long()
    s = torch.arange(10)[:, None] + torch.arange(10)[None, :]
    want = torch.where(s <= 8, 0, torch.where(s == 9, 1, 2))
    cls_ok = (got == want[None]).all(dim=(1, 2))
    step = code[:, 1] - code[:, 0]
    lin = (code - torch.arange(10).double()[None] * step[:, None]).abs().amax(1)
    return dict(slack=slack, cls_ok=cls_ok, lin=lin, code=code,
                carry_w=p["carry_w"].double()[:, 0], fold=p["fold"].double()[:, 0],
                knee=knee)


def domain_margin(sc, i, n=8):
    """Worst read-out margin for member `i` over every class pattern and digit pair, and
    the number of read-outs it gets wrong.  Only valid when the bank is saturated, which
    is what makes key and value depend on the class alone."""
    p_len = n + 2
    code = sc["code"][i]
    xp = code[:, None] + code[None, :]
    knee = sc["knee"][i]
    u = torch.clamp(BANK_W * (xp[..., None] - knee), 0.0, 1.0)
    cls = (u[..., 0].round() + u[..., 1].round()).long()
    key_cls = torch.tensor([KEY_W * float(u[cls == c][0, 1] - u[cls == c][0, 0])
                            for c in range(3)], dtype=torch.float64)
    val_cls = torch.tensor([float(u[cls == c][0, 1]) for c in range(3)],
                           dtype=torch.float64)

    pats = torch.tensor(list(itertools.product(range(3), repeat=n)))
    npat = pats.shape[0]
    seq = torch.zeros(npat, p_len, dtype=torch.long)
    seq[:, 1:n + 1] = pats
    idx = torch.arange(p_len)
    rel = LAM * (idx[:, None] - idx[None, :]).double()
    logit = key_cls[seq].unsqueeze(-2) + rel
    strict = (idx[:, None] > idx[None, :]) | ((idx[:, None] == 0) & (idx[None, :] == 0))
    aoa = idx[:, None] >= idx[None, :]
    neg = torch.tensor(-1e30, dtype=torch.float64)
    v = val_cls[seq]
    cin = (torch.softmax(torch.where(strict, logit, neg), -1) * v.unsqueeze(-2)).sum(-1)
    cout = (torch.softmax(torch.where(aoa, logit, neg), -1) * v.unsqueeze(-2)).sum(-1)

    true_cin = torch.zeros(npat, p_len, dtype=torch.float64)
    c = torch.zeros(npat, dtype=torch.long)
    for k in range(n):
        true_cin[:, k + 1] = c.double()
        c = torch.where(pats[:, k] == 2, 1, torch.where(pats[:, k] == 1, c, 0))
    true_cin[:, n + 1] = c.double()

    off = sc["carry_w"][i] * cin + sc["fold"][i] * cout
    s = torch.arange(10)[:, None] + torch.arange(10)[None, :]
    worst, bad = float("inf"), 0
    for q in range(1, n + 1):
        y = xp[None] + off[:, q][:, None, None]
        want = (s[None] + true_cin[:, q].long()[:, None, None]) % 10
        dist = (y[..., None] - code).abs()
        live = cls[None] == seq[:, q][:, None, None]
        bad += int(((dist.argmin(-1) != want) & live).sum())
        dt = dist.gather(-1, want.unsqueeze(-1)).squeeze(-1)
        do = dist.scatter(-1, want.unsqueeze(-1), 1e30).amin(-1)
        worst = min(worst, float(torch.where(live, do - dt,
                                             torch.full_like(dt, float("inf"))).min()))
    y = xp[0, 0] + off[:, n + 1]
    want = true_cin[:, n + 1].long()
    dist = (y[:, None] - code).abs()
    bad += int((dist.argmin(-1) != want).sum())
    worst = min(worst, float((dist.scatter(-1, want[:, None], 1e30).amin(-1)
                              - dist.gather(-1, want[:, None]).squeeze(-1)).min()))
    return worst, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    sc = screen(ck["params"])
    e = sc["slack"].shape[0]
    ok = sc["cls_ok"] & (sc["slack"] <= a.tol)
    print(f"{a.ckpt}: {e} members")
    print(f"  thresholds cut the 100 pairs correctly : {int(sc['cls_ok'].sum())}")
    print(f"  ... and the clamp bank fully saturated : {int(ok.sum())}")
    if not int(ok.sum()):
        print("  no saturated member; showing the least-slack ones instead")
        ok = sc["slack"] <= sc["slack"].kthvalue(min(e, a.k)).values

    cand = ok.nonzero().flatten().tolist()
    rows = []
    for i in cand[:max(a.k * 8, 64)]:
        m, bad = domain_margin(sc, i)
        rows.append((m, bad, i))
    rows.sort(key=lambda r: (r[1], -r[0]))
    print(f"  full-domain check on {len(rows)} of them (3^8 patterns x 100 pairs x place):")
    for m, bad, i in rows[:a.k]:
        print(f"    [{i:5d}] domain margin {m:+.4f}  wrong read-outs {bad:>7d}  "
              f"slack {float(sc['slack'][i]):.2e}  step {float(sc['code'][i][1]):.4f}  "
              f"carry_w {float(sc['carry_w'][i]):+.4f}  fold {float(sc['fold'][i]):+.4f}")
    best = rows[0]
    print(f"\nbest index: {best[2]}   domain margin {best[0]:+.4f}   wrong {best[1]}")


if __name__ == "__main__":
    main()
