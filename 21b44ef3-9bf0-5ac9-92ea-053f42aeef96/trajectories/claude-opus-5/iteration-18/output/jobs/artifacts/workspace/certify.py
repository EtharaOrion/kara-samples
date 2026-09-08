"""Whole-domain certificate for one shipped model.

Proves, in float64 and without sampling, that the shipped weights compute
a + b exactly for every input of the graded width (and for any width up to the
length the bound is evaluated at).  Three parts:

  1. gate.  For each of the 100 digit pairs, compute the exact key and value.
     The value must be exactly 0 on every pair with a+b <= 8 and exactly 1 on
     every pair with a+b >= 10 (so the attended value *is* the carry), and the
     key must be strictly lower on every pair with a+b == 9 than on any other
     pair (so transparent places are unattendable).
  2. attention leakage.  Given (1) the attention logits are constants that
     depend only on the class of each place, so the softmax mass escaping the
     intended position is bounded analytically for every sequence length.
  3. read-out margin.  For every digit pair and every carry-in, the exact
     residual must be closer to the correct code than to any other by more
     than twice the residual perturbation that (2) allows.

Also runs an independent exhaustive check over all 3^n carry-class patterns.
"""
import itertools
import math
import torch


def gate(code, knee, bank_w, key_w, val_w):
    """-> x, key, val over the 10x10 digit pairs (float64)."""
    x = code[:, None] + code[None, :]
    u = torch.clamp(bank_w * (x[..., None] - knee), 0.0, 1.0)
    return x, u @ key_w, u @ val_w


def certify(code, knee, fold, bank_w=(8.0, 8.0), key_w=(-400.0, 400.0),
            val_w=(0.0, 1.0), lam=-12.0, carry_w=1.0, Pmax=12, verbose=True):
    t = lambda v: torch.as_tensor(v, dtype=torch.float64)
    code, knee = t(code).cpu(), t(knee).cpu()
    bank_w, key_w, val_w = t(bank_w), t(key_w), t(val_w)
    fold, lam, carry_w = float(fold), float(lam), float(carry_w)
    rep, ok = {}, True

    # ---- 1. gate -------------------------------------------------------------
    x, key, val = gate(code, knee, bank_w, key_w, val_w)
    d = torch.arange(10)
    s = d[:, None] + d[None, :]
    absorb, transp, gener = s <= 8, s == 9, s >= 10
    nt = ~transp

    rep['val_absorb_exact_zero'] = bool((val[absorb] == 0).all())
    rep['val_generate_exact_one'] = bool((val[gener] == 1).all())
    ok &= rep['val_absorb_exact_zero'] and rep['val_generate_exact_one']
    k_nt_max, k_nt_min = float(key[nt].max()), float(key[nt].min())
    k_t_max = float(key[transp].max())
    rep['key_nontransparent'] = (k_nt_min, k_nt_max)
    rep['key_transparent_max'] = k_t_max
    rep['notch_depth'] = k_nt_min - k_t_max
    ok &= rep['notch_depth'] > 0
    rep['val_spread'] = float(val.max() - val.min())
    rep['x_absorb_max'] = float(x[absorb].max())
    rep['x_transp_min'] = float(x[transp].min())
    rep['x_transp_max'] = float(x[transp].max())
    rep['x_gen_min'] = float(x[gener].min())

    # ---- 2. attention leakage ------------------------------------------------
    # The intended target of a query at position p is the nearest place q* with
    # q* <= p (inclusive head) or q* < p (strict head) that is not transparent.
    # Its logit is at least k_nt_min - |lam|*d*.  Every other allowed place is
    # either non-transparent and strictly further away (logit at most
    # k_nt_max - |lam|*(d*+1)) or transparent (logit at most k_t_max, attained
    # at distance 0).
    al = abs(lam)
    Ls = []
    for P in range(2, Pmax + 1):
        worst = 0.0
        for dstar in range(0, P):
            far = (P - 1) * math.exp(k_nt_max - al * (dstar + 1) - (k_nt_min - al * dstar))
            near = (P - 1) * math.exp(k_t_max - (k_nt_min - al * dstar))
            worst = max(worst, far + near)
        Ls.append(worst)
    L = max(Ls)
    rep['max_leak_weight'] = L
    drift = (L / (1.0 + L)) * rep['val_spread'] * (abs(carry_w) + abs(fold))
    rep['max_residual_drift'] = drift

    # ---- 3. read-out margin --------------------------------------------------
    worst, wcase = float('inf'), None
    for a in range(10):
        for b in range(10):
            for cin in (0, 1):
                tot = a + b + cin
                r = code[a] + code[b] + carry_w * cin + fold * (1 if tot >= 10 else 0)
                dist = (r - code).abs()
                tg = tot % 10
                mg = float(torch.cat([dist[:tg], dist[tg + 1:]]).min()) - float(dist[tg])
                if mg < worst:
                    worst, wcase = mg, (a, b, cin)
    rep['worst_readout_margin'] = worst
    rep['worst_case'] = wcase
    rep['safety_factor'] = worst / (2 * drift) if drift > 0 else float('inf')
    ok &= worst > 2 * drift

    rep['code'] = [float(c) for c in code]
    rep['code_steps'] = [round(float(v), 5) for v in (code[1:] - code[:-1])]
    rep['knee'] = [float(v) for v in knee]
    rep['fold'] = fold
    rep['fold_over_mean_step'] = fold / (float(code[9] - code[0]) / 9.0)
    rep['certified'] = bool(ok)
    if verbose:
        for k, v in rep.items():
            print(f'  {k}: {v}')
    return ok, rep


def exhaustive_patterns(model, n=8, device='cpu'):
    """Run every one of the 3^n absorb/transparent/generate patterns through the
    real (float32) model, using a representative digit pair for each class."""
    reps = {0: (3, 4), 1: (4, 5), 2: (6, 7)}       # a+b = 7, 9, 13
    pats = list(itertools.product([0, 1, 2], repeat=n))
    da = torch.zeros(len(pats), n + 2, dtype=torch.long)
    db = torch.zeros(len(pats), n + 2, dtype=torch.long)
    for i, pat in enumerate(pats):
        for j, c in enumerate(pat):
            da[i, j + 1], db[i, j + 1] = reps[c]
    a_val = (da[:, 1:n + 1] * (10 ** torch.arange(n))).sum(1)
    b_val = (db[:, 1:n + 1] * (10 ** torch.arange(n))).sum(1)
    tgt = a_val + b_val
    with torch.no_grad():
        pred = model(da.to(device), db.to(device)).argmax(-1)[:, 1:].cpu()
    got = (pred * (10 ** torch.arange(n + 1))).sum(1)
    return int((got == tgt).sum()), len(pats)
