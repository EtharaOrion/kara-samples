"""On-GPU sampler for 8-digit (and other width) addition, with a held-out split.

All tensors are LSB-first digit arrays of shape (B, n).  Both operands are forced
to carry the full digit width (MSB >= 1), matching the graded distribution.

Held-out split: pairs are bucketed by a hash of (a, b); bucket 0 of 16 is never
trained on and is used for the held-out accuracy figures.
"""
import torch

HOLD_MOD = 16          # 1-in-16 of all operand pairs is held out


def pair_key(da, db):
    """(B,n) digit arrays -> (B,) int64 key a*10^n + b."""
    n = da.shape[1]
    pw = (10 ** torch.arange(n, device=da.device, dtype=torch.int64))
    a = (da * pw).sum(1)
    b = (db * pw).sum(1)
    return a * (10 ** n) + b


def is_heldout(da, db):
    k = pair_key(da, db)
    h = (k * 2654435761 + (k >> 17) * 40503) % 1000003
    return (h % HOLD_MOD) == 0


def _digits_for_class(cls, B, n, device, gen):
    """cls in {0: absorb (s<=8), 1: transparent (s==9), 2: generate (s>=10)}."""
    lo = torch.where(cls == 0, 0, torch.where(cls == 1, 9, 10))
    hi = torch.where(cls == 0, 8, torch.where(cls == 1, 9, 18))
    r = torch.rand(B, n, device=device, generator=gen)
    s = lo + (r * (hi - lo + 1).float()).long().clamp(max=(hi - lo))
    alo = (s - 9).clamp(min=0)
    ahi = s.clamp(max=9)
    r2 = torch.rand(B, n, device=device, generator=gen)
    a = alo + (r2 * (ahi - alo + 1).float()).long().clamp(max=(ahi - alo))
    b = s - a
    return a, b


def sample_raw(B, n, regime, device, gen):
    if regime == "nocarry":
        # every place sums to at most 9, so the answer is the digit-wise sum and no
        # carry ever occurs.  Curriculum stage A: this forces code[a]+code[b] to read
        # out as a+b for every a+b <= 9, i.e. it forces the code to be a ramp, with
        # no mod-10 wrap available to hide in.
        a = torch.randint(0, 10, (B, n), device=device, generator=gen)
        r = torch.rand(B, n, device=device, generator=gen)
        b = (r * (10 - a).float()).long()
        return a, torch.minimum(b, 9 - a)
    if regime == "uniform":
        da = torch.randint(0, 10, (B, n), device=device, generator=gen)
        db = torch.randint(0, 10, (B, n), device=device, generator=gen)
        return da, db
    if regime == "transparent":
        da = torch.randint(0, 10, (B, n), device=device, generator=gen)
        db = torch.randint(0, 10, (B, n), device=device, generator=gen)
        m = torch.rand(B, n, device=device, generator=gen) < 0.4
        db = torch.where(m, 9 - da, db)
        return da, db
    if regime == "chain":
        # long carry chains: mostly transparent/generate places
        r = torch.rand(B, n, device=device, generator=gen)
        cls = torch.where(r < 0.20, 0, torch.where(r < 0.65, 1, 2))
        return _digits_for_class(cls, B, n, device, gen)
    raise ValueError(regime)


def _force_full_width(da, db, gen):
    B = da.shape[0]
    dev = da.device
    top = torch.randint(1, 10, (B,), device=dev, generator=gen)
    da[:, -1] = torch.where(da[:, -1] == 0, top, da[:, -1])
    top = torch.randint(1, 10, (B,), device=dev, generator=gen)
    db[:, -1] = torch.where(db[:, -1] == 0, top, db[:, -1])
    return da, db


def sample(B, n, device, gen, mix=(0.35, 0.40, 0.25), heldout=False, tries=12,
           regimes=("uniform", "transparent", "chain"), full_width=True):
    """Mixture sampler.  heldout=False -> only training pairs, True -> only held-out."""
    da = torch.zeros(B, n, dtype=torch.int64, device=device)
    db = torch.zeros(B, n, dtype=torch.int64, device=device)
    need = torch.ones(B, dtype=torch.bool, device=device)
    over = HOLD_MOD + 4 if heldout else 1        # held-out pairs are 1-in-HOLD_MOD
    for _ in range(tries):
        m0 = int(need.sum())
        if m0 == 0:
            break
        m = m0 * over
        r = torch.rand(m, device=device, generator=gen)
        n1 = int((r < mix[0]).sum())
        n2 = int(((r >= mix[0]) & (r < mix[0] + mix[1])).sum())
        n3 = m - n1 - n2
        parts_a, parts_b = [], []
        for cnt, reg in zip((n1, n2, n3), regimes):
            if cnt:
                a, b = sample_raw(cnt, n, reg, device, gen)
                parts_a.append(a)
                parts_b.append(b)
        ca = torch.cat(parts_a, 0)
        cb = torch.cat(parts_b, 0)
        if full_width:
            ca, cb = _force_full_width(ca, cb, gen)
        ok = is_heldout(ca, cb) if heldout else ~is_heldout(ca, cb)
        sel = ok.nonzero(as_tuple=True)[0]
        idx = need.nonzero(as_tuple=True)[0]
        take = min(len(sel), len(idx))
        if take:
            da[idx[:take]] = ca[sel[:take]]
            db[idx[:take]] = cb[sel[:take]]
            need = need.clone()
            need[idx[:take]] = False
    return da, db


def to_tokens(da, db):
    """(B,n) digits -> (B,n+2) token digit arrays with (0,0) pads at both ends."""
    B, n = da.shape
    z = torch.zeros(B, 1, dtype=da.dtype, device=da.device)
    return torch.cat([z, da, z], 1), torch.cat([z, db, z], 1)


def targets(da, db):
    """(B,n) digits -> (B,n+2) target digits: [0, ans_0..ans_n]."""
    B, n = da.shape
    s = da + db
    carry = torch.zeros(B, dtype=da.dtype, device=da.device)
    outs = []
    for i in range(n):
        t = s[:, i] + carry
        outs.append(t % 10)
        carry = torch.div(t, 10, rounding_mode="floor")
    outs.append(carry)
    z = torch.zeros(B, 1, dtype=da.dtype, device=da.device)
    return torch.cat([z] + [o[:, None] for o in outs], 1)


def batch(B, n, device, gen, heldout=False, mix=(0.35, 0.40, 0.25), **kw):
    da, db = sample(B, n, device, gen, mix=mix, heldout=heldout, **kw)
    ta, tb = to_tokens(da, db)
    return ta, tb, targets(da, db), da, db
