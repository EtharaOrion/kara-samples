"""Training data generation (lives outside submission.py).

Operand pairs are produced as digit tensors, least-significant digit first, in
the slot layout the model consumes: one leading sentinel slot, then `ndig`
digit slots, then one slot for the final carry.  Because the model carries no
absolute position information, training can use short sequences (small `ndig`)
and evaluate on the full 14-digit task.
"""

import torch

MAXPOW = 14  # operands have at most 14 digits


def _labels(da, db):
    """Exact sum digits for digit tensors da, db of shape (B, S)."""
    s = da + db
    out = torch.empty_like(s)
    carry = torch.zeros_like(s[:, 0])
    for i in range(s.shape[1]):
        t = s[:, i] + carry
        out[:, i] = t % 10
        carry = t // 10
    return out


def _pad(d):
    """(B, ndig) digits -> (B, ndig + 2) slots: sentinel + digits + carry slot."""
    z = d.new_zeros(d.shape[0], 1)
    return torch.cat([z, d, z], dim=1)


def sample_digits(B, device, gen=None, ndig=MAXPOW):
    """Mixture of four regimes, returned as (da, db) of shape (B, ndig + 2)."""
    q = B // 4
    parts_a, parts_b = [], []

    def rnd(*shape):
        return torch.randint(0, 10, shape, device=device, generator=gen)

    # 1. uniform full-length operands
    parts_a.append(rnd(q, ndig))
    parts_b.append(rnd(q, ndig))

    # 2. independent random operand lengths
    a, b = rnd(q, ndig), rnd(q, ndig)
    pos = torch.arange(ndig, device=device)
    la = torch.randint(1, ndig + 1, (q, 1), device=device, generator=gen)
    lb = torch.randint(1, ndig + 1, (q, 1), device=device, generator=gen)
    parts_a.append(a * (pos < la))
    parts_b.append(b * (pos < lb))

    # 3. digits skewed towards 0 and 9 (carry-chain rich)
    d = rnd(2 * q, ndig)
    snap = torch.rand(2 * q, ndig, device=device, generator=gen)
    d = torch.where(snap < 0.2, torch.zeros_like(d), d)
    d = torch.where(snap > 0.8, torch.full_like(d, 9), d)
    parts_a.append(d[:q])
    parts_b.append(d[q:])

    # 4. forced propagate chains: b_i = 9 - a_i at a per-sample rate
    rest = B - 3 * q
    a = rnd(rest, ndig)
    b = rnd(rest, ndig)
    rate = 0.3 + 0.7 * torch.rand(rest, 1, device=device, generator=gen)
    m = torch.rand(rest, ndig, device=device, generator=gen) < rate
    b = torch.where(m, 9 - a, b)
    parts_a.append(a)
    parts_b.append(b)

    return _pad(torch.cat(parts_a, 0)), _pad(torch.cat(parts_b, 0))


def sample_uniform(B, device, gen=None, ndig=MAXPOW):
    d = torch.randint(0, 10, (2 * B, ndig), device=device, generator=gen)
    return _pad(d[:B]), _pad(d[B:])


def make_batch(B, device, uniform=False, gen=None, maxlen=MAXPOW):
    sampler = sample_uniform if uniform else sample_digits
    da, db = sampler(B, device, gen, maxlen)
    return da, db, _labels(da, db)
