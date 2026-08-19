"""Training-data generator (operand sampling + ground-truth sum digits).

Lives outside submission.py by design: this is labelled-data generation.
"""

import torch

N_POS = 15


def sum_digits(a_d, b_d):
    """Ground-truth digits of a+b, least-significant first. (B, N_POS)"""
    s = a_d + b_d
    carry = torch.zeros_like(s[:, 0])
    out = []
    for i in range(s.shape[1]):
        t = s[:, i] + carry
        out.append(t % 10)
        carry = torch.div(t, 10, rounding_mode='floor')
    return torch.stack(out, 1)


def _skewed_digits(bs, n, device, g):
    """Digits biased towards 9 and 0 (the values that drive carry behaviour)."""
    u = torch.rand(bs, n, device=device, generator=g)
    r = torch.randint(0, 10, (bs, n), device=device, generator=g)
    nine = torch.full_like(r, 9)
    zero = torch.zeros_like(r)
    d = torch.where(u < 0.35, nine, torch.where(u < 0.55, zero, r))
    return d


def gen_batch(bs, device, g=None, n_digits=14):
    """Sample operand digit arrays covering uniform, short, skewed and
    carry-chain-heavy regimes.  Returns (a_d, b_d, y) each (bs, N_POS)."""
    n = n_digits
    mode = torch.randint(0, 4, (bs, 1), device=device, generator=g)

    a = torch.randint(0, 10, (bs, n), device=device, generator=g)
    b = torch.randint(0, 10, (bs, n), device=device, generator=g)

    pos = torch.arange(n, device=device)[None, :]

    # mode 1: independent random operand lengths (covers small operands)
    la = torch.randint(1, n + 1, (bs, 1), device=device, generator=g)
    lb = torch.randint(1, n + 1, (bs, 1), device=device, generator=g)
    a1 = a * (pos < la)
    b1 = b * (pos < lb)

    # mode 2: digits skewed towards 9/0
    a2 = _skewed_digits(bs, n, device, g)
    b2 = _skewed_digits(bs, n, device, g)

    # mode 3: long propagate chains -- b_i = 9 - a_i at a random rate per sample
    p = torch.rand(bs, 1, device=device, generator=g) * 0.7 + 0.3
    prop = torch.rand(bs, n, device=device, generator=g) < p
    b3 = torch.where(prop, 9 - a, b)
    a3 = a

    a = torch.where(mode == 1, a1, torch.where(mode == 2, a2, torch.where(mode == 3, a3, a)))
    b = torch.where(mode == 1, b1, torch.where(mode == 2, b2, torch.where(mode == 3, b3, b)))

    pad = torch.zeros(bs, N_POS - n, dtype=a.dtype, device=device)
    a = torch.cat([a, pad], 1)
    b = torch.cat([b, pad], 1)
    return a, b, sum_digits(a, b)


def gen_uniform(bs, device, g=None, n_digits=14):
    """Plain uniform operands in [0, 10^14) -- the evaluation-style regime."""
    a = torch.randint(0, 10, (bs, n_digits), device=device, generator=g)
    b = torch.randint(0, 10, (bs, n_digits), device=device, generator=g)
    pad = torch.zeros(bs, N_POS - n_digits, dtype=a.dtype, device=device)
    a = torch.cat([a, pad], 1)
    b = torch.cat([b, pad], 1)
    return a, b, sum_digits(a, b)
