"""On-GPU sampler for digit-pair addition problems.

A sample is a pair of digit strings of `n` places (least significant first)
plus the exact answer digits.  Operands are always full width: the most
significant digit of each operand is non-zero, matching the graded domain.

Every sampled pair is assigned a hash bucket in 0..15 that depends only on the
pair itself.  Bucket 0 is the held-out split: the trainer masks it out of the
loss, evaluation uses only bucket 0.  So held-out accuracy is measured on
operand pairs the optimiser has provably never received gradient from.
"""

import torch


def hash_bucket(a, b):
    """Deterministic 1-in-16 bucket for each (a, b) digit-pair sample."""
    n = a.shape[-1]
    pw = 10 ** torch.arange(n, device=a.device, dtype=torch.long)
    ia = (a * pw).sum(-1)
    ib = (b * pw).sum(-1)
    h = ia * 2654435761 + ib * 1597334677
    h = h ^ (h >> 29)
    h = h * 1103515245
    h = h ^ (h >> 31)
    h = h * 12345
    h = h ^ (h >> 27)
    return h & 15


def answer_digits(a, b):
    """Exact answer digits (n+1 of them) by integer carry propagation."""
    n = a.shape[-1]
    out = []
    carry = torch.zeros_like(a[..., 0])
    for i in range(n):
        t = a[..., i] + b[..., i] + carry
        out.append(t % 10)
        carry = t // 10
    out.append(carry)
    return torch.stack(out, dim=-1)


def _randint(hi, shape, device, gen, lo=0):
    return torch.randint(lo, hi, shape, device=device, generator=gen)


def sample(batch, n, device, gen, p_uniform=0.35, p_mixed=0.40, p_chain=0.25,
           force_msb=True):
    """Sample a batch of addition problems.

    Three regimes, mixed per sample:
      * uniform            - both digits uniform on 0..9;
      * mixed              - each place is transparent (a+b == 9, i.e. it
                             propagates an incoming carry) with prob 0.4;
      * chain              - each place transparent with prob 0.8, giving long
                             carry chains where the answer at a place depends
                             on a place many positions away.

    Returns digit-pair tokens (batch, n+2) for each operand, the target digits
    (batch, n+2) with position 0 unused, and the hash bucket (batch,).
    """
    tot = p_uniform + p_mixed + p_chain
    r = torch.rand(batch, device=device, generator=gen) * tot
    p_trans = torch.where(
        r < p_uniform,
        torch.zeros((), device=device),
        torch.where(r < p_uniform + p_mixed,
                    torch.full((), 0.4, device=device),
                    torch.full((), 0.8, device=device)),
    )

    transparent = torch.rand(batch, n, device=device, generator=gen) < p_trans[:, None]
    a = _randint(10, (batch, n), device, gen)
    b = _randint(10, (batch, n), device, gen)
    b = torch.where(transparent, 9 - a, b)

    # The most significant place must be non-zero in both operands.  (Only
    # relaxed for the 1-place curriculum warm-up, which is not the real task.)
    top_t = transparent[:, n - 1] & force_msb
    top_a_t = _randint(9, (batch,), device, gen, lo=1)          # 1..8
    top_a_u = _randint(10, (batch,), device, gen, lo=1)         # 1..9
    top_b_u = _randint(10, (batch,), device, gen, lo=1)         # 1..9
    if force_msb:
        a[:, n - 1] = torch.where(top_t, top_a_t, top_a_u)
        b[:, n - 1] = torch.where(top_t, 9 - a[:, n - 1], top_b_u)

    bucket = hash_bucket(a, b)
    ans = answer_digits(a, b)                                    # (batch, n+1)

    pad = torch.zeros(batch, 1, dtype=torch.long, device=device)
    da = torch.cat([pad, a, pad], dim=1)                         # (batch, n+2)
    db = torch.cat([pad, b, pad], dim=1)
    tgt = torch.cat([pad, ans], dim=1)                           # (batch, n+2)
    return da, db, tgt, bucket


def from_ints(a, b, places, device):
    """Tokenise explicit integers (tensors of ints) into the sequence layout."""
    pw = 10 ** torch.arange(places, device=device, dtype=torch.long)
    da = (a[:, None] // pw) % 10
    db = (b[:, None] // pw) % 10
    ans = answer_digits(da, db)
    pad = torch.zeros(a.shape[0], 1, dtype=torch.long, device=device)
    return (torch.cat([pad, da, pad], 1), torch.cat([pad, db, pad], 1),
            torch.cat([pad, ans], 1), hash_bucket(da, db))
