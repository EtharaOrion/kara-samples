"""Operand sampling and the train/held-out split.

Every operand is a full-width 8-digit number in [10_000_000, 99_999_999].
A deterministic hash of the pair carves out HOLDOUT_BUCKETS^-1 of the pair space
as a held-out set that training never touches, so reported accuracy is measured
on pairs the model provably never saw.
"""

import torch

NDIG = 8
HOLDOUT_BUCKETS = 128  # bucket 0 (~0.78% of all pairs) is held out


def pair_bucket(a_val, b_val):
    """Deterministic bucket in [0, HOLDOUT_BUCKETS) for an operand pair."""
    h = (a_val * 1000003 + b_val * 1000033 + (a_val ^ b_val) * 7919) % 1000000007
    return (h * 2654435761) % HOLDOUT_BUCKETS


def digits_to_value(d):
    """(B, 8) digit tensor, LSB first -> (B,) int64 values."""
    place = torch.tensor([10 ** i for i in range(NDIG)], dtype=torch.int64, device=d.device)
    return (d.to(torch.int64) * place).sum(-1)


def sample_digits(n, prop_p, device, gen=None):
    """Sample digit pairs.  ``prop_p`` is the per-place probability of forcing
    a_j + b_j == 9, which is what makes a carry propagate through place j.
    Oversampling those places is the whole curriculum: uniform operands almost
    never contain a long carry chain."""
    a = torch.randint(0, 10, (n, NDIG), device=device, generator=gen)
    b = torch.randint(0, 10, (n, NDIG), device=device, generator=gen)
    if prop_p > 0:
        prop = torch.rand((n, NDIG), device=device, generator=gen) < prop_p
        b = torch.where(prop, 9 - a, b)
    else:
        prop = torch.zeros((n, NDIG), dtype=torch.bool, device=device)
    # Leading place must stay non-zero for both operands (full 8-digit width).
    pm = prop[:, NDIG - 1]
    a_hi_p = torch.randint(1, 9, (n,), device=device, generator=gen)   # 1..8
    a_hi_u = torch.randint(1, 10, (n,), device=device, generator=gen)  # 1..9
    b_hi_u = torch.randint(1, 10, (n,), device=device, generator=gen)
    a_hi = torch.where(pm, a_hi_p, a_hi_u)
    b_hi = torch.where(pm, 9 - a_hi, b_hi_u)
    a[:, NDIG - 1] = a_hi
    b[:, NDIG - 1] = b_hi
    return a, b


def chain_digits(n, device, gen=None):
    """Operand pairs whose carry must cross a maximal run of transparent places:
    one place generates a carry and every place above it sums to exactly 9.
    Uniform sampling essentially never produces these, so they are supplied
    directly -- they are ordinary (a, b) pairs, labelled by the true sum like
    every other sample."""
    a = torch.randint(0, 10, (n, NDIG), device=device, generator=gen)
    b = torch.randint(0, 10, (n, NDIG), device=device, generator=gen)
    start = torch.randint(0, NDIG, (n,), device=device, generator=gen)
    place = torch.arange(NDIG, device=device).unsqueeze(0)
    above = place > start.unsqueeze(1)
    at = place == start.unsqueeze(1)
    a = torch.where(at, a.clamp(min=1), a)                 # generating place: a>=1
    lo = 10 - a
    rnd = torch.randint(0, 10, (n, NDIG), device=device, generator=gen)
    b = torch.where(at, lo + rnd % (10 - lo).clamp(min=1), b)   # b >= 10 - a
    # transparent places above it: a_j + b_j == 9 for every digit pair, so a_j
    # ranges over all of 0..9 (the 9+0 and 0+9 pairs included -- restricting
    # them here leaves the model untrained on chains that run through a 9).
    b = torch.where(above, 9 - a, b)
    a[:, NDIG - 1] = a[:, NDIG - 1].clamp(min=1)           # full 8-digit width
    top = above[:, NDIG - 1]
    a[:, NDIG - 1] = torch.where(top, a[:, NDIG - 1].clamp(min=1, max=8), a[:, NDIG - 1])
    b[:, NDIG - 1] = torch.where(top, 9 - a[:, NDIG - 1], b[:, NDIG - 1].clamp(min=1))
    return a, b


def make_chain(n, device, gen=None, exclude_holdout=True, passes=2):
    a, b = chain_digits(n, device, gen)
    if exclude_holdout:
        for _ in range(passes):
            bad = (pair_bucket(digits_to_value(a), digits_to_value(b)) == 0).unsqueeze(1)
            ra, rb = chain_digits(n, device, gen)
            a = torch.where(bad, ra, a)
            b = torch.where(bad, rb, b)
    return a, b


def holdout_chain(n, device, seed=2024, chunk=1 << 21):
    """Chain cases drawn only from the held-out bucket."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    outs_a, outs_b, have = [], [], 0
    while have < n:
        a, b = chain_digits(chunk, device, gen)
        keep = pair_bucket(digits_to_value(a), digits_to_value(b)) == 0
        outs_a.append(a[keep])
        outs_b.append(b[keep])
        have += int(keep.sum())
    return torch.cat(outs_a)[:n], torch.cat(outs_b)[:n]


def targets_from_values(a_val, b_val):
    """(B,) values -> (B, 9) sum digits, LSB first."""
    s = a_val + b_val
    place = torch.tensor([10 ** i for i in range(NDIG + 1)], dtype=torch.int64, device=s.device)
    return (s.unsqueeze(-1) // place) % 10


def make_batch(n, device, prop_p=0.0, exclude_holdout=True, gen=None, passes=2):
    """Draw a training batch, resampling any pair that lands in the held-out
    bucket.  Done branch-free (no host sync) so the tiny model stays GPU-bound;
    two passes leave a residual holdout leak below 1e-6 of samples."""
    a, b = sample_digits(n, prop_p, device, gen)
    if exclude_holdout:
        for _ in range(passes):
            bad = (pair_bucket(digits_to_value(a), digits_to_value(b)) == 0).unsqueeze(1)
            ra, rb = sample_digits(n, prop_p, device, gen)
            a = torch.where(bad, ra, a)
            b = torch.where(bad, rb, b)
    return a, b


def make_holdout(n, device, prop_p=0.0, seed=1234, chunk=1 << 21):
    """Collect ``n`` pairs from the held-out bucket only."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    outs_a, outs_b, have = [], [], 0
    while have < n:
        a, b = sample_digits(chunk, prop_p, device, gen)
        keep = pair_bucket(digits_to_value(a), digits_to_value(b)) == 0
        outs_a.append(a[keep])
        outs_b.append(b[keep])
        have += int(keep.sum())
    a = torch.cat(outs_a)[:n]
    b = torch.cat(outs_b)[:n]
    return a, b
