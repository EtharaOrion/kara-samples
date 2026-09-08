"""On-GPU sampler for 8-digit addition.

Operands are always full width: a, b in [10_000_000, 99_999_999].
Digit tensors are LSB-first with N_POS=10 slots (pos0 and pos9 are (0,0)).

Three regimes are mixed, because uniform pairs almost never contain a long
carry chain and the chain is the only hard part of the task:
  uniform      -- iid digits
  transparent  -- each place independently forced to a_i + b_i == 9 w.p. p
  chain        -- a maximal carry chain: generator at place j, 9s above it

A hash of (a, b) defines a permanent held-out split so that "unseen pair"
evaluation really is unseen.
"""

import torch

N_POS = 10
PLACES = 8
POW10 = torch.tensor([10 ** i for i in range(PLACES)], dtype=torch.long)


def _hash(a, b):
    h = a * 0x9E3779B97F4A7C15 + b * 0xC2B2AE3D27D4EB4F
    h = h ^ (h >> 29)
    h = h * 0xBF58476D1CE4E5B9
    h = h ^ (h >> 32)
    return h


def to_int(dig, device):
    return (dig * POW10.to(device)).sum(-1)


def is_holdout(a, b, mod=64):
    return (_hash(a, b) % mod) == 0


def _uniform(B, device, g):
    a = torch.randint(0, 10, (B, PLACES), device=device, generator=g)
    b = torch.randint(0, 10, (B, PLACES), device=device, generator=g)
    return a, b


def _transparent(B, device, g, p=0.4):
    a, b = _uniform(B, device, g)
    m = torch.rand((B, PLACES), device=device, generator=g) < p
    b = torch.where(m, 9 - a, b)
    return a, b


def _chain(B, device, g):
    """Generator at place j, then a run of transparent (sum 9) places."""
    a, b = _uniform(B, device, g)
    j = torch.randint(0, PLACES, (B, 1), device=device, generator=g)
    L = torch.randint(0, PLACES, (B, 1), device=device, generator=g)
    idx = torch.arange(PLACES, device=device)[None, :]
    # generator place: force a_j + b_j >= 10
    gen = idx == j
    a = torch.where(gen, torch.randint(1, 10, (B, PLACES), device=device, generator=g), a)
    lo = 10 - a
    span = (10 - lo).clamp(min=1)
    bg = lo + (torch.rand((B, PLACES), device=device, generator=g) * span).long().clamp(max=9)
    b = torch.where(gen, bg.clamp(0, 9), b)
    # transparent run just above the generator
    run = (idx > j) & (idx <= j + L)
    b = torch.where(run, 9 - a, b)
    return a, b


def sample(B, device, g, mix=(0.35, 0.40, 0.25), holdout=None):
    """Returns (da, db, y) with da/db (B, N_POS) and y (B, N_POS) targets."""
    n1 = int(B * mix[0])
    n2 = int(B * mix[1])
    n3 = B - n1 - n2
    parts = [_uniform(n1, device, g), _transparent(n2, device, g), _chain(n3, device, g)]
    a = torch.cat([p[0] for p in parts], 0)
    b = torch.cat([p[1] for p in parts], 0)

    # full-width operands: most significant digit is never zero
    top = torch.randint(1, 10, (B, 2), device=device, generator=g)
    a[:, PLACES - 1] = torch.where(a[:, PLACES - 1] == 0, top[:, 0], a[:, PLACES - 1])
    b[:, PLACES - 1] = torch.where(b[:, PLACES - 1] == 0, top[:, 1], b[:, PLACES - 1])

    ai, bi = to_int(a, device), to_int(b, device)
    if holdout is not None:
        keep = is_holdout(ai, bi) if holdout else ~is_holdout(ai, bi)
        if not bool(keep.all()):
            a, b, ai, bi = a[keep], b[keep], ai[keep], bi[keep]
    return pack(a, b, ai, bi, device)


def pack(a, b, ai, bi, device):
    B = a.shape[0]
    z = torch.zeros(B, 1, dtype=torch.long, device=device)
    da = torch.cat([z, a, z], 1)
    db = torch.cat([z, b, z], 1)
    s = ai + bi
    y = torch.zeros(B, N_POS, dtype=torch.long, device=device)
    for p in range(1, N_POS):
        y[:, p] = (s // (10 ** (p - 1))) % 10
    return da, db, y


def from_ints(ai, bi, device):
    a = torch.stack([(ai // (10 ** i)) % 10 for i in range(PLACES)], 1)
    b = torch.stack([(bi // (10 ** i)) % 10 for i in range(PLACES)], 1)
    return pack(a, b, ai, bi, device)


def sample_uniform_pairs(B, device, g, holdout=True):
    """Plain uniform full-width operands, restricted to the held-out split."""
    out_a, out_b = [], []
    have = 0
    while have < B:
        ai = torch.randint(10_000_000, 100_000_000, (B,), device=device, generator=g)
        bi = torch.randint(10_000_000, 100_000_000, (B,), device=device, generator=g)
        m = is_holdout(ai, bi) if holdout else ~is_holdout(ai, bi)
        ai, bi = ai[m], bi[m]
        out_a.append(ai)
        out_b.append(bi)
        have += ai.numel()
    ai = torch.cat(out_a)[:B]
    bi = torch.cat(out_b)[:B]
    return from_ints(ai, bi, device)
