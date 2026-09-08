"""Synthetic 8-digit addition data.

All operands are full-width: a, b in [10_000_000, 99_999_999].
Digits are stored LSB-first, shape (n, 8).

A deterministic hash of (a, b) reserves ~1% of the pair space as a held-out
split that training never sees, so reported accuracy is on unseen pairs.
"""
import torch

POW10 = None


def _pow10(device):
    global POW10
    if POW10 is None or POW10.device != device:
        POW10 = torch.tensor([10 ** i for i in range(8)], dtype=torch.long, device=device)
    return POW10


def digits_to_value(d):
    return (d.long() * _pow10(d.device)).sum(-1)


def is_heldout(av, bv):
    """~1% of the (a, b) space, reserved for evaluation only."""
    h = av * 2654435761 + bv * 2246822519
    h = h ^ (h >> 13)
    h = h * 668265263
    h = h ^ (h >> 17)
    return (h % 1000) < 10


def _fix_msb(a, b, trans, gen):
    """Place 7 must satisfy a7 >= 1 and b7 >= 1 (full-width operands)."""
    n = a.shape[0]
    dev = a.device
    msb_t = trans[:, 7]
    a7t = torch.randint(1, 9, (n,), device=dev, generator=gen)  # 1..8 -> b7 = 9-a7 in 1..8
    a7u = torch.randint(1, 10, (n,), device=dev, generator=gen)
    b7u = torch.randint(1, 10, (n,), device=dev, generator=gen)
    a[:, 7] = torch.where(msb_t, a7t, a7u)
    b[:, 7] = torch.where(msb_t, 9 - a7t, b7u)
    return a, b


def sample_transparency(n, q, device, gen):
    """Digits where each place is 'carry-transparent' (a+b == 9) with prob q.

    q may be a scalar or an (n, 1) tensor of per-example probabilities.
    """
    a = torch.randint(0, 10, (n, 8), device=device, generator=gen)
    b = torch.randint(0, 10, (n, 8), device=device, generator=gen)
    trans = torch.rand((n, 8), device=device, generator=gen) < q
    b = torch.where(trans, 9 - a, b)
    return _fix_msb(a, b, trans, gen)


def sample_chain(n, device, gen, min_start=0):
    """A place that generates a carry, followed by a run of transparent places.

    Produces the longest carry chains, including the maximal one that ripples
    from place 0 all the way out of place 7.
    """
    a, b = sample_transparency(n, 1.0, device, gen)
    idx = torch.arange(8, device=device).view(1, 8)
    i0 = torch.randint(min_start, 8, (n, 1), device=device, generator=gen)

    # below the start place: unconstrained digits
    ar = torch.randint(0, 10, (n, 8), device=device, generator=gen)
    br = torch.randint(0, 10, (n, 8), device=device, generator=gen)
    below = idx < i0
    a = torch.where(below, ar, a)
    b = torch.where(below, br, b)

    # at the start place: force a + b >= 10
    ag = torch.randint(1, 10, (n,), device=device, generator=gen)
    u = (torch.rand((n,), device=device, generator=gen) * ag).long()
    bg = 10 - ag + u  # in 1..9, and ag + bg >= 10
    at = idx == i0
    a = torch.where(at, ag.view(n, 1), a)
    b = torch.where(at, bg.view(n, 1), b)
    # place 7 is either transparent (fixed by sample_transparency) or the
    # generating place (ag, bg are both >= 1), so it stays full-width.
    return a, b


def target_digits(a, b):
    """Exact sum digits (n, 9), LSB first; index 8 is the carry-out digit."""
    n = a.shape[0]
    carry = torch.zeros(n, dtype=torch.long, device=a.device)
    outs = []
    for i in range(8):
        t = a[:, i] + b[:, i] + carry
        outs.append(t % 10)
        carry = t // 10
    outs.append(carry)
    return torch.stack(outs, dim=1)


def to_tokens(a, b):
    """(n, 8) digit pairs -> (n, 10, 2) sequence: sink, 8 places, carry-out slot."""
    n = a.shape[0]
    z = torch.zeros((n, 1), dtype=torch.long, device=a.device)
    at = torch.cat([z, a, z], dim=1)
    bt = torch.cat([z, b, z], dim=1)
    return torch.stack([at, bt], dim=-1)


def make_batch(n, device, gen, frac_uniform=0.35, frac_chain=0.25, heldout=False):
    """Mixture batch. Returns (tokens (n,10,2), targets (n,9))."""
    n_u = int(n * frac_uniform)
    n_c = int(n * frac_chain)
    n_t = n - n_u - n_c
    parts_a, parts_b = [], []
    if n_u:
        au, bu = sample_transparency(n_u, 0.0, device, gen)
        parts_a.append(au)
        parts_b.append(bu)
    if n_t:
        q = torch.rand((n_t, 1), device=device, generator=gen)
        at_, bt_ = sample_transparency(n_t, q, device, gen)
        parts_a.append(at_)
        parts_b.append(bt_)
    if n_c:
        ac, bc = sample_chain(n_c, device, gen)
        parts_a.append(ac)
        parts_b.append(bc)
    a = torch.cat(parts_a, 0)
    b = torch.cat(parts_b, 0)

    keep = is_heldout(digits_to_value(a), digits_to_value(b))
    if not heldout:
        keep = ~keep
    a, b = a[keep], b[keep]
    return to_tokens(a, b), target_digits(a, b)


def heldout_set(n, device, seed, kind="uniform"):
    """Deterministic evaluation set drawn only from the held-out 1%."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    acca, accb = [], []
    got = 0
    while got < n:
        m = min(4 << 20, max(1 << 16, (n - got) * 1200))
        if kind == "uniform":
            a, b = sample_transparency(m, 0.0, device, gen)
        elif kind == "chain":
            a, b = sample_chain(m, device, gen)
        elif kind == "maxchain":
            a, b = sample_chain(m, device, gen, min_start=0)
            # keep only chains that start at place 0 and ripple all the way out
            s = a + b
            ok = (s[:, 0] >= 10) & (s[:, 1:] == 9).all(1)
            a, b = a[ok], b[ok]
        else:
            raise ValueError(kind)
        sel = is_heldout(digits_to_value(a), digits_to_value(b))
        a, b = a[sel], b[sel]
        if a.shape[0]:
            acca.append(a)
            accb.append(b)
            got += a.shape[0]
    a = torch.cat(acca, 0)[:n]
    b = torch.cat(accb, 0)[:n]
    return to_tokens(a, b), target_digits(a, b)
