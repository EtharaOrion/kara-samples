"""On-GPU sampler for n-place addition, with a deterministic held-out split.

A pair is assigned to a hash bucket in 0..15 derived from its digits. Bucket 0 is
NEVER shown during training and is used only for held-out evaluation, so "unseen"
means unseen by construction rather than by the low collision probability of random
8-digit sampling.

Sampling regimes (mixed in every batch):
  uniform : digits i.i.d. uniform
  trans   : each place independently forced to a_i + b_i == 9 with prob p
            (a "transparent" place propagates an incoming carry unchanged)
  chain   : place 0 forced to a+b >= 10, all higher places forced to a+b == 9
            (a maximal carry chain: the carry has to travel the whole width)
The MSB place is always forced so that both operands use the full digit width,
matching how the graded operands are drawn.
"""

import torch

POW = None


def _hash_bucket(A, Bm):
    """A, Bm: int64 [Bs, n] digit arrays. Returns int64 [Bs] in 0..15."""
    n = A.shape[1]
    pw = torch.pow(torch.full((n,), 10, dtype=torch.int64, device=A.device),
                   torch.arange(n, device=A.device))
    va = (A * pw).sum(1)
    vb = (Bm * pw).sum(1)
    x = va * 1000003 + vb * 1000000007 + 12345
    x = x ^ (x >> 30)
    x = x * (-4658895280553007687)
    x = x ^ (x >> 27)
    x = x * (-7723592293110705685)
    x = x ^ (x >> 31)
    return x & 15


def _uniform(bs, n, device, gen):
    return (torch.randint(0, 10, (bs, n), generator=gen, device=device),
            torch.randint(0, 10, (bs, n), generator=gen, device=device))


def _trans(bs, n, device, gen, p=0.4):
    a, b = _uniform(bs, n, device, gen)
    m = torch.rand(bs, n, generator=gen, device=device) < p
    b = torch.where(m, 9 - a, b)
    return a, b


def _chain(bs, n, device, gen):
    """Maximal carry chain: place 0 generates a carry, every higher place is
    transparent, so the carry has to be propagated across the whole width."""
    a, _ = _uniform(bs, n, device, gen)
    b = 9 - a                                              # all places transparent
    a0 = torch.randint(1, 10, (bs,), generator=gen, device=device)      # 1..9
    off = (torch.rand(bs, generator=gen, device=device) * a0).to(torch.int64)
    b0 = (10 - a0) + torch.clamp(off, max=a0 - 1)          # in [10-a0, 9] -> a0+b0 >= 10
    a = a.clone(); b = b.clone()
    a[:, 0] = a0
    b[:, 0] = b0
    return a, b


def _force_msb(a, b, gen):
    """Force both operands to use the full width (leading digit non-zero)."""
    bs = a.shape[0]
    dev = a.device
    top = a.shape[1] - 1
    a = a.clone(); b = b.clone()
    za = a[:, top] == 0
    zb = b[:, top] == 0
    ra = torch.randint(1, 10, (bs,), generator=gen, device=dev)
    rb = torch.randint(1, 10, (bs,), generator=gen, device=dev)
    a[:, top] = torch.where(za, ra, a[:, top])
    b[:, top] = torch.where(zb, rb, b[:, top])
    return a, b


def sample(bs, n, device, gen, mix=(0.35, 0.40, 0.25), force_msb=True):
    n1 = int(bs * mix[0]); n2 = int(bs * mix[1]); n3 = bs - n1 - n2
    parts = []
    if n1: parts.append(_uniform(n1, n, device, gen))
    if n2: parts.append(_trans(n2, n, device, gen))
    if n3: parts.append(_chain(n3, n, device, gen))
    a = torch.cat([x[0] for x in parts], 0)
    b = torch.cat([x[1] for x in parts], 0)
    if force_msb:
        a, b = _force_msb(a, b, gen)
    return a, b


def targets(a, b):
    """Exact answer digits for the P = n+2 token layout, computed by integer arithmetic
    on the *labels only*. Returns int64 [Bs, P]; entry i is the answer digit that
    position i must predict (position 0 is a don't-care and is masked out of the loss)."""
    bs, n = a.shape
    s = a + b
    carry = torch.zeros(bs, dtype=torch.int64, device=a.device)
    outs = []
    for i in range(n):
        t = s[:, i] + carry
        outs.append(t % 10)
        carry = t // 10
    outs.append(carry)                                     # leading digit
    y = torch.stack(outs, 1)                               # [bs, n+1]
    pad = torch.zeros(bs, 1, dtype=torch.int64, device=a.device)
    return torch.cat([pad, y], 1)                          # [bs, n+2]


def tokens(a, b):
    """Pad the digit arrays into the P = n+2 token layout."""
    bs = a.shape[0]
    z = torch.zeros(bs, 1, dtype=torch.int64, device=a.device)
    A = torch.cat([z, a, z], 1)
    B = torch.cat([z, b, z], 1)
    return A, B


def batch(bs, n, device, gen, held_out=False, **kw):
    """Draw a batch from the training side (bucket != 0) or the held-out side (bucket == 0)."""
    outs_a, outs_b = [], []
    got = 0
    for _ in range(64):
        a, b = sample(max(bs * 2, 256), n, device, gen, **kw)
        keep = (_hash_bucket(a, b) == 0) if held_out else (_hash_bucket(a, b) != 0)
        a, b = a[keep], b[keep]
        outs_a.append(a); outs_b.append(b)
        got += a.shape[0]
        if got >= bs:
            break
    a = torch.cat(outs_a, 0)[:bs]
    b = torch.cat(outs_b, 0)[:bs]
    A, B = tokens(a, b)
    return A, B, targets(a, b)
