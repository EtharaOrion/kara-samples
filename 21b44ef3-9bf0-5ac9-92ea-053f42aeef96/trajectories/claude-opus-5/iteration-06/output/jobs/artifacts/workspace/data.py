"""Training/evaluation data for 8-digit addition.  Training-side only."""

import torch

NP = 8          # digit places per operand
P = 10          # sequence positions: pad, 8 places, pad

# Deterministic hash of an operand pair -> a held-out bucket.  Pairs whose
# bucket is 0 are never trained on and are the only ones used for eval.
_HA = torch.tensor([1, 13, 169, 2197, 28561, 371293, 4826809, 62748517])
_HB = torch.tensor([7, 91, 1183, 15379, 199927, 2599051, 33787663, 439239743])
_MOD = 1000003
_HOLD = 16      # 1 pair in 16 is held out


def bucket(a, b):
    """a, b: (B, 8) digit tensors (LSB first) -> (B,) bucket index."""
    h = (a * _HA.to(a.device)).sum(-1) + (b * _HB.to(a.device)).sum(-1)
    return (h % _MOD) % _HOLD


def _uniform(n, dev, g):
    a = torch.randint(0, 10, (n, NP), device=dev, generator=g)
    b = torch.randint(0, 10, (n, NP), device=dev, generator=g)
    a[:, -1] = torch.randint(1, 10, (n,), device=dev, generator=g)
    b[:, -1] = torch.randint(1, 10, (n,), device=dev, generator=g)
    return a, b


def _structured(n, pa, pt, dev, g):
    """Sample each place from {absorb, transparent, generate} then fill digits.

    'transparent' places (a + b == 9) pass a carry through, so a run of them is
    what makes a carry travel far; sampling them explicitly lets the training
    mix contain long carry chains that uniform digits almost never produce.
    """
    u = torch.rand((n, NP), device=dev, generator=g)
    cls = (u > pa).long() + (u > pa + pt).long()        # 0 absorb, 1 transparent, 2 generate
    r1 = torch.rand((n, NP), device=dev, generator=g)
    r2 = torch.rand((n, NP), device=dev, generator=g)

    # absorb: s in [0, 8], a in [0, s]
    s0 = (r1 * 9).long()
    a0 = (r2 * (s0 + 1).float()).long()
    # transparent: a in [0, 9], b = 9 - a
    a1 = (r2 * 10).long()
    s1 = torch.full_like(a1, 9)
    # generate: s in [10, 18], a in [s - 9, 9]
    s2 = 10 + (r1 * 9).long()
    a2 = (s2 - 9) + (r2 * (19 - s2).float()).long()

    s = torch.where(cls == 0, s0, torch.where(cls == 1, s1, s2))
    a = torch.where(cls == 0, a0, torch.where(cls == 1, a1, a2))

    # the most significant place must keep both operands 8 digits wide
    m = NP - 1
    s0m = 2 + (r1[:, m] * 7).long()                      # absorb, s in [2, 8]
    a0m = 1 + (r2[:, m] * (s0m - 1).float()).long()
    a1m = 1 + (r2[:, m] * 8).long()                      # transparent, a in [1, 8]
    cm = cls[:, m]
    s[:, m] = torch.where(cm == 0, s0m, torch.where(cm == 1, torch.full_like(s0m, 9), s2[:, m]))
    a[:, m] = torch.where(cm == 0, a0m, torch.where(cm == 1, a1m, a2[:, m]))
    return a, s - a


# fraction of the batch, absorb prob, transparent prob
MIX = ((0.35, None, None), (0.30, 0.30, 0.40), (0.20, 0.10, 0.80), (0.15, 0.02, 0.96))


def raw_batch(n, dev, g, mix=MIX):
    parts = []
    for frac, pa, pt in mix:
        k = max(1, int(round(n * frac)))
        parts.append(_uniform(k, dev, g) if pa is None else _structured(k, pa, pt, dev, g))
    a = torch.cat([p[0] for p in parts])[:n]
    b = torch.cat([p[1] for p in parts])[:n]
    return a, b


def labels(a, b):
    """Exact answer digits by carry propagation.  (B, 8),(B, 8) -> (B, 9)."""
    s = a + b
    out, carry = [], torch.zeros_like(s[:, 0])
    for i in range(NP):
        t = s[:, i] + carry
        out.append(t % 10)
        carry = t // 10
    out.append(carry)
    return torch.stack(out, -1)


def _pad(a, b):
    z = torch.zeros_like(a[:, :1])
    return torch.cat([z, a, z], 1), torch.cat([z, b, z], 1)


def batch(n, dev, g, train=True, mix=MIX):
    """(da, db, y): padded digit streams (n, 10) and answer digits (n, 9)."""
    keep_a, keep_b, got = [], [], 0
    over = 1.5 if train else 1.2 * _HOLD
    while got < n:
        a, b = raw_batch(int(n * over) + 64, dev, g, mix)
        m = (bucket(a, b) != 0) if train else (bucket(a, b) == 0)
        a, b = a[m], b[m]
        keep_a.append(a)
        keep_b.append(b)
        got += a.shape[0]
    a = torch.cat(keep_a)[:n]
    b = torch.cat(keep_b)[:n]
    da, db = _pad(a, b)
    return da, db, labels(a, b)
