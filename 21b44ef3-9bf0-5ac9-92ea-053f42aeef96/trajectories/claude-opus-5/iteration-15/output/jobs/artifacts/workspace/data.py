"""On-GPU sampler for labelled addition problems (training side only).

Three regimes are mixed so that carry chains -- the part of addition that the
attention has to work for -- are not vanishingly rare:

  uniform      digits i.i.d. uniform, most-significant place forced non-zero
  transparent  each place independently forced to a+b == 9 with prob p, which
               makes carries propagate across it
  chain        an explicit maximal run of transparent places sitting on top of
               a generating place, i.e. the longest possible carry chain

A 1-in-16 hash bucket of the operand pairs is reserved as a held-out split and
is never trained on, so reported accuracy is on pairs the model has not seen.
"""
import torch

HOLDOUT_MOD = 16


def _hash_bucket(a, b):
    """Stable bucket in [0,16) from the digit pair sequence."""
    h = torch.zeros(a.shape[0], dtype=torch.int64, device=a.device)
    for i in range(a.shape[1]):
        h = (h * 1000003 + a[:, i] * 10 + b[:, i]) & 0x3FFFFFFFFFFF
    h = (h ^ (h >> 17)) * 2654435761
    return (h & 0x3FFFFFFFFFFF) % HOLDOUT_MOD


def _uniform(B, n, g, dev):
    return (torch.randint(0, 10, (B, n), generator=g, device=dev),
            torch.randint(0, 10, (B, n), generator=g, device=dev))


def _transparent(B, n, g, dev, p=0.4):
    a = torch.randint(0, 10, (B, n), generator=g, device=dev)
    b = torch.randint(0, 10, (B, n), generator=g, device=dev)
    m = torch.rand(B, n, generator=g, device=dev) < p
    return a, torch.where(m, 9 - a, b)


def _chain(B, n, g, dev):
    """A generating place at `s`, then a transparent run of length `L`."""
    a, b = _uniform(B, n, g, dev)
    s = torch.randint(0, n, (B, 1), generator=g, device=dev)
    L = torch.randint(0, n, (B, 1), generator=g, device=dev)
    idx = torch.arange(n, device=dev)[None, :]
    gen = idx == s
    run = (idx > s) & (idx <= s + L)
    # generating place: a+b >= 10
    ga = torch.randint(1, 10, (B, n), generator=g, device=dev)
    gb = torch.randint(0, 10, (B, n), generator=g, device=dev).clamp(min=1)
    gb = torch.maximum(gb, 10 - ga)
    a = torch.where(gen, ga, a)
    b = torch.where(gen, gb, b)
    # transparent run: a+b == 9
    b = torch.where(run, 9 - a, b)
    return a, b


def sample(B, n, g, dev, force_msb=True):
    """Returns tok_a, tok_b [B, n+2], target [B, n+2], keep [B] (not held out)."""
    parts = [_uniform(B // 3, n, g, dev),
             _transparent(B // 3, n, g, dev),
             _chain(B - 2 * (B // 3), n, g, dev)]
    a = torch.cat([p[0] for p in parts], 0)
    b = torch.cat([p[1] for p in parts], 0)
    if force_msb:
        hi = torch.randint(1, 10, (2, a.shape[0]), generator=g, device=dev)
        a[:, n - 1] = hi[0]
        b[:, n - 1] = hi[1]
    return pack(a, b)


def pack(a, b):
    """Pad the digit arrays into token/target tensors."""
    B, n, dev = a.shape[0], a.shape[1], a.device
    z = torch.zeros(B, 1, dtype=torch.long, device=dev)
    tok_a = torch.cat([z, a, z], 1)
    tok_b = torch.cat([z, b, z], 1)

    carry = torch.zeros(B, dtype=torch.long, device=dev)
    outs = [torch.zeros(B, dtype=torch.long, device=dev)]   # position 0 unused
    for i in range(n):
        t = a[:, i] + b[:, i] + carry
        outs.append(t % 10)
        carry = t // 10
    outs.append(carry)
    target = torch.stack(outs, 1)
    keep = _hash_bucket(a, b) != 0
    return tok_a, tok_b, target, keep


def heldout(B, n, g, dev, force_msb=True):
    """A batch restricted to the held-out bucket (for honest evaluation)."""
    ta, tb, tg, keep = sample(B, n, g, dev, force_msb)
    sel = ~keep
    return ta[sel], tb[sel], tg[sel], torch.ones(int(sel.sum()), dtype=torch.bool, device=dev)


def uniform_pairs(B, n, g, dev):
    """Plain uniform full-width operands, the graded distribution."""
    a = torch.randint(0, 10, (B, n), generator=g, device=dev)
    b = torch.randint(0, 10, (B, n), generator=g, device=dev)
    hi = torch.randint(1, 10, (2, B), generator=g, device=dev)
    a[:, n - 1] = hi[0]
    b[:, n - 1] = hi[1]
    return pack(a, b)
