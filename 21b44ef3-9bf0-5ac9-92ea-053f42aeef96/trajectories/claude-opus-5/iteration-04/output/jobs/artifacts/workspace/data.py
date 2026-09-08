"""On-GPU sampler for 8-digit addition, with a hashed held-out split.

Both operands always carry the full 8-digit width (place 7 nonzero in each),
matching the graded range [10_000_000, 99_999_999].

Per place we draw a *carry structure* so the hard cases (propagate runs that
move a carry across many places) are common enough to learn:
    uniform      a, b ~ U{0..9}
    transparent  a + b == 9    (a carry entering the place leaves it)
    generate     a + b >= 10   (the place emits a carry on its own)
    absorb       a + b <= 8    (the place stops any incoming carry)
"""
import torch

T = 10          # sequence length: sink + 8 places + carry-out slot
P = 8           # number of real places


def _cat_choice(probs, shape, device, g):
    """Sample category indices with the given per-category probabilities."""
    p = torch.tensor(probs, device=device).cumsum(0)
    u = torch.rand(shape, device=device, generator=g)
    return (u.unsqueeze(-1) > p).sum(-1).clamp_(max=len(probs) - 1)


def _draw(kind, shape, device, g, msb):
    """Digit pairs (a, b) for one place given a structure code.

    kind: 0 uniform, 1 transparent, 2 generate, 3 absorb.
    msb:  if True both digits are forced >= 1 (full operand width).
    """
    lo = 1 if msb else 0
    a = torch.randint(lo, 10, shape, device=device, generator=g)
    b = torch.randint(lo, 10, shape, device=device, generator=g)

    # transparent: a + b == 9
    at = torch.randint(lo, 10 - lo, shape, device=device, generator=g)
    bt = 9 - at

    # generate: a + b >= 10, uniform over the admissible sums
    s = torch.randint(10, 19, shape, device=device, generator=g)
    alo = torch.clamp(s - 9, min=lo)
    ahi = torch.clamp(s - lo, max=9)
    ag = alo + (torch.rand(shape, device=device, generator=g)
                * (ahi - alo + 1).float()).long().clamp_(max=0 if False else 10 ** 9)
    ag = torch.minimum(ag, ahi)
    bg = s - ag

    # absorb: a + b <= 8
    s2 = torch.randint(2 * lo, 9, shape, device=device, generator=g)
    aalo = torch.clamp(s2 - 9, min=lo)
    aahi = torch.clamp(s2 - lo, max=9)
    aa = aalo + (torch.rand(shape, device=device, generator=g)
                 * (aahi - aalo + 1).float()).long()
    aa = torch.minimum(aa, aahi)
    ba = s2 - aa

    A = torch.stack([a, at, ag, aa])
    B = torch.stack([b, bt, bg, ba])
    idx = kind.unsqueeze(0)
    return A.gather(0, idx).squeeze(0), B.gather(0, idx).squeeze(0)


# mixture over generation modes: (weight, per-place structure probabilities)
MODES = (
    (0.35, (1.00, 0.00, 0.00, 0.00)),   # uniform digits
    (0.40, (0.60, 0.40, 0.00, 0.00)),   # transparent-enriched
    (0.25, (0.05, 0.72, 0.15, 0.08)),   # long carry chains
)


def sample(B, device, g=None, modes=MODES):
    """Returns digits (B, P, 2) long, LSB first."""
    w = torch.tensor([m[0] for m in modes], device=device)
    mode = _cat_choice((w / w.sum()).tolist(), (B,), device, g)
    kinds = torch.empty(B, P, dtype=torch.long, device=device)
    for mi, (_, probs) in enumerate(modes):
        k = _cat_choice(probs, (B, P), device, g)
        kinds = torch.where((mode == mi).unsqueeze(-1), k, kinds)
    msb = torch.zeros(B, P, dtype=torch.bool, device=device)
    msb[:, P - 1] = True
    a, b = _draw(kinds, (B, P), device, g, False)
    am, bm = _draw(kinds, (B, P), device, g, True)
    a = torch.where(msb, am, a)
    b = torch.where(msb, bm, b)
    return torch.stack([a, b], -1)


def tokens(dig):
    """(B, P, 2) digit pairs -> (B, T, 2) token sequence with sink + carry slot."""
    z = dig.new_zeros(dig.shape[0], 1, 2)
    return torch.cat([z, dig, z], 1)


def targets(dig):
    """(B, P, 2) -> (B, 9) true output digits, LSB first."""
    s = dig[..., 0] + dig[..., 1]
    out = []
    carry = torch.zeros_like(s[:, 0])
    for i in range(P):
        t = s[:, i] + carry
        out.append(t % 10)
        carry = (t >= 10).long()
    out.append(carry)
    return torch.stack(out, -1)


_MIX = torch.tensor([1, 11, 121, 1331, 14641, 161051, 1771561, 19487171,
                     3, 33, 363, 3993, 43923, 483153, 5314683, 58461509],
                    dtype=torch.long)


def held_out(dig, mod=64):
    """True for pairs reserved for evaluation (excluded from training)."""
    m = _MIX.to(dig.device)
    flat = torch.cat([dig[..., 0], dig[..., 1]], -1)
    hsh = (flat * m).sum(-1) * 2654435761 + 1442695040888963407
    hsh = hsh ^ (hsh >> 29)
    hsh = hsh * 6364136223846793005
    hsh = hsh ^ (hsh >> 32)
    return (hsh % mod) == 0


def train_batch(B, device, g=None, modes=MODES):
    """A training batch with held-out pairs resampled away."""
    # Three unconditional resampling rounds; no host sync in the training loop.
    # 1/64 of draws land in the held-out bucket, so the residual leak rate is
    # 64**-4 = 6e-8 of samples -- far below the resolution of any measurement here.
    dig = sample(B, device, g, modes)
    for _ in range(3):
        rep = sample(B, device, g, modes)
        dig = torch.where(held_out(dig).view(-1, 1, 1), rep, dig)
    return dig


def eval_set(n, device, seed, modes=MODES):
    """n held-out pairs (hash bucket 0), drawn from the given mixture."""
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    got = 0
    while got < n:
        dig = sample(min(max(n * 80, 1 << 16), 1 << 21), device, g, modes)
        dig = dig[held_out(dig)]
        out.append(dig)
        got += dig.shape[0]
    return torch.cat(out)[:n]


UNIFORM = ((1.0, (1.0, 0.0, 0.0, 0.0)),)
HARD = ((0.5, (0.05, 0.72, 0.15, 0.08)), (0.5, (0.50, 0.50, 0.0, 0.0)))
