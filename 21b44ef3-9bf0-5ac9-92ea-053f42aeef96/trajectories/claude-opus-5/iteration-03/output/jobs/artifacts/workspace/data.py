"""Synthetic 8-digit addition data, generated on-GPU, LSB-first.

Three sampling modes are mixed:
  uniform      - every place iid uniform
  transparent  - each place is a "carry-transparent" pair (a+b == 9) with
                 probability p, which is what makes carries propagate
  chain        - an explicit generate place followed by a run of transparent
                 places, i.e. a maximal carry chain

A hash of the operand pair defines a held-out class that training never sees.
"""
import torch

NDIG = 8
HOLDOUT_MOD = 64


def pair_hash(a_dig, b_dig):
    """Deterministic bucket in [0, HOLDOUT_MOD) from the digit tensors."""
    w = torch.tensor([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000],
                     device=a_dig.device, dtype=torch.long)
    a = (a_dig.long() * w).sum(-1)
    b = (b_dig.long() * w).sum(-1)
    h = a * 2654435761 + b * 40503 + (a ^ b) * 97
    return (h % HOLDOUT_MOD).abs()


def _ri(lo, hi, shape, device, gen):
    return torch.randint(lo, hi, shape, device=device, generator=gen)


def sample(B, device, gen, mix=(0.35, 0.40, 0.25), p_trans=0.40, p_nogen=0.4):
    """Return (a_dig, b_dig) of shape (B, 8), LSB first, both operands 8-wide."""
    n = NDIG
    idx = torch.arange(n, device=device).view(1, n)
    msb = idx == (n - 1)

    # --- uniform component (MSB forced non-zero) -------------------------
    a_u = _ri(0, 10, (B, n), device, gen)
    b_u = _ri(0, 10, (B, n), device, gen)
    a_u = torch.where(msb, _ri(1, 10, (B, n), device, gen), a_u)
    b_u = torch.where(msb, _ri(1, 10, (B, n), device, gen), b_u)

    # --- transparent component (a + b == 9) ------------------------------
    # At the MSB both operands must stay >= 1, so a is drawn from 1..8 there.
    a_t = _ri(0, 10, (B, n), device, gen)
    a_t = torch.where(msb, _ri(1, 9, (B, n), device, gen), a_t)
    b_t = 9 - a_t

    # --- generate component (a + b >= 10) --------------------------------
    a_g = _ri(1, 10, (B, n), device, gen)
    span = _ri(0, 10, (B, n), device, gen) % a_g
    b_g = (10 - a_g) + span

    r = torch.rand((B, 1), device=device, generator=gen)
    m_uni = r < mix[0]
    m_chain = r >= (mix[0] + mix[1])

    # per-place transparency for the "transparent-enriched" mode
    take_t = torch.rand((B, n), device=device, generator=gen) < p_trans
    take_t = take_t & ~m_uni & ~m_chain

    # Explicit maximal chain: a generate at place s with transparent places on
    # (s, e].  With probability p_nogen the generate is dropped and the whole
    # run 0..e is transparent, so the carry into place e+1 has to be resolved
    # all the way back against the boundary slot (carry-in 0).  Without this
    # case the model never learns to look past a long transparent run to the
    # sink, and mispredicts things like 98190410 + 91809589.
    s = _ri(0, n, (B, 1), device, gen)
    e = s + (torch.rand((B, 1), device=device, generator=gen) * (n - s)).long()
    e = e.clamp(max=n - 1)
    no_gen = torch.rand((B, 1), device=device, generator=gen) < p_nogen
    chain_gen = m_chain & (idx == s) & ~no_gen
    chain_tr = m_chain & (idx <= e) & torch.where(no_gen, idx >= 0, idx > s)

    take_t = take_t | chain_tr

    a = torch.where(take_t, a_t, a_u)
    b = torch.where(take_t, b_t, b_u)
    a = torch.where(chain_gen, a_g, a)
    b = torch.where(chain_gen, b_g, b)
    return a, b


def sample_train(B, device, gen, **kw):
    """Sample a training batch with held-out pairs resampled away.

    Returns (a, b, weight) where weight is 0 for the vanishing fraction of rows
    that stayed in the held-out bucket after several resampling rounds.
    """
    a, b = sample(B, device, gen, **kw)
    bad = pair_hash(a, b) == 0
    # One resampling round, no host sync in the training loop.  The residual
    # held-out fraction is 64**-2 ~ 2e-4 and those rows get zero loss weight,
    # so the model never sees a pair from the evaluation bucket.
    a2, b2 = sample(B, device, gen, **kw)
    m = bad.view(B, 1)
    a = torch.where(m, a2, a)
    b = torch.where(m, b2, b)
    return a, b, (pair_hash(a, b) != 0).float()


def sample_eval(B, device, gen, **kw):
    """Sample from the held-out bucket only (rejection-free approximation:
    generate a large pool and keep the held-out rows)."""
    out_a, out_b = [], []
    got = 0
    while got < B:
        a, b = sample(B * HOLDOUT_MOD // 4 + 1024, device, gen, **kw)
        keep = pair_hash(a, b) == 0
        a, b = a[keep], b[keep]
        out_a.append(a)
        out_b.append(b)
        got += int(a.shape[0])
    a = torch.cat(out_a)[:B]
    b = torch.cat(out_b)[:B]
    return a, b


def targets(a_dig, b_dig):
    """Digits of a + b, LSB first, length 9 (8 places + carry-out)."""
    s = a_dig + b_dig
    carry = torch.zeros_like(s[:, :1])
    outs = []
    for i in range(NDIG):
        t = s[:, i:i + 1] + carry
        outs.append(t % 10)
        carry = t // 10
    outs.append(carry)
    return torch.cat(outs, dim=1)
