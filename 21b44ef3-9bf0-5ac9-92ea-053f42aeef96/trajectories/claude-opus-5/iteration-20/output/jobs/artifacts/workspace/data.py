"""On-GPU sampler for digit-pair addition problems.

Everything is generated from `torch.randint`; the held-out split is a
deterministic 1-in-16 hash bucket over the digit strings of the operand pair,
so a pair seen in training can never appear in evaluation and vice versa.
"""

import torch

HELD_OUT_MOD = 16
_PRIME = 1000003


def bucket(da, db):
    """Rolling polynomial hash of the interleaved digit strings -> 0..15."""
    h = torch.zeros(da.shape[0], dtype=torch.long, device=da.device)
    for k in range(da.shape[1]):
        h = (h * 131 + da[:, k] + 1) % _PRIME
        h = (h * 131 + db[:, k] + 1) % _PRIME
    return h % HELD_OUT_MOD


def _raw(batch, places, device, gen, p_transparent, p_msb_nonzero, max_digit=9):
    hi = max_digit + 1
    da = torch.randint(0, hi, (batch, places), device=device, generator=gen)
    db = torch.randint(0, hi, (batch, places), device=device, generator=gen)
    if max_digit < 9:
        # Digit-range curriculum: only the sub-alphabet 0..max_digit appears,
        # so the code table grows outward one entry at a time.
        return da, db

    # Per-sample transparency rate, so a batch mixes plain problems with ones
    # full of a+b==9 places (which is what makes carries travel).
    rate = p_transparent[torch.randint(0, len(p_transparent), (batch, 1),
                                       device=device, generator=gen)]
    make_t = torch.rand(batch, places, device=device, generator=gen) < rate
    db = torch.where(make_t, 9 - da, db)

    # A slice of the batch is a maximal carry chain: place 0 generates, every
    # later place is transparent.  This is the longest-range routing case.
    chain = torch.rand(batch, 1, device=device, generator=gen) < 0.15
    chain_a = torch.randint(0, 10, (batch, places), device=device, generator=gen)
    a0 = torch.randint(1, 10, (batch,), device=device, generator=gen)
    off = (torch.rand(batch, device=device, generator=gen) * a0).long().clamp(max=9)
    chain_a[:, 0] = a0
    chain_b = 9 - chain_a
    chain_b[:, 0] = (10 - a0) + off
    da = torch.where(chain, chain_a, da)
    db = torch.where(chain, chain_b, db)

    if p_msb_nonzero > 0 and places > 1:
        force = torch.rand(batch, device=device, generator=gen) < p_msb_nonzero
        hi = places - 1
        da[:, hi] = torch.where(force, da[:, hi].clamp(min=1), da[:, hi])
        db[:, hi] = torch.where(force, db[:, hi].clamp(min=1), db[:, hi])
    return da, db


DEFAULT_RATES = (0.0, 0.0, 0.1, 0.25, 0.45, 0.7, 0.9)


def sample(batch, places, device, gen, p_transparent=DEFAULT_RATES,
           p_msb_nonzero=0.5, held_out=False, tries=8, max_digit=9):
    """Draw a batch from one side of the held-out split.

    held_out=False -> training side (bucket != 0); True -> the 1-in-16 bucket 0.
    """
    rates = torch.tensor(p_transparent, device=device)
    if held_out:
        # Bucket 0 is rare, so oversample a pool and keep the hits.
        keep_a, keep_b, have = [], [], 0
        while have < batch:
            da, db = _raw(batch * 24, places, device, gen, rates, p_msb_nonzero, max_digit)
            hit = bucket(da, db) == 0
            keep_a.append(da[hit])
            keep_b.append(db[hit])
            have += int(hit.sum())
        return torch.cat(keep_a)[:batch], torch.cat(keep_b)[:batch]

    da, db = _raw(batch, places, device, gen, rates, p_msb_nonzero, max_digit)
    for _ in range(tries):
        bad = bucket(da, db) == 0
        if not bool(bad.any()):
            break
        na, nb = _raw(batch, places, device, gen, rates, p_msb_nonzero, max_digit)
        m = bad.unsqueeze(1)
        da = torch.where(m, na, da)
        db = torch.where(m, nb, db)
    return da, db


def tokens_and_targets(da, db):
    """(B,n) digit arrays -> tokens (B,n+2,2) and answer digits (B,n+1)."""
    batch, places = da.shape
    pad = torch.zeros(batch, 1, dtype=da.dtype, device=da.device)
    tok_a = torch.cat([pad, da, pad], dim=1)
    tok_b = torch.cat([pad, db, pad], dim=1)
    tokens = torch.stack([tok_a, tok_b], dim=-1)

    carry = torch.zeros(batch, dtype=da.dtype, device=da.device)
    out = []
    for k in range(places):
        s = da[:, k] + db[:, k] + carry
        out.append(s % 10)
        carry = torch.div(s, 10, rounding_mode="floor")
    out.append(carry)
    return tokens, torch.stack(out, dim=1)
