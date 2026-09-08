"""On-GPU sampler for n-digit addition, with a deterministic train/held-out split.

Operand pairs are split 15:1 by a deterministic hash of (a, b).  Bucket 0 is
never trained on and is used for held-out evaluation, so reported accuracy is
always on pairs the model has not seen.

Three sampling regimes are mixed, because uniform digits almost never produce
long carry chains and the carry-lookahead mechanism only shows up under them:

  uniform      : digits iid uniform
  transparent  : each place independently made "transparent" (a+b == 9) w.p. 0.4
  chain        : same but w.p. 0.9, which yields near-maximal carry chains

Both operands always have a non-zero leading digit, matching the graded domain.
"""

import torch

_M = 2147483647  # 2**31 - 1


def to_int(d):
    """d: [..., n] LSB-first digits -> int64 value."""
    n = d.shape[-1]
    pw = (10 ** torch.arange(n, device=d.device, dtype=torch.int64))
    return (d.to(torch.int64) * pw).sum(-1)


def bucket(a_int, b_int):
    """Deterministic 0..15 bucket for an operand pair.  Bucket 0 is held out."""
    h = (a_int % _M) * 1000003 % _M
    h = (h + b_int % _M) % _M
    h = h * 16777619 % _M
    h = (h ^ (h >> 13)) % _M
    h = h * 2246822519 % _M
    return h % 16


def _transparent(a, b, p, gen):
    """With probability p per place, force b = 9 - a (a carry-transparent place)."""
    m = torch.rand(a.shape, generator=gen, device=a.device) < p
    return torch.where(m, 9 - a, b)


def sample(B, n, device, gen, mix=(0.35, 0.40, 0.25)):
    """Returns LSB-first digit tensors a, b of shape [B, n]."""
    a = torch.randint(0, 10, (B, n), generator=gen, device=device)
    b = torch.randint(0, 10, (B, n), generator=gen, device=device)

    r = torch.rand((B, 1), generator=gen, device=device)
    p_tr = torch.zeros((B, 1), device=device)
    p_tr = torch.where(r < mix[0], torch.zeros_like(p_tr), p_tr)
    p_tr = torch.where((r >= mix[0]) & (r < mix[0] + mix[1]),
                       torch.full_like(p_tr, 0.40), p_tr)
    p_tr = torch.where(r >= mix[0] + mix[1], torch.full_like(p_tr, 0.90), p_tr)
    b = _transparent(a, b, p_tr, gen)

    # Half the structured rows get a fresh uniform bottom place, so long
    # transparent runs appear both with and without an incoming carry to
    # propagate through them.
    reseed = (torch.rand((B,), generator=gen, device=device) < 0.5) & (p_tr[:, 0] > 0)
    b0 = torch.randint(0, 10, (B,), generator=gen, device=device)
    b[:, 0] = torch.where(reseed, b0, b[:, 0])

    # leading digit non-zero for both operands; keep transparency where possible
    hi_a = a[:, n - 1]
    hi_b = b[:, n - 1]
    tr_hi = (hi_a + hi_b) == 9
    fix_a = torch.randint(1, 9, (B,), generator=gen, device=device)   # 1..8
    new_a = torch.where(tr_hi, fix_a, torch.randint(1, 10, (B,), generator=gen, device=device))
    keep = (hi_a >= 1) & (hi_b >= 1)
    a[:, n - 1] = torch.where(keep, hi_a, new_a)
    b[:, n - 1] = torch.where(keep, hi_b, torch.where(tr_hi, 9 - new_a,
                                                      torch.randint(1, 10, (B,), generator=gen, device=device)))
    return a, b


def pad(a, b):
    """[B,n] digits -> [B,n+2] with (0,0) pads at both ends."""
    B, n = a.shape
    z = torch.zeros(B, 1, dtype=a.dtype, device=a.device)
    return torch.cat([z, a, z], 1), torch.cat([z, b, z], 1)


def targets(a, b):
    """Answer digits at padded positions.  Position p holds answer digit p-1;
    position 0 is a don't-care.  Returns [B, n+2] with position 0 set to 0 and
    a mask marking the positions that count."""
    B, n = a.shape
    s = a + b
    carry = torch.zeros(B, dtype=a.dtype, device=a.device)
    out = []
    for i in range(n):
        t = s[:, i] + carry
        out.append(t % 10)
        carry = (t >= 10).to(a.dtype)
    out.append(carry)                       # final carry -> position n+1
    t = torch.stack(out, 1)                 # [B, n+1] answer digits 0..n
    z = torch.zeros(B, 1, dtype=a.dtype, device=a.device)
    tgt = torch.cat([z, t], 1)              # [B, n+2]
    mask = torch.ones(B, n + 2, dtype=torch.bool, device=a.device)
    mask[:, 0] = False
    return tgt, mask


def batch(B, n, device, gen, held_out=False, mix=(0.35, 0.40, 0.25)):
    """Sample a batch restricted to (or to the complement of) the held-out bucket.

    Rejection-samples until enough rows are on the requested side of the split.
    """
    keep_a, keep_b = [], []
    have = 0
    for _ in range(64):
        a, b = sample(max(B, 1024), n, device, gen, mix)
        bk = bucket(to_int(a), to_int(b))
        sel = (bk == 0) if held_out else (bk != 0)
        a, b = a[sel], b[sel]
        keep_a.append(a); keep_b.append(b)
        have += a.shape[0]
        if have >= B:
            break
    a = torch.cat(keep_a)[:B]
    b = torch.cat(keep_b)[:B]
    return a, b
