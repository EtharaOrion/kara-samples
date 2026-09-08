"""On-GPU synthetic data for digit-pair addition.

Everything is generated here (never in submission.py).  Operand pairs are split
deterministically into train / held-out buckets by hashing the digit pairs, so the
numbers reported at the end are measured on pairs the optimiser never saw.
"""
import torch

HOLDOUT_MOD = 16          # 1 pair in 16 is reserved for evaluation
MASK30 = (1 << 30) - 1


def holdout_bucket(a, b):
    """Deterministic hash of an operand pair -> bucket in [0, HOLDOUT_MOD).

    a, b: integer tensors (B, n) of digits, least-significant first.  The hash only
    depends on the pair of numbers, so a pair lands in the same bucket at every width.
    """
    h = torch.zeros(a.shape[0], dtype=torch.int64, device=a.device)
    for k in range(a.shape[1]):
        h = (h * 1000003 + a[:, k] * 10 + b[:, k]) & MASK30
    h = ((h ^ (h >> 13)) * 1000003) & MASK30
    h = h ^ (h >> 7)
    return h % HOLDOUT_MOD


def _uniform(bs, n, device, g):
    a = torch.randint(0, 10, (bs, n), device=device, generator=g)
    b = torch.randint(0, 10, (bs, n), device=device, generator=g)
    return a, b


def sample_digits(bs, n, device, g, p_transparent=0.0, p_generate=0.0):
    """Digit pairs with a controllable share of carry-transparent (a+b == 9) and
    carry-generating (a+b >= 10) places.  Transparent runs are what make carries travel,
    so the mix controls how long the carry chains get."""
    a, b = _uniform(bs, n, device, g)

    if p_transparent > 0:
        m = torch.rand((bs, n), device=device, generator=g) < p_transparent
        b = torch.where(m, 9 - a, b)

    if p_generate > 0:
        m = torch.rand((bs, n), device=device, generator=g) < p_generate
        # a in 1..9 so that a generating partner exists; b in 10-a .. 9
        ga = torch.randint(1, 10, (bs, n), device=device, generator=g)
        r = torch.rand((bs, n), device=device, generator=g)
        gb = 10 - ga + (r * ga.to(r.dtype)).long()      # uniform in [10-ga, 9]
        a = torch.where(m, ga, a)
        b = torch.where(m, gb.clamp(0, 9), b)

    return a, b


def force_full_width(a, b, g):
    """Every graded operand has its full digit width, so never train with a zero MSB."""
    n = a.shape[1]
    if n == 0:
        return a, b
    bs = a.shape[0]
    top_a = torch.randint(1, 10, (bs,), device=a.device, generator=g)
    top_b = torch.randint(1, 10, (bs,), device=a.device, generator=g)
    a = a.clone()
    b = b.clone()
    a[:, n - 1] = torch.where(a[:, n - 1] == 0, top_a, a[:, n - 1])
    b[:, n - 1] = torch.where(b[:, n - 1] == 0, top_b, b[:, n - 1])
    return a, b


def answer_digits(a, b):
    """Ripple-carry reference answer: (B, n+1) digits, least-significant first."""
    bs, n = a.shape
    out = torch.zeros(bs, n + 1, dtype=torch.long, device=a.device)
    carry = torch.zeros(bs, dtype=torch.long, device=a.device)
    for k in range(n):
        s = a[:, k] + b[:, k] + carry
        out[:, k] = s % 10
        carry = s // 10
    out[:, n] = carry
    return out


def to_tokens(a, b):
    """(B, n, ·) digits -> (B, n+2, 2) token sequence with a (0, 0) pad at each end."""
    bs, n = a.shape
    tok = torch.zeros(bs, n + 2, 2, dtype=torch.long, device=a.device)
    tok[:, 1:n + 1, 0] = a
    tok[:, 1:n + 1, 1] = b
    return tok


def make_batch(bs, n, device, g, p_transparent=0.25, p_generate=0.2,
               want_holdout=False, full_width=True):
    """A batch of (tokens, targets).  Targets are (B, n+2): position p carries answer
    digit p-1, and position 0 (the leading pad) carries digit 0."""
    # Forcing a non-zero leading digit only makes sense once there is more than one place;
    # at n == 1 it would hide the digit 0 from training entirely.
    full_width = full_width and n >= 2
    draw = bs * (HOLDOUT_MOD + 4) if want_holdout else bs * 2

    keep_a, keep_b = [], []
    have = 0
    for _ in range(32):
        a, b = sample_digits(draw, n, device, g, p_transparent, p_generate)
        if full_width:
            a, b = force_full_width(a, b, g)
        sel = (holdout_bucket(a, b) == 0) if want_holdout else (holdout_bucket(a, b) != 0)
        a, b = a[sel], b[sel]
        keep_a.append(a)
        keep_b.append(b)
        have += a.shape[0]
        if have >= bs:
            break
    a = torch.cat(keep_a)[:bs]
    b = torch.cat(keep_b)[:bs]
    assert a.shape[0] == bs, f"only got {a.shape[0]} of {bs}"

    tgt = torch.zeros(a.shape[0], n + 2, dtype=torch.long, device=device)
    tgt[:, 1:] = answer_digits(a, b)
    return to_tokens(a, b), tgt


def mixed_batch(bs, n, device, g, want_holdout=False):
    """Three regimes side by side: plain uniform digits, transparency-enriched digits
    (long carry chains), and generate-heavy digits."""
    q = max(1, bs // 3)
    parts = [
        make_batch(q, n, device, g, 0.0, 0.0, want_holdout),
        make_batch(q, n, device, g, 0.45, 0.15, want_holdout),
        make_batch(bs - 2 * q, n, device, g, 0.15, 0.45, want_holdout),
    ]
    return (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]))
