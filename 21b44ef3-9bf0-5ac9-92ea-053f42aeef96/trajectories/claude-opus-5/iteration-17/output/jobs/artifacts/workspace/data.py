"""On-GPU sampler for n-place addition, with a deterministic held-out split.

The held-out split is a 1-in-16 hash bucket over the (a, b) operand pair.  The
hash is computed from the LSB-first digit arrays with fixed weights, so a pair
hashes identically no matter how many leading zero places the representation
carries.  Training draws only from buckets != 0; evaluation draws only from
bucket 0, so no evaluated pair is ever trained on.
"""
import torch

_MOD = 1 << 31
_MAXN = 32
_WA = []
_WB = []
_x = 1
for _i in range(_MAXN):
    _WA.append(_x)
    _x = (_x * 1103515245 + 12345) % _MOD
_x = 7
for _i in range(_MAXN):
    _WB.append(_x)
    _x = (_x * 1103515245 + 12345) % _MOD

_WCACHE = {}


def _weights(n, device):
    key = (n, str(device))
    if key not in _WCACHE:
        _WCACHE[key] = (
            torch.tensor(_WA[:n], device=device, dtype=torch.int64),
            torch.tensor(_WB[:n], device=device, dtype=torch.int64),
        )
    return _WCACHE[key]


def bucket(a, b):
    """a, b: (B, n) int64 LSB-first digits -> (B,) bucket in [0, 16)."""
    n = a.shape[-1]
    wa, wb = _weights(n, a.device)
    h = ((a * wa).sum(-1) + (b * wb).sum(-1)) % _MOD
    h = (h * 2654435761) % _MOD
    return h % 16


def _raw_sample(B, n, device, p_trans, gen):
    """Digit pairs with per-place probability p_trans of being carry-transparent
    (a_i + b_i == 9).  Both operands are forced to full width (top digit >= 1),
    which is what the grader uses."""
    a = torch.randint(0, 10, (B, n), device=device, generator=gen)
    b = torch.randint(0, 10, (B, n), device=device, generator=gen)
    if p_trans > 0:
        m = torch.rand((B, n), device=device, generator=gen) < p_trans
        b = torch.where(m, 9 - a, b)

    # top place, kept full-width for both operands
    top_t = torch.rand((B,), device=device, generator=gen) < p_trans
    a_t = torch.randint(1, 9, (B,), device=device, generator=gen)   # 1..8
    b_t = 9 - a_t                                                  # 1..8
    a_u = torch.randint(1, 10, (B,), device=device, generator=gen)
    b_u = torch.randint(1, 10, (B,), device=device, generator=gen)
    a[:, n - 1] = torch.where(top_t, a_t, a_u)
    b[:, n - 1] = torch.where(top_t, b_t, b_u)
    return a, b


def sample(B, n, device, regimes=(0.0, 0.4, 0.9), gen=None, held_out=False,
           tries=12):
    """Draw B pairs, split evenly across the transparency regimes, restricted to
    the held-out bucket (held_out=True) or its complement (held_out=False)."""
    outa = torch.empty(0, n, dtype=torch.int64, device=device)
    outb = torch.empty(0, n, dtype=torch.int64, device=device)
    over = 40 if held_out else 2
    for _ in range(tries):
        need = B - outa.shape[0]
        if need <= 0:
            break
        m = max(1024, need * over)
        per = m // len(regimes) + 1
        aa, bb = [], []
        for pt in regimes:
            a, b = _raw_sample(per, n, device, pt, gen)
            aa.append(a)
            bb.append(b)
        a = torch.cat(aa)
        b = torch.cat(bb)
        keep = (bucket(a, b) == 0) if held_out else (bucket(a, b) != 0)
        outa = torch.cat([outa, a[keep]])
        outb = torch.cat([outb, b[keep]])
    return outa[:B], outb[:B]


def uniform_pairs(B, n, device, gen=None, held_out=None):
    """Plain uniform full-width operands (the grading distribution)."""
    a = torch.randint(0, 10, (B, n), device=device, generator=gen)
    b = torch.randint(0, 10, (B, n), device=device, generator=gen)
    a[:, n - 1] = torch.randint(1, 10, (B,), device=device, generator=gen)
    b[:, n - 1] = torch.randint(1, 10, (B,), device=device, generator=gen)
    if held_out is None:
        return a, b
    keep = (bucket(a, b) == 0) if held_out else (bucket(a, b) != 0)
    return a[keep], b[keep]


def tokens(a, b):
    """(B, n) digits -> (B, n+2, 2) tokens with (0,0) pads at both ends."""
    B, n = a.shape
    z = torch.zeros(B, 1, dtype=a.dtype, device=a.device)
    A = torch.cat([z, a, z], dim=1)
    Bd = torch.cat([z, b, z], dim=1)
    return torch.stack([A, Bd], dim=-1)


def targets(a, b):
    """(B, n) digits -> (B, n+2) target digits with -100 at position 0.

    Position p carries answer digit p-1; the last position carries the final
    carry-out digit.  Computed here with plain integer arithmetic purely to
    LABEL training data -- the model never sees it at inference.
    """
    B, n = a.shape
    s = a + b
    carry = torch.zeros(B, dtype=a.dtype, device=a.device)
    outs = []
    for i in range(n):
        t = s[:, i] + carry
        outs.append(t % 10)
        carry = torch.div(t, 10, rounding_mode="floor")
    outs.append(carry)
    y = torch.stack(outs, dim=1)                       # (B, n+1)
    pad = torch.full((B, 1), -100, dtype=a.dtype, device=a.device)
    return torch.cat([pad, y], dim=1)                  # (B, n+2)
