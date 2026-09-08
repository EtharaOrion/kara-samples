"""Training-side machinery: digit-pair data, held-out split, and an
ensemble-batched mirror of the forward pass in model_src.py.

Everything here stays out of the graded file.  The mirror trains E independent
members at once by giving every parameter a leading ensemble axis; Adam is
elementwise, so the members never interact except through the shared batch.
"""

import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- digit pairs

_A = torch.arange(10).repeat_interleave(10)
_B = torch.arange(10).repeat(10)
PAIRS = torch.stack([_A, _B], dim=1).to(DEV)              # (100, 2)
_S = (_A + _B).to(DEV)

# carry classes of a place: 0 = absorb (a+b <= 8), 1 = transparent (== 9),
# 2 = generate (>= 10).  Sizes 45 / 10 / 45.
CLASS_OF = torch.where(_S < 9, 0, torch.where(_S == 9, 1, 2)).to(DEV)
_lists = [torch.nonzero(CLASS_OF == k).squeeze(1) for k in range(3)]
CLASS_SIZE = torch.tensor([len(x) for x in _lists], device=DEV)
CLASS_TABLE = torch.zeros(3, int(CLASS_SIZE.max()), dtype=torch.long, device=DEV)
for k, x in enumerate(_lists):
    CLASS_TABLE[k, : len(x)] = x

# per-place class profiles and how often each is used
PROFILES = torch.tensor(
    [[0.45, 0.10, 0.45],    # matches uniform digits
     [0.30, 0.40, 0.30],    # transparent-enriched
     [0.05, 0.90, 0.05]],   # long carry chains
    device=DEV)
PROFILE_MIX = torch.tensor([0.35, 0.40, 0.25], device=DEV)


def hash_bucket(a, b):
    """Deterministic 1-in-16 bucket of the operand pair, computed on the
    zero-padded 8-digit form so the split does not depend on the width."""
    n = a.shape[1]
    powers = (10 ** torch.arange(n, device=a.device)).to(torch.int64)
    av = (a.to(torch.int64) * powers).sum(-1)
    bv = (b.to(torch.int64) * powers).sum(-1)
    h = av * 100000000 + bv
    h = (h ^ (h >> 29)) & 0x3FFFFFFF
    h = (h * 1103515245 + 12345) & 0x3FFFFFFF
    h = (h ^ (h >> 15)) & 0x3FFFFFFF
    return h % 16


def raw_sample(n, batch, gen=None):
    """Digit pairs of width n from the profile mixture."""
    pick = torch.multinomial(PROFILE_MIX, batch, replacement=True, generator=gen)
    probs = PROFILES[pick]                                       # (batch, 3)
    cdf = probs.cumsum(-1).unsqueeze(1)                           # (batch, 1, 3)
    r = torch.rand(batch, n, 1, device=DEV, generator=gen)
    cls = (r > cdf).sum(-1).clamp(max=2)                          # (batch, n)
    sizes = CLASS_SIZE[cls]
    j = (torch.rand(batch, n, device=DEV, generator=gen) * sizes).long()
    j = torch.minimum(j, sizes - 1)
    idx = CLASS_TABLE[cls, j]
    return PAIRS[idx, 0], PAIRS[idx, 1]


def answer(a, b):
    """True sum digits, LSB first, width n+1."""
    n = a.shape[1]
    out = torch.zeros(a.shape[0], n + 1, dtype=torch.long, device=a.device)
    carry = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
    for i in range(n):
        t = a[:, i] + b[:, i] + carry
        out[:, i] = t % 10
        carry = t // 10
    out[:, n] = carry
    return out


def raw_uniform(n, batch, gen=None):
    """Uniform digits -- the graded distribution -- with the most significant
    place forced non-zero, as every graded operand is full width."""
    a = torch.randint(0, 10, (batch, n), device=DEV, generator=gen)
    b = torch.randint(0, 10, (batch, n), device=DEV, generator=gen)
    if n > 1:
        a[:, -1] = torch.randint(1, 10, (batch,), device=DEV, generator=gen)
        b[:, -1] = torch.randint(1, 10, (batch,), device=DEV, generator=gen)
    return a, b


def _split_draw(draw, n, batch, split, gen):
    """Draw `batch` pairs lying on the requested side of the held-out split by
    oversampling and filtering, so neither side is approximated."""
    want_heldout = split == "heldout"
    over = 24 if want_heldout else 2
    got_a, got_b, have = [], [], 0
    for _ in range(200):
        a, b = draw(n, max(batch * over, 1024), gen)
        keep = (hash_bucket(a, b) == 0) == want_heldout
        got_a.append(a[keep])
        got_b.append(b[keep])
        have += int(keep.sum())
        if have >= batch:
            break
    a = torch.cat(got_a)[:batch]
    b = torch.cat(got_b)[:batch]
    assert a.shape[0] == batch
    assert not bool(((hash_bucket(a, b) == 0) != want_heldout).any())
    return a, b, answer(a, b)


def sample(n, batch, split="train", gen=None):
    """Carry-structured digit pairs from one side of the held-out split."""
    if split == "any":
        a, b = raw_sample(n, batch, gen)
        return a, b, answer(a, b)
    return _split_draw(raw_sample, n, batch, split, gen)


def uniform_sample(n, batch, split="train", gen=None):
    """Uniform full-width digit pairs from one side of the held-out split."""
    if split == "any":
        a, b = raw_uniform(n, batch, gen)
        return a, b, answer(a, b)
    return _split_draw(raw_uniform, n, batch, split, gen)


# ------------------------------------------------------------- ensemble model

PARAM_NAMES = ["code_free", "carry_w", "knee", "fold",
               "log_slope", "log_key", "log_rec", "log_ls"]

SHIP = {"gate_slope": 8.0, "key_scale": 400.0, "recency": -12.0}


def init_params(E, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def u(shape, lo, hi):
        return torch.rand(*shape, device=DEV, generator=g) * (hi - lo) + lo

    return {
        "code_free": u((E, 9), -2.0, 2.0),          # code[1..9]; code[0] = 0
        "carry_w": u((E,), -2.0, 2.0),
        "knee": u((E, 2), -4.0, 4.0),
        "fold": u((E,), -2.0, 2.0),
        "log_slope": u((E,), -0.5, 1.0),            # gate slope in [0.6, 2.7]
        "log_key": u((E,), 0.0, 1.5),               # key contrast in [1, 4.5]
        "log_rec": u((E,), -0.7, 0.7),              # recency in [-2, -0.5]
        "log_ls": u((E,), 0.0, 1.0),                # read-out temperature
    }


def full_code(p):
    E = p["code_free"].shape[0]
    zero = torch.zeros(E, 1, device=p["code_free"].device, dtype=p["code_free"].dtype)
    return torch.cat([zero, p["code_free"]], dim=1)             # (E, 10)


def residual(p, da, db, ship_consts=False, kill_key=False):
    """Ensemble forward up to the residual stream.  Returns (res, code, ls).

    ship_consts=True replaces the three sharpness values with the constants the
    graded file carries, which is how a member is checked before shipping.
    kill_key=True freezes the content term of the attention scores at its batch
    mean, which is the attention ablation used in the audit.
    """
    code = full_code(p)
    z = code[:, da] + code[:, db]                                # (E, B, n)
    pad = torch.zeros(z.shape[0], z.shape[1], 1, device=z.device, dtype=z.dtype)
    z = torch.cat([pad, z, pad], dim=2)                          # (E, B, P)
    E, B, P = z.shape

    if ship_consts:
        slope = torch.full((E,), SHIP["gate_slope"], device=z.device)
        kscale = torch.full((E,), SHIP["key_scale"], device=z.device)
        rec = torch.full((E,), SHIP["recency"], device=z.device)
        ls = torch.ones(E, device=z.device)
    else:
        slope = p["log_slope"].exp()
        kscale = p["log_key"].exp()
        rec = -p["log_rec"].exp()
        ls = p["log_ls"].exp()

    u = torch.clamp(slope[:, None, None, None]
                    * (z.unsqueeze(-1) - p["knee"][:, None, None, :]), 0.0, 1.0)
    val = u[..., 1]
    key = kscale[:, None, None] * (u[..., 1] - u[..., 0])
    if kill_key:
        key = key.mean(dim=1, keepdim=True).expand_as(key)

    pos = torch.arange(P, device=z.device)
    gap = (pos.unsqueeze(-1) - pos.unsqueeze(0)).to(z.dtype)     # (P, P)
    score = key[:, :, None, :] + rec[:, None, None, None] * gap
    blocked = torch.full_like(gap, -1e9)
    strict = score + torch.where(gap > 0, torch.zeros_like(gap), blocked)
    incl = score + torch.where(gap >= 0, torch.zeros_like(gap), blocked)

    ci = torch.einsum("ebpq,ebq->ebp", strict.softmax(-1), val)
    co = torch.einsum("ebpq,ebq->ebp", incl.softmax(-1), val)
    res = z + p["carry_w"][:, None, None] * ci + p["fold"][:, None, None] * co
    return res[:, :, 1:], code, ls


def fwd(p, da, db, ship_consts=False, kill_key=False):
    """Ensemble forward.  Returns (E, B, n+1, 10) logits."""
    res, code, ls = residual(p, da, db, ship_consts=ship_consts, kill_key=kill_key)
    return -ls[:, None, None, None] * (res.unsqueeze(-1) - code[:, None, None, :]).abs()


def margins(p, da, db, tgt, ship_consts=True):
    """Per-position read-out margin: how much closer the residual sits to the
    right prototype than to the nearest wrong one, in absolute code units.
    Positive everywhere means the sum is right."""
    res, code, _ = residual(p, da, db, ship_consts=ship_consts)
    d = (res.unsqueeze(-1) - code[:, None, None, :]).abs()       # (E, B, n+1, 10)
    t = tgt[None, :, :, None].expand(d.shape[0], -1, -1, 1)
    dt = d.gather(-1, t).squeeze(-1)
    dw = d.scatter(-1, t, float("inf")).min(-1).values
    return dw - dt


def loss_and_exact(p, da, db, tgt, ship_consts=False):
    """Returns per-member mean NLL and per-member count of exactly right sums.
    The count is an integer: exact-match rates computed as float means of
    boolean tensors do not compare equal to 1."""
    logits = fwd(p, da, db, ship_consts=ship_consts)
    logp = logits.log_softmax(-1)
    t = tgt[None, :, :, None].expand(logits.shape[0], -1, -1, 1)
    nll = -logp.gather(-1, t).squeeze(-1)                        # (E, B, n+1)
    ok = (logits.argmax(-1) == tgt[None]).all(-1)                # (E, B)
    return nll.mean(dim=(1, 2)), ok.sum(dim=1)


@torch.no_grad()
def exact_counts(p, cases, ship_consts=False, chunk=None):
    """Exact-match counts over a list of (da, db, tgt) batches."""
    E = p["code_free"].shape[0]
    tot = torch.zeros(E, dtype=torch.long, device=DEV)
    seen = 0
    for da, db, tgt in cases:
        step = chunk or da.shape[0]
        for s in range(0, da.shape[0], step):
            sl = slice(s, s + step)
            logits = fwd(p, da[sl], db[sl], ship_consts=ship_consts)
            tot += (logits.argmax(-1) == tgt[None, sl]).all(-1).sum(dim=1)
        seen += da.shape[0]
    return tot, seen


def sat_penalty(p, margin=1.0, ship_consts=False):
    """How far the clamp bank is from being saturated on every token it can
    ever see -- all 100 digit pairs plus the (0,0) pad -- computed in closed
    form from the weights.  A sampled version of this misses rare pairs and
    lets one of them sit mid-clamp while accuracy still reads 1.0.

    Saturation is what makes key and value functions of the carry class alone,
    which is what the whole-domain certificate rests on.
    """
    code = full_code(p)                                          # (E, 10)
    z = code[:, :, None] + code[:, None, :]                      # (E, 10, 10)
    slope = (torch.full_like(p["log_slope"], SHIP["gate_slope"]) if ship_consts
             else p["log_slope"].exp())
    e = slope[:, None, None, None] * (z[..., None] - p["knee"][:, None, None, :])
    slack = torch.maximum(-e, e - 1.0)                           # >= 0 iff saturated
    return torch.relu(margin - slack).mean(dim=(1, 2, 3))


def per_member_clip(params, max_norm=1.0):
    sq = None
    for t in params.values():
        if t.grad is None:
            continue
        g = t.grad.reshape(t.shape[0], -1).pow(2).sum(1)
        sq = g if sq is None else sq + g
    if sq is None:
        return
    scale = (max_norm / (sq.sqrt() + 1e-12)).clamp(max=1.0)
    for t in params.values():
        if t.grad is None:
            continue
        t.grad.mul_(scale.reshape(-1, *([1] * (t.dim() - 1))))


def take(params, idx):
    return {k: v[idx].detach().clone() for k, v in params.items()}


def repeat(params, times):
    return {k: v.repeat_interleave(times, dim=0).detach().clone()
            for k, v in params.items()}
