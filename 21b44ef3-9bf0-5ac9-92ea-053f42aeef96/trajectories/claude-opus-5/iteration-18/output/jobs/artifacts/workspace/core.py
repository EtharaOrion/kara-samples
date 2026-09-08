"""Core: ensemble-batched model forward + on-GPU data sampler.

Everything carries a leading ensemble axis E so that E independent models
(different seeds / different random inits) train simultaneously on a shared
batch.  AdamW is elementwise, so this is exactly E independent optimisers
provided gradient clipping is done per member.

Token layout (LSB-first), for n digit places:
    P = n + 2 positions
    pos 0        : pad, digit pair (0, 0)
    pos 1..n     : the digit pairs (a_i, b_i), i = 0..n-1
    pos n+1      : pad, digit pair (0, 0)
Position p predicts answer digit p-1, so the whole sum (n+1 digits) comes out
of a single forward pass.  Position 0's prediction is unused.
"""
import math
import torch

# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

PARAM_KEYS = ['code', 'bank_w', 'knee', 'key_w', 'val_w', 'val_b',
              'q', 'lam', 'carry_w', 'fold', 'res_b', 'ls']


def masks(P, device, dtype=torch.float32):
    idx = torch.arange(P, device=device)
    dist = (idx[:, None] - idx[None, :]).to(dtype)          # p - q, >=0 allowed
    m_incl = idx[None, :] <= idx[:, None]
    m_strict = idx[None, :] < idx[:, None]
    # position 0 has no strictly-earlier position; let it look at itself so the
    # softmax is well defined.  Its output is never read.
    m_strict = m_strict | ((idx[:, None] == 0) & (idx[None, :] == 0))
    return dist, m_strict, m_incl


def bank(par, x):
    """Clamp bank.  x [E,B,P] -> u [E,U,B,P].

    Unit j is clamp(bank_w_j * (x - knee_j), 0, 1); knee_j is the position of
    the lower kink, the upper kink sits at knee_j + 1/bank_w_j.
    """
    bw = par['bank_w'][:, :, None, None]
    kn = par['knee'][:, :, None, None]
    return torch.clamp(bw * (x[:, None] - kn), 0.0, 1.0)


def forward(par, da, db, dist=None, m_strict=None, m_incl=None, want=None):
    """da, db: [B,P] long.  Returns logits [E,B,P,10] (neg squared distance)."""
    code = par['code']                                  # [E,10]
    E = code.shape[0]
    P = da.shape[1]
    if dist is None:
        dist, m_strict, m_incl = masks(P, code.device, code.dtype)
    x = code[:, da] + code[:, db]                       # [E,B,P]
    u = bank(par, x)                                    # [E,U,B,P]
    key = (par['key_w'][:, :, None, None] * u).sum(1)   # [E,B,P]
    val = (par['val_w'][:, :, None, None] * u).sum(1) + par['val_b'][:, None, None]
    sc = par['q'][:, None, None, None] * key[:, :, None, :] \
        + par['lam'][:, None, None, None] * dist        # [E,B,P,P]
    neg = torch.finfo(sc.dtype).min
    a_s = torch.softmax(sc.masked_fill(~m_strict, neg), dim=-1)
    a_i = torch.softmax(sc.masked_fill(~m_incl, neg), dim=-1)
    v = val[:, :, None, :]
    As = (a_s * v).sum(-1)
    Ai = (a_i * v).sum(-1)
    r = x + par['carry_w'][:, None, None] * As \
        + par['fold'][:, None, None] * Ai + par['res_b'][:, None, None]
    logits = -par['ls'][:, None, None, None] * (r[..., None] - code[:, None, None, :]) ** 2
    if want is not None:
        want.update(dict(x=x, u=u, key=key, val=val, a_s=a_s, a_i=a_i, r=r))
    return logits


def loss_and_acc(par, da, db, tgt, dist=None, m_strict=None, m_incl=None,
                 ls_loss=1.0):
    """CE over positions 1.. ; returns (loss [E], exact-match acc [E])."""
    logits = forward(par, da, db, dist, m_strict, m_incl)
    lg = logits[:, :, 1:, :] * ls_loss                  # drop position 0
    t = tgt[None, :, 1:, None].expand(lg.shape[0], -1, -1, 1)
    lp = lg - torch.logsumexp(lg, dim=-1, keepdim=True)
    nll = -lp.gather(-1, t).squeeze(-1)                 # [E,B,P-1]
    loss = nll.mean(dim=(1, 2))
    with torch.no_grad():
        ok = (lg.argmax(-1) == tgt[None, :, 1:]).all(-1).float().mean(-1)
    return loss, ok


@torch.no_grad()
def exact_acc(par, da, db, tgt, chunk=None):
    dist, ms, mi = masks(da.shape[1], par['code'].device, par['code'].dtype)
    logits = forward(par, da, db, dist, ms, mi)
    pred = logits[:, :, 1:, :].argmax(-1)
    return (pred == tgt[None, :, 1:]).all(-1).float().mean(-1)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

_M1, _M2 = 1000003, 999983


def bucket(da, db, n):
    """Deterministic 1-in-16 hash bucket of a digit-pair sample (train/held-out)."""
    ka = torch.zeros(da.shape[0], dtype=torch.long, device=da.device)
    kb = torch.zeros_like(ka)
    for i in range(n - 1, -1, -1):
        ka = ka * 10 + da[:, i + 1]
        kb = kb * 10 + db[:, i + 1]
    h = (ka % _M1) * 7919 + (kb % _M2) * 104729 + (ka % 97) * 15486277
    return h % 16


def _digits(B, n, g, device):
    return torch.randint(0, 10, (B, n), generator=g, device=device)


def _raw(B, n, g, device, p_uni, p_tr, tr_rate, full_width):
    """Draw B samples of n places under a mixture of carry regimes."""
    a = _digits(B, n, g, device)
    b = _digits(B, n, g, device)
    kind = torch.rand(B, generator=g, device=device)

    # regime 2: per-place transparency (a+b == 9) at rate tr_rate
    tr = torch.rand((B, n), generator=g, device=device) < tr_rate
    b_tr = 9 - a
    use_tr = (kind >= p_uni) & (kind < p_uni + p_tr)
    b = torch.where(tr & use_tr[:, None], b_tr, b)

    # regime 3: one generating place followed by a maximal transparent chain
    if n >= 2:
        start = torch.randint(0, n, (B,), generator=g, device=device)
        ln = torch.randint(1, n + 1, (B,), generator=g, device=device)
        pos = torch.arange(n, device=device)[None, :]
        is_gen = pos == start[:, None]
        is_chain = (pos > start[:, None]) & (pos <= (start + ln)[:, None])
        # generating place: a+b >= 10 (needs a >= 1, so lift a first)
        a_gen = torch.where(a < 1, torch.ones_like(a), a)
        lo = 10 - a_gen                                   # in 1..9
        span = (10 - lo).float()
        b_gen = lo + (torch.rand((B, n), generator=g, device=device) * span).long()
        b_gen = b_gen.clamp(min=1, max=9)
        b_gen = torch.maximum(b_gen, lo)
        use_ch = kind >= p_uni + p_tr
        a = torch.where(is_gen & use_ch[:, None], a_gen, a)
        b = torch.where(is_gen & use_ch[:, None], b_gen, b)
        b = torch.where(is_chain & use_ch[:, None], 9 - a, b)

    if full_width and n >= 1:
        force = torch.rand(B, generator=g, device=device) < 0.5
        top_a = 1 + torch.randint(0, 9, (B,), generator=g, device=device)
        top_b = 1 + torch.randint(0, 9, (B,), generator=g, device=device)
        a[:, n - 1] = torch.where(force, top_a, a[:, n - 1])
        b[:, n - 1] = torch.where(force, top_b, b[:, n - 1])
    return a, b


def pack(a, b):
    """digits [B,n] -> tokens [B,n+2] with (0,0) pads at both ends, plus targets."""
    B, n = a.shape
    z = torch.zeros(B, 1, dtype=a.dtype, device=a.device)
    da = torch.cat([z, a, z], 1)
    db = torch.cat([z, b, z], 1)
    s = a + b
    carry = torch.zeros(B, dtype=a.dtype, device=a.device)
    outs = []
    for i in range(n):
        t = s[:, i] + carry
        outs.append(t % 10)
        carry = t // 10
    outs.append(carry)
    tgt = torch.stack(outs, 1)                       # [B, n+1] answer digits
    tgt = torch.cat([torch.zeros(B, 1, dtype=a.dtype, device=a.device), tgt], 1)
    return da, db, tgt                               # tgt[:,p] is digit p-1


def held_set(B, n, g, device, **kw):
    """Build exactly B samples that all fall in held-out hash bucket 0."""
    das, dbs, tgts, got = [], [], [], 0
    while got < B:
        a, b = _raw(4 * B, n, g, device, kw.get('p_uni', 0.35), kw.get('p_tr', 0.40),
                    kw.get('tr_rate', 0.40), kw.get('full_width', True))
        da, db, tgt = pack(a, b)
        keep = bucket(da, db, n) == 0
        das.append(da[keep]); dbs.append(db[keep]); tgts.append(tgt[keep])
        got += int(keep.sum())
    da = torch.cat(das)[:B]; db = torch.cat(dbs)[:B]; tgt = torch.cat(tgts)[:B]
    assert bool((bucket(da, db, n) == 0).all())
    return da, db, tgt


def sample(B, n, g, device, split='train', p_uni=0.35, p_tr=0.40,
           tr_rate=0.40, full_width=True, tries=8):
    """Rejection-sample so that train and held-out sets are disjoint by hash."""
    want = 0 if split == 'held' else None
    a, b = _raw(B, n, g, device, p_uni, p_tr, tr_rate, full_width)
    da, db, tgt = pack(a, b)
    for _ in range(tries):
        hb = bucket(da, db, n)
        bad = (hb == 0) if split == 'train' else (hb != 0)
        if not bool(bad.any()):
            break
        k = int(bad.sum())
        a2, b2 = _raw(k, n, g, device, p_uni, p_tr, tr_rate, full_width)
        da2, db2, tgt2 = pack(a2, b2)
        da[bad], db[bad], tgt[bad] = da2, db2, tgt2
    return da, db, tgt
