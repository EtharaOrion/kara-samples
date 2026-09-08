"""Exact, function-preserving rewrites of a trained member, plus analysis.

Every rewrite here is a symmetry of the architecture (a change of coordinates), so the
model's answers are unchanged by construction; `check_exact` re-verifies that in
float64 on a large sample. Nothing here fits, snaps or hand-sets a value.

The symmetry group of the forward map:
  translation  code -> code + t,  rb -> rb - t,  bb -> bb - bw*t
  scale        code,rb,e -> mu*(.),  bw -> bw/mu,  ls -> ls/mu^2         (mu != 0)
  temperature  ls -> lam*ls  (lam > 0; argmax over d is unchanged)
  query/key    q -> q/nu,  kw -> nu*kw   (nu > 0; only the product enters)
  value affine v -> alpha*v + beta is undone by  e -> e/alpha  and a shift of rb
               (the induced change of the bank input is undone by bb)
  key offset   k -> k + const is invisible to the softmax (constant across j)
"""

import torch

import core, data

KEYS = list(core.SPEC.keys())


def to64(p):
    return {k: v.detach().to(torch.float64).clone() for k, v in p.items()}


def member(p, i):
    return {k: v[i:i + 1].clone() for k, v in p.items()}


def reachable_u(p):
    """All residual values u that any position can ever take: the 100 digit pairs
    (the (0,0) pad is pair (0,0), already included)."""
    a = torch.arange(10, device=p["code"].device).repeat_interleave(10)
    b = torch.arange(10, device=p["code"].device).repeat(10)
    return p["code"][0][a] + p["code"][0][b] + p["rb"][0], (a + b)


# --------------------------------------------------------------------- rewrites

def gauge_q(p):
    """q -> 1, absorbed into kw."""
    p = {k: v.clone() for k, v in p.items()}
    q = p["q"][0].clone()
    s = torch.sign(q)
    p["kw"] = p["kw"] * q
    p["q"] = torch.ones_like(p["q"])
    return p


def gauge_ls(p):
    """ls -> 1 (argmax over d is invariant to a positive rescale of the logits)."""
    p = {k: v.clone() for k, v in p.items()}
    assert float(p["ls"][0]) > 0
    p["ls"] = torch.ones_like(p["ls"])
    return p


def gauge_translate(p, t=None):
    """code -> code + t (default: put code[0] at the origin)."""
    p = {k: v.clone() for k, v in p.items()}
    if t is None:
        t = -p["code"][0, 0].clone()
    p["bb"] = p["bb"] - p["bw"] * t
    p["code"] = p["code"] + t
    p["rb"] = p["rb"] - t
    return p


def gauge_scale(p, mu=None):
    """Rescale the residual stream (default: put e[0] at 1)."""
    p = {k: v.clone() for k, v in p.items()}
    if mu is None:
        mu = 1.0 / p["e"][0, 0].clone()
    p["code"] = p["code"] * mu
    p["rb"] = p["rb"] * mu
    p["e"] = p["e"] * mu
    p["bw"] = p["bw"] / mu
    p["ls"] = p["ls"] / (mu * mu)
    return p


def gauge_value(p, alpha, beta):
    """v -> alpha*v + beta, undone exactly by e and rb (and bb for the bank input)."""
    p = {k: v.clone() for k, v in p.items()}
    esum = p["e"][0, 0] + p["e"][0, 1]
    p["vw"] = p["vw"] * alpha
    p["vb"] = p["vb"] * alpha + beta
    p["e"] = p["e"] / alpha
    shift = -beta * esum / alpha                     # additive change wanted on z
    p["bb"] = p["bb"] - p["bw"] * shift
    p["rb"] = p["rb"] + shift
    return p


def drop_units(p, keep):
    """Remove bank units. Units that are constant over the reachable set are folded
    into vb (their value part) and dropped from the key (a constant key offset is
    invisible to the softmax)."""
    p = {k: v.clone() for k, v in p.items()}
    u, _ = reachable_u(p)
    g = torch.clamp(u[:, None] * p["bw"][0] + p["bb"][0], 0.0, 1.0)         # [100,U]
    drop = [j for j in range(g.shape[1]) if j not in keep]
    for j in drop:
        const = g[:, j]
        inert = float(p["kw"][0, j].abs()) == 0.0 and float(p["vw"][0, j].abs()) == 0.0
        if inert:
            continue                       # already folded into another unit
        assert float(const.max() - const.min()) < 1e-12, f"unit {j} is not constant"
        p["vb"] = p["vb"] + p["vw"][:, j] * const[0]
    for name in ("bw", "bb", "kw", "vw"):
        p[name] = p[name][:, keep].contiguous()
    return p


def flip_unit(p, j):
    """g_j -> 1 - g_j.  clamp(-x+1,0,1) == 1 - clamp(x,0,1) holds for every real x,
    so this is exact everywhere, not just on the reachable set. The induced constant
    in the key is a uniform offset (softmax-invariant) and the one in the value moves
    into vb."""
    p = {k: v.clone() for k, v in p.items()}
    p["bb"][:, j] = 1.0 - p["bb"][:, j]
    p["bw"][:, j] = -p["bw"][:, j]
    p["vb"] = p["vb"] + p["vw"][:, j]
    p["vw"][:, j] = -p["vw"][:, j]
    p["kw"][:, j] = -p["kw"][:, j]
    return p


def merge_units(p, j, i):
    """Fold unit j into unit i when their bank outputs coincide on the reachable set."""
    p = {k: v.clone() for k, v in p.items()}
    p["kw"][:, i] = p["kw"][:, i] + p["kw"][:, j]
    p["vw"][:, i] = p["vw"][:, i] + p["vw"][:, j]
    p["kw"][:, j] = 0.0
    p["vw"][:, j] = 0.0
    return p


def narrow_bank(p, B=8.0):
    """Remove `bw` by fixing the ramp width to 1/B, holding each knee where training
    put it.  For bw < 0 the unit ramps down over [theta - 1/|bw|, theta]; for bw > 0 it
    ramps up over [theta, theta + 1/bw]. Increasing |bw| shrinks that interval around
    the same theta, so the new ramp is a SUBSET of the old one. If no reachable
    residual value lay inside the old ramp, none lies inside the new one either and
    the substitution is exactly function-preserving on the whole input domain.

    Returns (p_new, ok, report). `ok` is False if some |bw| exceeds B (the ramp would
    widen) or if the bank was not already saturated.
    """
    p = {k: v.clone() for k, v in p.items()}
    bw = p["bw"][0]
    theta = -p["bb"][0] / bw
    u, s = reachable_u(p)
    pre_old = u[:, None] * bw + p["bb"][0]
    slack_old = float(torch.maximum(-pre_old, pre_old - 1.0).min())
    widen = [j for j in range(bw.shape[0]) if float(bw[j].abs()) > B]
    newbw = torch.where(bw < 0, torch.full_like(bw, -B), torch.full_like(bw, B))
    p["bw"] = newbw.unsqueeze(0)
    p["bb"] = (-newbw * theta).unsqueeze(0)
    pre_new = u[:, None] * p["bw"][0] + p["bb"][0]
    slack_new = float(torch.maximum(-pre_new, pre_new - 1.0).min())
    ok = (not widen) and slack_old > 0 and slack_new > 0
    return p, ok, dict(slack_old=slack_old, slack_new=slack_new,
                       bw_old=[float(x) for x in bw], widened=widen)


def unit_sets(p):
    """For each bank unit, the set of sums s for which it outputs 1 (None if not binary)."""
    u, s = reachable_u(p)
    g = torch.clamp(u[:, None] * p["bw"][0] + p["bb"][0], 0.0, 1.0)
    out = []
    for j in range(g.shape[1]):
        col = g[:, j]
        if not bool(((col < 1e-12) | (col > 1 - 1e-12)).all()):
            out.append(None)
        else:
            out.append(frozenset(int(x) for x in s[col > 0.5].unique()))
    return out, g


def normalize(p, verbose=True):
    """Exact gauge normalisation. Returns (p, info) with
    q=1, ls=1, code[0]=0, e[0]=1, vw=(0,1), vb=0, two bank units in the
    {1[s<=8], 1[s>=10]} basis. Raises if the member is not in the expected form."""
    p0 = p
    log = []
    p = gauge_q(p)

    # --- drop units that are constant over the whole input domain
    sets, g = unit_sets(p)
    LE8 = frozenset(range(0, 9)); GE10 = frozenset(range(10, 19))
    GE9 = frozenset(range(9, 19)); LE9 = frozenset(range(0, 10))
    live = [j for j, sg in enumerate(sets)
            if sg is not None and len(sg) not in (0, 19)]
    dead = [j for j in range(len(sets)) if j not in live]
    for j in dead:
        if sets[j] is None:
            raise ValueError(f"unit {j} is not saturated; cannot reduce exactly")
    p = drop_units(p, keep=live) if dead else p
    log.append(f"dropped constant units {dead}")

    # --- flip so every unit is one of the two canonical indicators
    sets, _ = unit_sets(p)
    for j, sg in enumerate(sets):
        if sg in (GE9, LE9):
            p = flip_unit(p, j)
    sets, _ = unit_sets(p)
    for j, sg in enumerate(sets):
        if sg not in (LE8, GE10):
            raise ValueError(f"unit {j} computes {sorted(sg)}, not 1[s<=8] or 1[s>=10]")

    # --- merge duplicates, then keep exactly one of each kind
    keep = {}
    for j, sg in enumerate(sets):
        if sg in keep:
            p = merge_units(p, j, keep[sg])
        else:
            keep[sg] = j
    if set(keep) != {LE8, GE10}:
        raise ValueError(f"bank does not span both indicators: {[sorted(x) for x in keep]}")
    order = [keep[LE8], keep[GE10]]
    p = drop_units(p, keep=order)
    log.append(f"merged to 2 units, order {order}")

    p = canonical_value(p)                 # vw -> (0,1), vb -> 0
    p = gauge_scale(p)                     # e[0] -> 1
    p = gauge_translate(p)                 # code[0] -> 0
    p = gauge_ls(p)                        # ls -> 1
    if verbose:
        for line in log:
            print("  ", line)
    return p, log


def canonical_value(p):
    """Rescale the value stream so absorb -> 0 and generate -> 1."""
    u, s = reachable_u(p)
    g = torch.clamp(u[:, None] * p["bw"][0] + p["bb"][0], 0.0, 1.0)
    v = (g * p["vw"][0]).sum(-1) + p["vb"][0]
    vA = v[s <= 8].mean()
    vG = v[s >= 10].mean()
    alpha = 1.0 / (vG - vA)
    beta = -vA * alpha
    return gauge_value(p, alpha, beta)


# ------------------------------------------------------------------ diagnostics

def profile(p):
    """Key / value / residual as a function of s = a + b over the reachable set."""
    u, s = reachable_u(p)
    g = torch.clamp(u[:, None] * p["bw"][0] + p["bb"][0], 0.0, 1.0)
    k = (g * p["kw"][0]).sum(-1) * p["q"][0]
    v = (g * p["vw"][0]).sum(-1) + p["vb"][0]
    rows = []
    for ss in range(19):
        m = s == ss
        rows.append((ss, float(u[m].mean()), float(k[m].min()), float(k[m].max()),
                     float(v[m].min()), float(v[m].max())))
    return rows


def code_fit(p):
    c = p["code"][0]
    d = torch.arange(10, dtype=c.dtype, device=c.device)
    A = torch.stack([d, torch.ones_like(d)], 1)
    sol = torch.linalg.lstsq(A, c[:, None]).solution[:, 0]
    resid = (c - (A @ sol)).abs().max()
    return float(sol[0]), float(sol[1]), float(resid)


@torch.no_grad()
def check_exact(p1, p2, n_list=(8, 5, 12), total=200000, seed=1234):
    """Confirm two parameter sets give identical argmax decisions (float64)."""
    dev = p1["code"].device
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    worst = 0.0
    agree = 0
    seen = 0
    for n in n_list:
        left = total
        while left > 0:
            m = min(8192, left)
            A, B, Y = data.batch(m, n, dev, gen)
            o1 = core.forward_core(p1, A, B)
            o2 = core.forward_core(p2, A, B)
            a1 = o1.argmax(-1)[:, :, 1:]
            a2 = o2.argmax(-1)[:, :, 1:]
            agree += int((a1 == a2).all(-1).sum())
            seen += a1.shape[1]
            left -= m
    return agree / seen


@torch.no_grad()
def accuracy(p, n=8, total=1 << 20, held_out=True, seed=99, bs=8192):
    dev = p["code"].device
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    hit = 0; seen = 0
    while seen < total:
        m = min(bs, total - seen)
        A, B, Y = data.batch(m, n, dev, gen, held_out=held_out)
        out = core.forward_core(p, A, B)
        ok = (out.argmax(-1)[:, :, 1:] == Y[None, :, 1:]).all(-1)
        hit += int(ok.sum()); seen += A.shape[0]
    return hit / seen
