"""Whole-domain correctness certificate for a reduced member.

The graded domain has ~8.1e15 operand pairs, so sampling cannot establish
correctness. This factors the model exactly instead:

  Part 1 (bank saturation, 100 checks). Every position's residual value u comes from
    one of the 100 digit pairs. Check that both clamp pre-activations lie strictly
    outside [0,1] for all 100, so the bank output is exactly binary, and that the two
    units compute exactly 1[a+b<=8] and 1[a+b>=10]. This makes each place's key and
    value depend on its digit pair only through its class A/T/G (absorb / transparent
    / generate) -- and it is what makes freezing the ramp width `bw` exact.

  Part 2 (attention, 3^8 = 6561 checks). With classes fixed, the key vector of a
    length-10 sequence is determined by the class pattern of the 8 places. Enumerate
    all 6561 patterns, run the exact softmax in float64, and measure the largest
    deviation delta between the two heads' outputs and the ideal carry-in / carry-out
    of the standard carry recurrence.

  Part 3 (read-out, 100 x 2 checks). For every digit pair and every carry-in, check
    the noise-free z decodes to the right digit, and record its distance m to the
    nearest decision boundary of the tied nearest-prototype read-out.

  Verdict: if (|e0| + |e1|) * delta < m, every input in the domain is answered
    correctly, with a safety factor of m / ((|e0|+|e1|) * delta).
"""

import itertools
import torch

A_, T_, G_ = 0, 1, 2


def bank_report(p):
    """Part 1. Returns (ok, min slack, class of each of the 100 pairs)."""
    dev = p["code"].device
    a = torch.arange(10, device=dev).repeat_interleave(10)
    b = torch.arange(10, device=dev).repeat(10)
    u = p["code"][0][a] + p["code"][0][b] + p["rb"][0]
    pre = u[:, None] * p["bw"][0] + p["bb"][0]                  # [100,2]
    # how far each pre-activation sits outside the ramp interval [0,1]; must be > 0
    # for every one of the 100 pairs and both units, or the bank is not exactly binary
    slack_min = float(torch.maximum(-pre, pre - 1.0).min())
    g = torch.clamp(pre, 0.0, 1.0)
    s = a + b
    want0 = (s <= 8).to(g.dtype)
    want1 = (s >= 10).to(g.dtype)
    ok = bool((g[:, 0] == want0).all() and (g[:, 1] == want1).all())
    cls = torch.where(s <= 8, torch.full_like(s, A_),
                      torch.where(s == 9, torch.full_like(s, T_), torch.full_like(s, G_)))
    return ok, slack_min, cls, u


def _ideal_carries(pattern):
    """Standard carry recurrence over classes; returns carry_in / carry_out per place."""
    cin, cout = [], []
    c = 0
    for cl in pattern:
        cin.append(c)
        c = 1 if cl == G_ else (0 if cl == A_ else c)
        cout.append(c)
    return cin, cout


def _check_assumptions(p):
    """Part 2 replaces the network by a class-level model. That substitution is only
    valid for the specific constants below, so check them rather than assume them."""
    kw = p["kw"][0].tolist()
    vw = p["vw"][0].tolist()
    if kw[0] != kw[1]:
        raise ValueError(f"kw={kw}: absorb and generate places must share a key for the "
                         f"class-level model to be exact")
    if vw != [0.0, 1.0]:
        raise ValueError(f"vw={vw}: the value must be exactly bank unit 1")
    if float(p["vb"][0]) != 0.0:
        raise ValueError(f"vb={float(p['vb'][0])}: the value must have no offset")
    if float(p["q"][0]) != 1.0:
        raise ValueError(f"q={float(p['q'][0])}: the certificate assumes unit query scale")
    if float(p["lam"][0, 0]) >= 0 or float(p["lam"][0, 1]) >= 0:
        raise ValueError(f"lam={p['lam'][0].tolist()}: recency bias must be negative")


def attention_report(p, n=8, chunk=512):
    """Part 2. Max deviation of the two heads from the ideal carry signals."""
    _check_assumptions(p)
    dev = p["code"].device
    dt = torch.float64
    P = n + 2
    K = float(p["kw"][0, 0])            # key of a non-transparent place (both units equal)
    lam = p["lam"][0].to(dt)
    idx = torch.arange(P, device=dev)
    dist = (idx[:, None] - idx[None, :]).to(dt)
    strict = (dist >= 1.0).clone(); strict[0, 0] = True
    mask = torch.stack([strict, dist >= 0.0], 0)

    pats = list(itertools.product((A_, T_, G_), repeat=n))
    worst = 0.0
    worst_pat = None
    for s0 in range(0, len(pats), chunk):
        blk = pats[s0:s0 + chunk]
        M = len(blk)
        cl = torch.tensor([[A_] + list(q) + [A_] for q in blk], device=dev)     # [M,P]
        key = torch.where(cl == T_, torch.zeros(1, dtype=dt, device=dev),
                          torch.full((1,), K, dtype=dt, device=dev))            # [M,P]
        val = (cl == G_).to(dt)                                                 # v(T)=0
        lg = key[:, None, None, :] + lam[None, :, None, None] * dist[None, None]
        lg = lg.masked_fill(~mask[None], -1e300)
        att = torch.softmax(lg, dim=-1)
        c = (att * val[:, None, None, :]).sum(-1)                               # [M,2,P]
        # ideal carries, vectorised over the pattern block
        cin = torch.zeros(M, P, dtype=dt, device=dev)
        cout = torch.zeros(M, P, dtype=dt, device=dev)
        cur = torch.zeros(M, dtype=dt, device=dev)
        for t in range(P):
            cin[:, t] = cur
            ct = cl[:, t]
            cur = torch.where(ct == G_, torch.ones_like(cur),
                              torch.where(ct == A_, torch.zeros_like(cur), cur))
            cout[:, t] = cur
        ideal = torch.stack([cin, cout], 1)                                     # [M,2,P]
        dev_max = (c - ideal).abs().amax(dim=(1, 2))
        j = int(dev_max.argmax())
        if float(dev_max[j]) > worst:
            worst = float(dev_max[j]); worst_pat = blk[j]
    return worst, worst_pat


def readout_report(p):
    """Part 3. Smallest distance from a noise-free z to a read-out decision boundary."""
    dev = p["code"].device
    code = p["code"][0].to(torch.float64)
    e0 = float(p["e"][0, 0]); e1 = float(p["e"][0, 1])
    rb = float(p["rb"][0])
    a = torch.arange(10, device=dev).repeat_interleave(10)
    b = torch.arange(10, device=dev).repeat(10)
    worst = float("inf"); bad = 0; worst_case = None
    for cin in (0, 1):
        s = a + b + cin
        y = s % 10
        cout = (s >= 10).to(torch.float64)
        z = code[a] + code[b] + rb + e0 * cin + e1 * cout
        d = (z[:, None] - code[None, :]).abs()                       # [100,10]
        pred = d.argmin(1)
        bad += int((pred != y).sum())
        # distance to the nearest Voronoi boundary of the correct prototype
        mid = 0.5 * (code[None, :] + code[y][:, None])
        gapd = (z[:, None] - mid).abs()
        gapd[torch.arange(100, device=dev), y] = float("inf")
        mrg = gapd.min(1).values
        k = int(mrg.argmin())
        if float(mrg[k]) < worst:
            worst = float(mrg[k]); worst_case = (int(a[k]), int(b[k]), cin)
    return bad, worst, worst_case


def certify(p, n=8, verbose=True):
    ok, slack, cls, u = bank_report(p)
    delta, wpat = attention_report(p, n=n)
    bad, margin, wcase = readout_report(p)
    e = p["e"][0]
    amp = float(e.abs().sum())
    bound = amp * delta
    verdict = ok and bad == 0 and bound < margin and slack > 0
    if verbose:
        print(f"  part 1 bank      : exact class split={ok}  min saturation slack={slack:.4f}")
        print(f"  part 2 attention : max carry deviation delta={delta:.3e} over 3^{n}"
              f"={3**n} patterns (worst {wpat})")
        print(f"  part 3 read-out  : wrong decodings={bad}/200  min boundary margin={margin:.5f}"
              f"  (worst {wcase})")
        print(f"  combine          : |e|_1 * delta = {bound:.3e}  <  margin {margin:.5f} ?"
              f"  safety x{margin / max(bound, 1e-300):.3g}")
        print(f"  VERDICT: {'PROVEN CORRECT on the whole domain' if verdict else 'NOT PROVEN'}")
    return verdict, dict(bank_ok=ok, slack=slack, delta=delta, margin=margin,
                         bound=bound, bad=bad, safety=margin / max(bound, 1e-300))
