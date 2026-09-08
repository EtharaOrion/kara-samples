"""Reduce a trained phase-2 member to the 12-parameter shipping form.

Two kinds of move are used, and they are kept strictly separate:

  * exact rewrites  - re-express the same function in different coordinates:
      flip a negative-slope clamp unit (clamp(-y)=1-clamp(y+1)), permute units,
      fold q into the key weights, renormalise the value stream into the carry
      write weights, flip the sign of the code, rescale the code so the carry
      write is 1, and set the read-out temperature to 1 (argmax is invariant to a
      positive logit scale).  Each is checked numerically against the parent.

  * constant substitution - replace saturation/sharpness constants that carry no
      fitted information (clamp slope, key read-off, attention temperature and
      recency slope, the value/residual offsets) with fixed values, holding the
      learned knee positions.  This changes the function, so it is followed by a
      short polish on real data and then by certify.py, which proves exactness
      over the whole domain.

What is left learned: the 9 free code entries, the 2 bank knees, the fold.
"""
import argparse
import torch

import arch, data, certify

# fixed architectural constants of the shipping form
SLOPE = 8.0          # clamp saturation slope
KEY = 400.0          # key read-off magnitude (transparent notch depth)
LAM = -12.0          # recency slope of the relative-position bias
PMAX = 26            # widths the certificate is issued for


def _one(p):
    return {k: v.clone() for k, v in p.items()}


def agree(p, q, cfgp, cfgq, device, n=8, N=4096, seed=5):
    """max |logit difference| and argmax agreement between two members."""
    g = torch.Generator(device=device).manual_seed(seed)
    ta, tb, _, _, _ = data.batch(N, n, device, g)
    with torch.no_grad():
        A = arch.forward({k: v[None].to(device) for k, v in p.items()}, ta, tb, cfgp)
        B = arch.forward({k: v[None].to(device) for k, v in q.items()}, ta, tb, cfgq)
    return float((A - B).abs().max()), float((A.argmax(-1) == B.argmax(-1)).float().mean())


def canonicalise(p, cfg):
    """Exact rewrites only.  Returns (params, cfg, log)."""
    p = _one(p)
    log = []
    U = cfg["U"]
    # 1. every clamp unit gets a positive slope:  clamp(wx+b) = 1 - clamp(|w|x + 1 - b)
    for j in range(U):
        if float(p["Bw"][0, j]) < 0:
            w, b = float(p["Bw"][0, j]), float(p["bb"][j])
            p["Bw"][0, j] = -w
            p["bb"][j] = 1.0 - b
            p["vb"] = p["vb"] + p["vw"][j]
            p["vw"][j] = -p["vw"][j]
            p["kw"][j] = -p["kw"][j]      # + const on every key: cancels inside softmax
            log.append(f"flip unit {j}")
    # 2. fold the query scale into the key weights
    p["kw"] = p["kw"] * p["q"]
    p["q"] = torch.ones_like(p["q"])
    log.append("q -> 1")
    # 3. order units by knee position
    knee = (-p["bb"] / p["Bw"][0])
    order = torch.argsort(knee)
    for k in ("bb", "kw", "vw"):
        p[k] = p[k][order]
    p["Bw"] = p["Bw"][:, order]
    log.append(f"unit order {order.tolist()}, knees {sorted([round(float(x),4) for x in knee])}")
    return p, cfg, log


def value_normalise(p, cfg):
    """Exact: make the attention value binary {0,1} by moving its affine part into
    the carry write weights and the residual bias."""
    p = _one(p)
    # value on the three classes, using the (post-canonicalisation) unit semantics
    v_abs = float(p["vb"])                                 # both units off
    v_gen = float(p["vb"] + p["vw"].sum())                 # both units on
    vgap = v_gen - v_abs
    assert abs(vgap) > 1e-6, f"degenerate value gap {vgap}"
    p["rb"] = p["rb"] + (p["w1"] + p["w2"]) * v_abs
    p["w1"] = p["w1"] * vgap
    p["w2"] = p["w2"] * vgap
    p["vw"] = torch.tensor([0.0, 1.0], device=p["vw"].device)
    p["vb"] = torch.zeros_like(p["vb"])
    return p, [f"value normalised (v_abs {v_abs:.4f}, gap {vgap:.4f})"]


def scale_gauge(p, cfg):
    """Exact: flip the code sign if needed and rescale so the carry write is 1.
    The read-out is a positive-homogeneous score, so argmax is unchanged."""
    p = _one(p)
    log = []
    if float(p["w1"][0]) < 0:                 # code -> -code
        p["code"] = -p["code"]
        p["Bw"] = -p["Bw"]
        p["w1"] = -p["w1"]
        p["w2"] = -p["w2"]
        p["rb"] = -p["rb"]
        log.append("code sign flipped")
    lam_ = 1.0 / float(p["w1"][0])
    p["code"] = p["code"] * lam_
    p["rb"] = p["rb"] * lam_
    p["w2"] = p["w2"] * lam_
    p["w1"] = torch.ones_like(p["w1"])
    p["Bw"] = p["Bw"] / lam_
    p["ls"] = torch.ones_like(p["ls"])
    log.append(f"scaled by {lam_:.5f} so carry write = 1; ls -> 1")
    return p, log


def substitute(p, cfg, slope=SLOPE, key=KEY, lam=LAM):
    """Constant substitution: hold the learned knees, fix everything else."""
    p = _one(p)
    knee = (-p["bb"] / p["Bw"][0]).clone()
    dev = p["bb"].device
    p["Bw"] = torch.full_like(p["Bw"], slope)
    p["bb"] = -slope * knee
    p["kw"] = torch.tensor([-key, key], device=dev)
    p["vw"] = torch.tensor([0.0, 1.0], device=dev)
    p["vb"] = torch.zeros_like(p["vb"])
    p["rb"] = torch.zeros_like(p["rb"])
    p["q"] = torch.ones_like(p["q"])
    p["lam"] = torch.full_like(p["lam"], lam)
    p["ls"] = torch.ones_like(p["ls"])
    p["w1"] = torch.ones_like(p["w1"])
    return p, [f"knees held at {[round(float(x),5) for x in knee]}; "
               f"slope={slope} key={key} lam={lam}; vb,rb->0; q,w1,ls->1"]


def structure_ok(p, cfg):
    """Does this member implement the intended absorb / transparent / generate split?"""
    with torch.no_grad():
        code = arch.full_code({k: v[None] for k, v in p.items()}, cfg)[0, :, 0].double()
        A = torch.arange(10)
        x = code[A][:, None] + code[A][None, :]
        z = x[..., None] * p["Bw"][0].double() + p["bb"].double()
        u = z.clamp(0, 1)
        s = A[:, None] + A[None, :]
        want0 = (s >= 9).double()
        want1 = (s >= 10).double()
        e0 = float((u[..., 0] - want0).abs().max())
        e1 = float((u[..., 1] - want1).abs().max())
        kw = p["kw"].double()
        ok_key = bool(kw[0] < 0 and (kw[0] + kw[1]) > kw[0])
    return (e0 < 1e-6 and e1 < 1e-6 and ok_key), dict(unit0_err=e0, unit1_err=e1, key_ok=ok_key)


def reduce_member(p, cfg, device="cuda", verbose=True):
    p = {k: v.to(device).double().float() for k, v in p.items()}
    logs = []
    p1, cfg1, l = canonicalise(p, cfg); logs += l
    d, ag = agree(p, p1, cfg, cfg1, device)
    logs.append(f"  after canonicalise: max|dlogit| {d:.3e}, argmax agree {ag:.6f}")
    p2, l = value_normalise(p1, cfg1); logs += l
    d, ag = agree(p, p2, cfg, cfg1, device)
    logs.append(f"  after value_normalise: max|dlogit| {d:.3e}, argmax agree {ag:.6f}")
    p3, l = scale_gauge(p2, cfg1); logs += l
    d, ag = agree(p, p3, cfg, cfg1, device)
    logs.append(f"  after scale_gauge: argmax agree {ag:.6f} (logits rescaled by design)")
    ok, info = structure_ok(p3, cfg1)
    logs.append(f"  structure check: {ok} {info}")
    p4, l = substitute(p3, cfg1); logs += l
    d, ag = agree(p, p4, cfg, cfg1, device)
    logs.append(f"  after substitute: argmax agree {ag:.6f}")
    if verbose:
        for s in logs:
            print(s)
    return p4, cfg1, ok, logs
