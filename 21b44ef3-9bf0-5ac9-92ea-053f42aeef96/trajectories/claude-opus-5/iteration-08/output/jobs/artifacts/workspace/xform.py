"""Exact re-writes of a trained checkpoint.

Every op here is a change of coordinates or a removal of something already
exactly zero: the function the network computes is unchanged, only its
parameterisation shrinks.  Each op is verified in float64 against the original
before being written out, and every rung is retrained afterwards, so the shipped
weights are always training output.

  python xform.py --ckpt in.pt --out out.pt --ops drop,pinrow,signin,opin
"""
import argparse, json
import torch
import lib

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def probe(n=512, places=(4, 8, 12)):
    g = torch.Generator(device=DEV).manual_seed(4242)
    out = []
    for np_ in places:
        a, b = lib.sample_digits(n, np_, DEV, g)
        out.append(lib.pad_places(a, b))
    return out


def logits64(p, cfg, pr):
    p64 = {k: v.double() for k, v in p.items()}
    return [lib.fwd(p64, cfg, pa, pb) for pa, pb in pr]


# Ops whose precondition is established by an anneal (rather than by exact
# equality of stored numbers) hold only to the precision the weights are stored
# in, so they are checked for arg-max preservation at float-noise tolerance
# instead of bit-exactness.  Every rewrite is followed by retraining regardless.
TOL = dict(share=1e-4)


def check(p0, cfg0, p1, cfg1, pr, tol=1e-9):
    l0, l1 = logits64(p0, cfg0, pr), logits64(p1, cfg1, pr)
    worst, agree = 0.0, 1.0
    for x, y in zip(l0, l1):
        worst = max(worst, float((x - y).abs().max() / x.abs().max()))
        agree = min(agree, float((x.argmax(-1) == y.argmax(-1)).double().mean()))
    assert worst < tol and agree == 1.0, f"rewrite not exact: rel={worst:.3e} argmax_agree={agree}"
    return worst


# ------------------------------------------------------------------- ops ----

def op_drop(p, cfg, arg):
    """Remove bank units and residual axes whose read/write weights are all zero."""
    cfg = dict(cfg)
    p = dict(p)
    kd = ["o_free"] if cfg["kv"] in ("share", "one") else ["k_o", "v_o"]
    # with o_pin, o_free omits unit 0's read-out, which is the pinned 1.0
    off = 1 if (cfg["kv"] in ("share", "one") and cfg["o_pin"]) else 0
    def reads(u):
        if u < off:
            return True                                   # the pinned read-out is 1.0
        return any(p[n][:, u - off].abs().max() > 0 for n in kd)
    keep1 = [u for u in range(cfg["U"])
             if reads(u) or (cfg["tie"] and p["p_out"][:, u].abs().max() > 0)]
    if len(keep1) < cfg["U"]:
        assert all(u in keep1 for u in range(off)), "unit 0 carries the pinned read-out"
        for n in kd:
            p[n] = p[n][:, [u - off for u in keep1[off:]]]
        p["b1"] = p["b1"][:, keep1]
        if cfg["f1_in"] == "free":
            p["w1"] = p["w1"][:, :, keep1]
        if cfg["tie"]:
            p["p_out"] = p["p_out"][:, keep1]
        cfg["U"] = len(keep1)
        if cfg["tie"]:
            cfg["U2"] = cfg["U"]
    if not cfg["tie"]:
        keep2 = [u for u in range(cfg["U2"]) if p["p_out"][:, u].abs().max() > 0]
        if len(keep2) < cfg["U2"]:
            p["b2"] = p["b2"][:, keep2]
            p["p_out"] = p["p_out"][:, keep2]
            if cfg["f2_in"] == "free":
                p["w2"] = p["w2"][:, :, keep2]
            cfg["U2"] = len(keep2)
    keepc = [c for c in range(cfg["C"]) if p["code_free"][:, :, c].abs().max() > 0]
    if len(keepc) < cfg["C"]:
        assert 0 in keepc, "axis 0 carries the pinned row / sign reads"
        p["code_free"] = p["code_free"][:, :, keepc]
        p["p_out"] = p["p_out"][:, :, keepc]
        if cfg["kv"] == "share":
            p["w_v"] = p["w_v"][:, keepc]
        else:
            p["v_o"] = p["v_o"][:, :, keepc]
        if cfg["f1_in"] == "free":
            p["w1"] = p["w1"][:, keepc]
        if cfg["f2_in"] == "free" and not cfg["tie"]:
            p["w2"] = p["w2"][:, keepc]
        cfg["C"] = len(keepc)
    return p, cfg


def op_share(p, cfg, arg):
    """kv split -> share, valid when v_o is exactly parallel to k_o."""
    assert cfg["kv"] == "split"
    p, cfg = dict(p), dict(cfg)
    k, v = p["k_o"], p["v_o"]                                   # (E,U), (E,U,C)
    kn = k.norm(dim=1, keepdim=True)
    kh = k / kn
    c = torch.einsum("eu,euc->ec", kh, v)                       # (E,C)
    resid = (v - kh[..., None] * c[:, None, :]).abs().max()
    assert resid < 1e-6 * v.abs().max(), f"value axis not parallel to key axis ({resid:.2e})"
    p["o_free"] = k
    p["b_q"] = torch.ones(k.shape[0], device=k.device, dtype=k.dtype)
    p["w_v"] = c / kn
    del p["k_o"], p["v_o"]
    cfg["kv"] = "share"
    return p, cfg


def op_tie(p, cfg, arg):
    assert not cfg["tie"] and cfg["U2"] == cfg["U"]
    p, cfg = dict(p), dict(cfg)
    if cfg["f2_in"] == "free":
        assert (p["w2"] - p["w1"]).abs().max() == 0
        del p["w2"]
    else:
        assert float(cfg["f2_sign"]) == float(cfg["f1_sign"])
    assert (p["b2"] - p["b1"]).abs().max() == 0
    del p["b2"]
    cfg["tie"] = True
    return p, cfg


def op_lamfix(p, cfg, arg):
    assert cfg["lam_learn"]
    p, cfg = dict(p), dict(cfg)
    assert (p["lam"] - float(cfg["lam"])).abs().max() == 0, p["lam"]
    del p["lam"]
    cfg["lam_learn"] = False
    return p, cfg


def op_pinrow(p, cfg, arg):
    """Use the residual scale (and, at C=2, rotation) freedom to pin one code row
    to e_0, so that row costs no parameters."""
    assert cfg["pin_row"] < 0
    p, cfg = dict(p), dict(cfg)
    code = p["code_free"]
    E, _, C = code.shape
    r = int(arg) if arg else None
    if r is None:
        cand = code[:, :, 0] if C == 1 else code.norm(dim=2)
        r = int((cand.min(0).values * (code[:, :, 0].min(0).values > 0)).argmax()) if C == 1 \
            else int(cand.min(0).values.argmax())
    c = code[:, r, :]                                           # (E,C)
    if cfg["f1_in"] == "sign" or cfg["f2_in"] == "sign":
        assert C == 1 and (c[:, 0] > 0).all(), "sign-pinned banks need a positive scale"
    if C == 1:
        s = 1.0 / c[:, 0]                                       # (E,)
        M = s.reshape(E, 1, 1)
        Minv_T = (1.0 / s).reshape(E, 1, 1)
    else:
        t = c.norm(dim=1)
        cs, sn = c[:, 0] / t, c[:, 1] / t
        R = torch.stack([torch.stack([cs, sn], 1), torch.stack([-sn, cs], 1)], 1)   # (E,2,2)
        M = R / t.reshape(E, 1, 1)
        Minv_T = R * t.reshape(E, 1, 1)                          # (M^-1)^T = R * t
    p["code_free"] = torch.einsum("edc,ekc->ekd", M, code)
    p["p_out"] = torch.einsum("edc,euc->eud", M, p["p_out"])
    if cfg["kv"] == "share":
        p["w_v"] = torch.einsum("edc,ec->ed", M, p["w_v"])
    else:
        p["v_o"] = torch.einsum("edc,euc->eud", M, p["v_o"])
    if cfg["f1_in"] == "free":
        p["w1"] = torch.einsum("edc,ecu->edu", Minv_T, p["w1"])
    else:
        p["b1"] = p["b1"] * M.reshape(E, 1)
        if cfg["kv"] == "share":
            p["b_q"] = p["b_q"] / M.reshape(E)
        else:
            p["k_o"] = p["k_o"] / M.reshape(E, 1)
    if not cfg["tie"]:
        if cfg["f2_in"] == "free":
            p["w2"] = torch.einsum("edc,ecu->edu", Minv_T, p["w2"])
        else:
            p["b2"] = p["b2"] * M.reshape(E, 1)
    elif cfg["f1_in"] == "sign":
        pass
    ok = (p["code_free"][:, r] - torch.eye(C, device=code.device, dtype=code.dtype)[0]).abs().max()
    assert ok < 1e-6, ok
    p["code_free"] = torch.cat([p["code_free"][:, :r], p["code_free"][:, r + 1:]], 1)
    cfg["pin_row"] = r
    return p, cfg


def op_signin(p, cfg, arg):
    """Absorb |input weight| of each ReLU unit into its readers, leaving only the
    unit's fixed +-1 orientation.  Exact: relu(w x + b) = |w| relu(sign(w) x + b/|w|)."""
    which = arg or "1"
    p, cfg = dict(p), dict(cfg)
    for w in which.split("+"):
        if w == "1":
            assert cfg["f1_in"] == "free" and cfg["C"] == 1
            m = p["w1"][:, 0, :]                                # (E,U)
            assert (m.abs() > 1e-8).all(), "dead unit in bank 1"
            sg = m.sign()
            assert (sg == sg[:1, :1]).all(), "bank 1 orientation is not uniform"
            p["b1"] = p["b1"] / m.abs()
            if cfg["kv"] == "share":
                p["o_free"] = p["o_free"] * m.abs()
            else:
                p["k_o"] = p["k_o"] * m.abs()
                p["v_o"] = p["v_o"] * m.abs()[..., None]
            if cfg["tie"]:
                p["p_out"] = p["p_out"] * m.abs()[..., None]
            del p["w1"]
            cfg["f1_in"] = "sign"
            cfg["f1_sign"] = float(sg[0, 0])
        else:
            assert cfg["f2_in"] == "free" and cfg["C"] == 1 and not cfg["tie"]
            m = p["w2"][:, 0, :]
            assert (m.abs() > 1e-8).all(), "dead unit in bank 2"
            sg = m.sign()
            assert (sg == sg[:1, :1]).all(), "bank 2 orientation is not uniform"
            p["b2"] = p["b2"] / m.abs()
            p["p_out"] = p["p_out"] * m.abs()[..., None]
            del p["w2"]
            cfg["f2_in"] = "sign"
            cfg["f2_sign"] = float(sg[0, 0])
    return p, cfg


def op_opin(p, cfg, arg):
    """Use the key-axis scale freedom to pin o[0] = 1."""
    assert cfg["kv"] == "share" and not cfg["o_pin"]
    p, cfg = dict(p), dict(cfg)
    o = p["o_free"]
    j = int(arg) if arg else int(o.abs().min(0).values.argmax())
    if j != 0:                                                  # permute that unit to the front
        idx = [j] + [u for u in range(cfg["U"]) if u != j]
        for n in (["o_free", "b1"] + (["p_out"] if cfg["tie"] else [])):
            p[n] = p[n][:, idx]
        if cfg["f1_in"] == "free":
            p["w1"] = p["w1"][:, :, idx]
        o = p["o_free"]
    t = o[:, :1]
    assert (t.abs() > 1e-8).all()
    p["o_free"] = (o / t)[:, 1:]
    p["b_q"] = p["b_q"] * t[:, 0]
    p["w_v"] = p["w_v"] * t
    cfg["o_pin"] = True
    return p, cfg


def _perm_bank1(p, cfg, idx):
    p = dict(p)
    for n in (["o_free", "b1"] if cfg["kv"] == "share" else ["k_o", "b1"]):
        p[n] = torch.gather(p[n], 1, idx)
    if cfg["kv"] == "split":
        p["v_o"] = torch.gather(p["v_o"], 1, idx[..., None].expand(-1, -1, cfg["C"]))
    if cfg["f1_in"] == "free":
        p["w1"] = torch.gather(p["w1"], 2, idx[:, None].expand(-1, cfg["C"], -1))
    if cfg["tie"]:
        p["p_out"] = torch.gather(p["p_out"], 1, idx[..., None].expand(-1, -1, cfg["C"]))
    return p


def op_sortu(p, cfg, arg):
    """Permute ReLU units so the ones that matter least come last.

    A permutation of hidden units is an exact symmetry of the network, and it is
    what makes an index-based unit cut meaningful: warm-started members each have
    their own ordering, so `--shrink u1:7` only means "cut the least useful unit"
    once every member has been sorted.  Importance is the standard deviation,
    over a probe batch, of the unit's contribution to what reads it.
    """
    assert cfg["f1_in"] == "free" and not cfg["o_pin"], "sort before pinning"
    pr = probe()
    with torch.no_grad():
        code = lib.full_code(p, cfg)
        h, g = [], []
        for pa, pb in pr:
            x_ = code[:, pa] + code[:, pb]
            hh = lib._bank(p, cfg, x_, 1)                     # (E,B,P,U)
            h.append(hh.reshape(hh.shape[0], -1, hh.shape[-1]))
            if not cfg["tie"]:
                gg = lib._bank(p, cfg, x_, 2)
                g.append(gg.reshape(gg.shape[0], -1, gg.shape[-1]))
        h = torch.cat(h, 1)
        rd = p["o_free"].abs() if cfg["kv"] == "share" else \
            p["k_o"].abs() + p["v_o"].norm(dim=2)
        imp1 = h.std(1) * rd
        idx1 = imp1.argsort(1, descending=True)
    p = _perm_bank1(p, cfg, idx1)
    cfg = dict(cfg)
    if not cfg["tie"]:
        with torch.no_grad():
            imp2 = torch.cat(g, 1).std(1) * p["p_out"].norm(dim=2)
            idx2 = imp2.argsort(1, descending=True)
        for n in ["b2"]:
            p[n] = torch.gather(p[n], 1, idx2)
        if cfg["f2_in"] == "free":
            p["w2"] = torch.gather(p["w2"], 2, idx2[:, None].expand(-1, cfg["C"], -1))
        p["p_out"] = torch.gather(p["p_out"], 1, idx2[..., None].expand(-1, -1, cfg["C"]))
    return p, cfg


def op_prows(p, cfg, arg):
    """Record which fold rows are non-zero in cfg["p_rows"] and drop the rest.

    Only usable when bank 2 is tied to bank 1: an all-zero p_out row still has to
    be evaluated by op_drop's rules because bank 1 needs the unit for the key.
    """
    assert cfg["tie"] and not cfg["p_rows"]
    p, cfg = dict(p), dict(cfg)
    keep = [u for u in range(cfg["U"]) if p["p_out"][:, u].abs().max() > 0]
    assert len(keep) < cfg["U"], "no zero fold rows"
    p["p_out"] = p["p_out"][:, keep]
    cfg["p_rows"] = keep
    return p, cfg


def op_dropscale(p, cfg, arg):
    """Remove the training-only read-out temperature exp(ls).

    Every logit of a member is divided by the same strictly positive number, so
    the arg-max -- which is the entire inference path -- is unchanged.  Verified
    below as exact proportionality plus arg-max agreement rather than as an
    equality of logits.
    """
    assert cfg["logit_scale"]
    p, cfg = dict(p), dict(cfg)
    p.pop("ls")
    cfg["logit_scale"] = False
    return p, cfg


def op_zerobias(p, cfg, arg):
    """Remove the read-out bias `by` with the C=1 residual scale/shift gauge.

    At C=1 the stream can be re-coordinated by ``code -> u*code + t``.  Because a
    token is ``code[a] + code[b]`` the residual picks up ``2t`` while the read-out
    compares against one code entry, which picks up ``t``; the leftover ``t`` is
    exactly what `by` absorbs, so ``by -> u*by - t`` and two numbers -- ``by`` and
    the pinned code row -- can be fixed at once.  Take ``t = u*by`` and
    ``u = 1/(code[pin] + by)``: `by` becomes 0 and the pinned row stays 1, so the
    bias is a training scaffold that costs nothing in the end.

    ``u`` may be negative.  A negative scale reverses the residual stream, which
    the ReLU banks do not commute with -- but reversing their fixed input signs
    at the same time is an exact symmetry, so the general rewrite splits ``u``
    into a positive scale ``|u|`` that the banks absorb and a sign that flows
    into the bank input signs, `w_v` and `p_out`.  Logits scale by ``u**2 > 0``,
    so the arg-max, and hence every answer, is unchanged.
    """
    assert cfg["C"] == 1 and cfg["y_bias"] and cfg["pin_row"] >= 0
    p, cfg = dict(p), dict(cfg)
    by = p.pop("by")                                        # (E,)
    den = 1.0 + by                                          # code[pin] == 1
    assert (den.abs() > 1e-6).all(), f"shift gauge is singular: code[pin]+by = {den}"
    sg = torch.sign(den)
    assert (sg == sg[:1]).all(), "members disagree on the shift-gauge orientation"
    sigma = float(sg[0])
    u = 1.0 / den
    a, t = u.abs(), u * by
    E = by.shape[0]
    a1, t1 = a.reshape(E, 1), t.reshape(E, 1)
    p["code_free"] = u.reshape(E, 1, 1) * p["code_free"] + t.reshape(E, 1, 1)

    # bank 1 (and, when tied, bank 2 -- both see a shift of 2t)
    if cfg["f1_in"] == "free":
        p["w1"] = sigma * p["w1"]
        p["b1"] = a1 * p["b1"] - 2 * t1 * p["w1"][:, 0, :]
    else:
        s1 = float(cfg["f1_sign"])
        p["b1"] = a1 * p["b1"] - 2 * t1 * sigma * s1
        cfg["f1_sign"] = sigma * s1
    if not cfg["tie"]:
        if cfg["f2_in"] == "free":
            p["w2"] = sigma * p["w2"]
            p["b2"] = a1 * p["b2"] - 2 * t1 * p["w2"][:, 0, :]
        else:
            s2 = float(cfg["f2_sign"])
            p["b2"] = a1 * p["b2"] - 2 * t1 * sigma * s2
            cfg["f2_sign"] = sigma * s2
    else:
        cfg["f2_sign"] = cfg["f1_sign"]     # unread under tie; keep the cfg honest
    if cfg["kv"] == "share":
        p["b_q"] = p["b_q"] / a
        p["w_v"] = sigma * p["w_v"]
    else:
        p["k_o"] = p["k_o"] / a.reshape(E, 1)
        p["v_o"] = sigma * p["v_o"]
    p["p_out"] = sigma * p["p_out"]
    cfg["y_bias"] = False
    return p, cfg


def op_kvone(p, cfg, arg):
    """Make the key scale and the value scale one parameter.

    A place broadcasts a scalar key ``b_q*z`` and hands on a value ``z*w_v``; both
    are read off the same bank-1 axis, so the only thing separating them is the
    two scales.  They are not free of each other -- the value has to shift the
    residual by one code step and the key gap has to stay inside the distance
    penalty -- and with |lam| bigger than the code step a single shared scale can
    satisfy both.  Applied after ``--shrink kvs`` has annealed them together, so
    the numbers are already equal and this only removes the duplicate.
    """
    assert cfg["kv"] == "share" and not cfg["y_bias"], "fix the shift gauge first"
    p, cfg = dict(p), dict(cfg)
    d = float((p["b_q"] - p["w_v"][:, 0]).abs().max() / p["b_q"].abs().max())
    assert d < 1e-6, f"key and value scales are still {d:.2e} apart; anneal further"
    w = p["w_v"].clone()
    w[:, 0] = p.pop("b_q")
    p["w_v"] = w
    cfg["kv"] = "one"
    return p, cfg


def check_prop(p0, cfg0, p1, cfg1, pr, tol=1e-9):
    """Checker for re-writes that rescale the read-out.

    Pinning a code row rescales the whole residual stream, and dropping the
    training-only temperature divides by exp(ls); both multiply every logit of a
    member by one strictly positive number.  The decode is an arg-max, so the
    model's function -- the digit it predicts for every input -- is unchanged,
    even though the logits themselves are not.  That is what is checked here:
    positive proportionality, per member, plus arg-max agreement.
    """
    l0, l1 = logits64(p0, cfg0, pr), logits64(p1, cfg1, pr)
    worst, agree = 0.0, 1.0
    for a, b in zip(l0, l1):
        E = a.shape[0]
        u, v = a.reshape(E, -1), b.reshape(E, -1)
        gam = (u * v).sum(1) / (v * v).sum(1)
        assert (gam > 0).all(), f"read-out rescale is not positive: {gam}"
        worst = max(worst, float((u - gam[:, None] * v).abs().max() / u.abs().max()))
        agree = min(agree, float((a.argmax(-1) == b.argmax(-1)).double().mean()))
    assert worst < tol and agree == 1.0, f"not a positive rescale: rel={worst:.3e} agree={agree}"
    return worst


OPS = dict(drop=op_drop, share=op_share, tie=op_tie, lamfix=op_lamfix,
           pinrow=op_pinrow, signin=op_signin, opin=op_opin, dropscale=op_dropscale,
           prows=op_prows, sortu=op_sortu, zerobias=op_zerobias, kvone=op_kvone)
CHECKERS = dict(dropscale=check_prop, pinrow=check_prop, zerobias=check_prop)


def apply_ops(p, cfg, ops, verbose=True):
    pr = probe()
    for item in ops.split(","):
        if not item:
            continue
        name, _, arg = item.partition(":")
        p1, cfg1 = OPS[name](p, cfg, arg)
        tol = TOL.get(name, 1e-9)
        rel = CHECKERS.get(name, check)(p, cfg, p1, cfg1, pr, tol=tol)
        p, cfg = p1, cfg1
        if verbose:
            kind = "exact" if rel < 1e-9 else "arg-max preserving"
            print(f"  {name:8s} -> {lib.n_params(cfg):3d} params  ({kind}, rel={rel:.1e})  "
                  f"U={cfg['U']} U2={cfg['U2']} C={cfg['C']}")
    return p, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ops", required=True)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--pick", type=int, default=-1,
                    help="operate on a single member; needed for rewrites whose "
                         "result is a cfg-level constant (unit orientations), "
                         "which independently trained members need not share")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    cfg0 = lib.default_cfg(**ck["cfg"])
    sel = slice(args.pick, args.pick + 1) if args.pick >= 0 else slice(0, args.keep)
    p32 = {n: t[sel].to(DEV) for n, t in ck["params"].items()}
    print(f"{args.ckpt}: {lib.n_params(cfg0)} params, scores {[round(s,5) for s in ck['scores'][sel]]}")
    # the rewrites are done in float64 so that "exact" means exact rather than
    # float32 round-off amplified by the squared-distance read-out
    p, cfg = apply_ops({n: t.double() for n, t in p32.items()}, cfg0, args.ops)
    p = {n: t.float() for n, t in p.items()}
    rel = check_prop(p32, cfg0, p, cfg, probe(), tol=1e-4)
    print(f"  float32 round trip vs original: positive-rescale rel={rel:.2e}, arg-max identical")
    torch.save(dict(cfg=cfg, params={n: t.cpu() for n, t in p.items()},
                    scores=ck["scores"][sel], uniform=ck.get("uniform", [])[sel],
                    chain=ck.get("chain", [])[sel]), args.out)
    print("saved", args.out, json.dumps(cfg))


if __name__ == "__main__":
    main()
