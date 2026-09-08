"""Warm-start remapping between two architecture configs.

The parameter cascade shrinks one architectural element at a time and retrains
after every cut.  This module turns a parent checkpoint into a sensible
starting point for the child.  Two things happen:

  * exact gauge absorption -- several of the things we cut are pure gauge.  The
    nearest-prototype readout -(qv - code_d)^2 is invariant under
    qv -> w qv + b together with code -> w code + b, so a 1x1 readout matrix,
    a readout bias, a logit scale, the attention output scale and a residual
    write bias on the readout axis can all be folded into the code table with
    no change to any prediction.  Those cuts start the child at the parent's
    exact function.

  * axis remapping -- when a read/write range or d_model changes, a weight is
    expanded to full residual width using the parent's range and re-sliced with
    the child's, so every surviving coefficient keeps its meaning.  This part is
    approximate by construction (a narrowed range really does drop terms) and is
    what the retraining after each rung is for.
"""
import torch

# name -> (kind, cfg_key, ffn_key); kind says how the weight meets the residual
#   "read"   last axis (after transpose) indexes the residual over cfg[key]
#   "write"  last axis (after transpose) indexes the residual over cfg[key]
SPEC = {
    "f1_w":  ("read", "f1_in", "f1"),
    "f1_b":  (None, None, "f1"),
    "f1_o":  ("write", "f1_out", "f1"),
    "f1_ob": ("write", "f1_out", None),
    "w_q":   ("read", "qk_in", None),
    "w_k":   ("read", "qk_in", None),
    "w_v":   ("read", "v_in", None),
    "w_o":   ("write", "o_out", None),
    "f2_w":  ("read", "f2_in", "f2"),
    "f2_b":  (None, None, "f2"),
    "f2_o":  ("write", "f2_out", "f2"),
    "f2_ob": ("write", "f2_out", None),
    "r_w":   ("read", "r_in", None),
}

# which dim of the stored tensor indexes residual axes
AXIS = {"f1_w": 1, "f1_o": 1, "f1_ob": 0, "w_q": 0, "w_k": 0, "w_v": 0,
        "w_o": 0, "f2_w": 1, "f2_o": 1, "f2_ob": 0, "r_w": 0}


def code_of(cfg, params):
    """Reconstruct the full (10, cd) code table from stored + pinned entries."""
    cd, nfix = cfg["code_dim"], cfg.get("code_fix", 0)
    p = params["code_p"].double().reshape(-1)
    if nfix and p.numel() == 10 * cd - nfix:
        one = p.new_ones(1)
        if cd == 1:
            p = torch.cat([p.new_zeros(1), one, p] if nfix > 1
                          else [p[:1], one, p[1:]])
        else:
            canon = torch.tensor([1., 0., 0., 1., 0., 0.], dtype=p.dtype)
            p = torch.cat([canon[:nfix], p])
    return p.reshape(10, cd)


def code_to_p(cfg, C):
    """Inverse of `code_of`: drop the entries the config pins."""
    nfix, flat = cfg.get("code_fix", 0), C.reshape(-1)
    if nfix and cfg["code_dim"] == 1 and nfix == 1:
        return torch.cat([flat[:1], flat[2:]])       # entry 1 is the pinned unit
    return flat[nfix:]


def rescale_axis(cfg, p, C, axis, s):
    """Multiply one residual axis by s.  An exact reparametrisation: every
    weight that writes the axis is scaled by s and every weight that reads it by
    1/s.  The code table counts as a write (the embedding) and, under an
    identity readout, as the matching read, so scaling it once does both."""
    if s == 0.0:
        return C
    for name, (kind, key, _) in SPEC.items():
        if name not in p or kind is None:
            continue
        lo, hi = cfg[key]
        if not lo <= axis < hi:
            continue
        j, ax = axis - lo, AXIS[name]
        v = p[name]
        v = v.transpose(ax, -1) if v.dim() > 1 else v
        v[..., j] *= s if kind == "write" else 1.0 / s
        p[name] = v.transpose(ax, -1).contiguous() if p[name].dim() > 1 else v
    if axis < cfg["code_dim"]:
        C = C.clone()
        C[:, axis] *= s
    return C


def _solo_axis(cfg, key):
    """The single residual axis a range covers, or None if it covers more."""
    lo, hi = cfg[key]
    return lo if hi - lo == 1 else None


def _f1_level(cfg, p, C, axis, sums):
    """What the first FFN writes onto `axis`, averaged over the tokens whose two
    digits sum to each of `sums`.  Used to read off the constant part of a
    feature, which is the part a merge has to account for."""
    lo, hi = cfg["f1_in"]
    flo = cfg["f1_out"][0]
    tot = 0.0
    for t in sums:
        a = min(t, 9)
        z = C.new_zeros(cfg["d"])
        z[0] = C[a, 0] + C[t - a, 0]
        h = torch.relu(z[lo:hi] @ p["f1_w"].T + p["f1_b"])
        tot += float(h @ p["f1_o"][:, axis - flo])
    return tot / len(sums)


def absorb(pcfg, pparams, ccfg):
    """Fold the gauge parameters the child drops into the code table.

    Returns (params, code) with `params` missing every absorbed entry.  Only
    applied for the 1-D nearest-prototype readout, where the gauge is the
    scalar affine group and the algebra is exact.
    """
    p = {k: v.double().clone() for k, v in pparams.items()}
    C = code_of(pcfg, pparams)                                   # (10, cd)
    if pcfg["code_dim"] != 1 or pcfg.get("out_mode") != "proto":
        return p, C

    def drops(key, default):
        return bool(pcfg.get(key, default)) and not bool(ccfg.get(key, default))

    def pins(flag):
        return bool(ccfg.get(flag, False)) and not bool(pcfg.get(flag, False))

    cd = pcfg["code_dim"]
    # The answer axis, if both sides read it straight out of the residual.  A
    # readout matrix or bias would break the tie between the prototypes and the
    # embedding -- they are one tensor -- so the axis-0 gauge is only available
    # when the readout is the identity on both sides.
    ra = _solo_axis(pcfg, "r_in")
    if not (pcfg.get("read_id", False) and ccfg.get("read_id", False)
            and tuple(pcfg["r_in"]) == tuple(ccfg["r_in"])
            and "r_w" not in p and "r_b" not in p and ra is not None and ra < cd):
        ra = None

    # A child that gives a scale factor back where the parent had it pinned
    # starts with it at 1, which is the parent's function exactly; leaving it to
    # a random draw would throw the warm start away for no reason.
    for flag, nm, key in (("q_fixed", "w_q", "qk_in"), ("v_fixed", "w_v", "v_in"),
                          ("o_fixed", "w_o", "o_out")):
        if pcfg.get(flag, False) and not ccfg.get(flag, False) and nm not in p:
            lo, hi = pcfg[key]
            p[nm] = torch.ones(hi - lo, dtype=torch.float64)

    def reads(key, axis):
        lo, hi = pcfg[key]
        return axis is not None and lo <= axis < hi

    def free(axis):
        """Can this residual axis be rescaled?  Only while every read of it is
        still a learned weight: a ReLU pinned to unit input weight reads the
        axis as it stands and has nothing left to take the factor up in."""
        return not any(pcfg.get(f"{f}_id_in", False) and reads(f"{f}_in", axis)
                       for f in ("f1", "f2"))

    # --- translation: code -> code + beta ---------------------------------
    # The prototypes move by beta but the embedding, being a sum of two codes,
    # moves by 2 beta, so the readout only stays put if a constant -beta is
    # written back onto the answer axis.  Reading that in reverse: the constant
    # write the child gives up is exactly a shift of the prototypes, paid for
    # by the FFN biases that see the embedding move.
    if ra is not None and drops("f2_ob", False) and "f2_ob" in p \
            and reads("f2_out", ra) and not reads("qk_in", ra) \
            and not reads("v_in", ra):
        beta = float(p["f2_ob"][ra - pcfg["f2_out"][0]])
        C = C + beta
        for f in ("f1", "f2"):
            lo, hi = pcfg[f"{f}_in"]
            if f"{f}_w" in p and lo <= ra < hi:
                p[f"{f}_b"] = p[f"{f}_b"] - p[f"{f}_w"][:, ra - lo] * (2.0 * beta)
        p.pop("f2_ob")

    # --- making room for the key and the value to share an axis -----------
    # Adding the value feature onto the key axis also adds it to the score, and
    # a generator would then outrank a nearer absorber -- the mechanism inverts.
    # So before the merge, shrink the value until its stray contribution to the
    # score is a fraction of one step of the position bias, and hand the factor
    # to the output scale, which is exact and is what that scale is for.
    pq, pv = (_solo_axis(pcfg, k) for k in ("qk_in", "v_in"))
    if (pq is not None and pv is not None and pq != pv and "w_o" in p
            and _solo_axis(ccfg, "qk_in") == _solo_axis(ccfg, "v_in")
            and "alibi" in p and "f1_o" in p):
        # How far to shrink the value is a real trade-off, and both ends of it
        # are visible in the parent's own weights.  Shrink too little and the
        # value's share of the score outranks a step of the position bias, and
        # the attention picks a far generator over a near absorber.  Shrink too
        # much and the *key* -- which comes along as well, and is only flat to
        # within its own drift -- gets that drift multiplied back up on the
        # answer axis, where a fraction of a code step is already a wrong digit.
        # Balance the two.
        sums = [t for t in range(19) if t != 9]
        lv = lambda ax: [_f1_level(pcfg, p, C, ax, [t]) for t in sums]
        ky, vl = lv(pq), lv(pv)
        bq = abs(float(p.get("b_q", torch.zeros(())).reshape(())))
        drift, spread = max(ky) - min(ky), max(vl) - min(vl)
        qmax = max(abs(k + bq) for k in ky) + bq
        gap = float(C.reshape(-1).sort()[0].diff().abs().mean())
        A = abs(float(p["alibi"].reshape(())))
        c = min(1.0, (drift * A / max(gap * spread * qmax, 1e-12)) ** 0.5)
        C = rescale_axis(pcfg, p, C, pv, c)
        p["w_o"] = p["w_o"] / c

        # The key rides in on the value as well, and at every position the
        # attention can pick -- every non-nine -- it sits at the same level, so
        # what reaches the answer axis is one constant.  A constant there is the
        # code's origin: shift the prototypes to meet it and let the two FFN
        # biases absorb the shift the embedding sees.
        S = float(p["w_o"].reshape(())) * (sum(ky) / len(ky))
        C = C - S
        for f, k in (("f1", 2.0), ("f2", 1.0)):
            lo, hi = pcfg[f"{f}_in"]
            if f"{f}_w" in p and ra is not None and lo <= ra < hi:
                p[f"{f}_b"] = p[f"{f}_b"] + (k * S) * p[f"{f}_w"][:, ra - lo]

    # --- residual-axis rescalings -----------------------------------------
    # The answer axis can only be rescaled while every write onto it can be
    # rescaled too, which stops being true as soon as the attention output
    # scale is pinned.  So its one degree of freedom is spent on the logit
    # scale when the value scale has an axis of its own, and on the value and
    # output scales otherwise.
    qa, va, oa = (_solo_axis(pcfg, k) for k in ("qk_in", "v_in", "o_out"))
    a0 = ra is not None and ("w_o" in p or not reads("o_out", ra))

    def resc(axis, s):
        nonlocal C, a0
        if axis is None or abs(s) < 1e-8 or not free(axis):
            return False
        if axis < cd and not (axis == ra and a0):
            return False
        if axis == ra:
            a0 = False
        C = rescale_axis(pcfg, p, C, axis, s)
        return True

    # The logit scale leaves without a trace: it multiplies every logit at a
    # position by one positive number, so no argmax moves.  It has an exact form
    # too -- it is s^2 for a rescaling of the answer axis -- but spending the
    # axis on it is a trap: the child comes out correct and flat, its prototypes
    # squeezed into a tenth of the margin the loss reads, and the first
    # optimiser step throws the solution away.  The axis is worth more as a unit
    # for the code (see below), so take the cheap route here.
    if drops("logit_scale", True) and "lsc" in p:
        p.pop("lsc")

    def resc_qk(s):
        """Rescale the score axis.  Where the value shares it -- the two-axis
        layout, one feature serving as both key and value -- the value comes
        along for the ride, and only an unpinned output scale can put it back."""
        if qa is None or (qa == va and "w_o" not in p):
            return False
        if not resc(qa, s):
            return False
        if qa == va:
            p["w_o"] = p["w_o"] / s
        return True

    # The attention's three scalar scales are redundant: the query axis and one
    # more axis can each be rescaled, and only the product of the value and
    # output scales is observable.  Spend them in that order and all three go.
    if pins("q_fixed") and "w_q" in p and resc_qk(float(p["w_q"].reshape(()))):
        p.pop("w_q")

    s_v = float(p["w_v"].reshape(())) if "w_v" in p and va is not None else 1.0
    s_o = float(p["w_o"].reshape(())) if "w_o" in p and oa is not None else 1.0
    if pins("v_fixed") and "w_v" in p and va is not None and va != qa \
            and va >= cd:
        if "w_o" in p and oa is not None:       # only the product is observable
            p["w_v"] = p["w_v"] * p["w_o"].reshape(())
            p.pop("w_o")
            a0 = a0 and not reads("o_out", ra)
        if resc(va, float(p["w_v"].reshape(()))):
            p.pop("w_v")
    elif pins("o_fixed") and oa is not None and "w_o" in p \
            and (pins("v_fixed") or "w_v" not in p) \
            and oa == ra and resc(oa, 1.0 / (s_v * s_o)):
        p.pop("w_o")
        p.pop("w_v", None)

    # Pinning the relative-position slope multiplies the whole score by the
    # ratio the pin asks for, which the key axis can undo on the content term --
    # the score is quadratic in that axis, so the axis takes the square root.
    # Only the ratio is undone: what survives is a softmax temperature of the
    # same factor, so the pin is exact in the limit where the attention has
    # already saturated, which is where a trained one sits.  That makes the
    # direction matter.  Pinning the slope *above* what the parent learned
    # sharpens the attention it hands down; pinning it below blurs it, and a
    # blurred pick reads a blend of two values instead of one.
    if pcfg.get("alibi_fix") is None and ccfg.get("alibi_fix") is not None \
            and "alibi" in p and qa is not None and qa >= cd:
        a = float(p["alibi"].reshape(()))
        s = abs(ccfg["alibi_fix"] / a) ** 0.5 if abs(a) > 1e-8 else 0.0
        if abs(a) > 1e-8 and a * ccfg["alibi_fix"] > 0 and resc_qk(s):
            for nm, f in (("b_q", s), ("b_k", s), ("self_bias", s * s)):
                if nm in p:
                    p[nm] = p[nm] * f
        p.pop("alibi")

    # --- the unit of the answer axis --------------------------------------
    def scale_answer(s):
        """Rescale the answer axis.  When the attention's output scale is
        already pinned its share of that axis has to be scaled upstream, on the
        value axis, which works as long as nothing in between reads the value
        with a weight of its own."""
        nonlocal C
        via_v = (va is not None and va != ra and va != qa and free(va)
                 and "w_v" not in p and "w_o" not in p)
        if ra is None or cd != 1 or abs(s) < 1e-8 or not free(ra):
            return False
        if not ("w_o" in p or not reads("o_out", ra) or via_v):
            return False
        C = rescale_axis(pcfg, p, C, ra, s)
        if via_v:
            C = rescale_axis(pcfg, p, C, va, s)
        return True

    # Pinning entry 1 to 1 names that unit.  (The origin is not free here -- the
    # constant write that would move it was already spent absorbing `f2_ob` --
    # so entry 0 is pinned, when it is pinned at all, as a normalisation the
    # child has to retrain under rather than an identity.)
    nfix_c, nfix_p = ccfg.get("code_fix", 0), pcfg.get("code_fix", 0)
    if nfix_c > nfix_p and cd == 1 and float(C[1, 0]) != 0:
        scale_answer(1.0 / float(C[1, 0]))

    # Where the unit is left free, set it so neighbouring prototypes sit about
    # 1 apart.  The readout scores a digit by -(qv - code)^2, so that spacing is
    # the margin the loss actually sees; a code squeezed by an absorbed logit
    # scale is just as correct and completely untrainable.
    elif nfix_c == 0 and cd == 1:
        g = float(C.reshape(-1).sort()[0].diff().pow(2).mean().sqrt())
        if not 0.8 < g < 1.25 and g > 1e-8:
            scale_answer(1.0 / g)

    # A ReLU reading one axis can always be rewritten with unit input weight,
    # exactly, as long as that weight is positive; a unit facing the other way
    # has no such form and is handed to the child switched off, for it to find
    # a new use for.
    for f in ("f1", "f2"):
        if not (ccfg.get(f"{f}_id_in", False) and f"{f}_w" in p
                and pcfg[f"{f}_in"][1] - pcfg[f"{f}_in"][0] == 1):
            continue
        w = p.pop(f"{f}_w").reshape(-1)
        good = w > 1e-6
        p[f"{f}_b"] = torch.where(good, p[f"{f}_b"] / w.clamp(min=1e-6),
                                  p[f"{f}_b"])
        p[f"{f}_o"] = p[f"{f}_o"] * torch.where(good, w, torch.zeros_like(w)) \
            .reshape((-1,) + (1,) * (p[f"{f}_o"].dim() - 1))

    return p, C


def _ffn_inputs(cfg, params, n=8192, seed=11):
    """What each FFN actually sees, taken from the parent's own forward pass.

    Which units a shrinking bank should keep is a question about the function
    the bank realises on real inputs, not about the size of any one weight, so
    the inputs have to come from somewhere.  They come from here."""
    import data, probe
    m = load(cfg, params)[0].double()
    d = torch.cat([data.eval_set(n // 2, "cpu", seed, data.UNIFORM),
                   data.eval_set(n // 2, "cpu", seed + 1, data.HARD)])
    with torch.no_grad():
        mid = probe.run(m, data.tokens(d))[2]
    return {k: v.reshape(-1, v.shape[-1]).double() for k, v in mid.items()}


def _bank(cfg, p, f):
    """(W, b, O) for an FFN, with the read weights the config does not learn
    written out explicitly so every bank has the same three pieces."""
    W, k = p.get(f"{f}_w"), cfg[f"{f}_in"][1] - cfg[f"{f}_in"][0]
    if W is None:
        W = (torch.tensor(cfg["f1_signs"]).reshape(-1, 1)
             if cfg.get(f"{f}_fixed_in", False) else torch.ones(cfg[f], k))
    return W.double(), p[f"{f}_b"].double(), p[f"{f}_o"].double()


def _fit(H, Y, const=False, ridge=1e-9):
    """Output weights that best reproduce Y from the hidden activations H.

    With `const`, a free constant is fitted alongside them and returned
    separately: a constant on a feature axis is not part of the feature, and
    the model has somewhere cheaper to keep it (see `_uncentre`)."""
    A = torch.cat([H, H.new_ones(H.shape[0], 1)], 1) if const else H
    G = A.T @ A
    G = G + torch.eye(G.shape[0], dtype=G.dtype) * (ridge * G.diagonal().mean())
    O = torch.linalg.solve(G, A.T @ Y)
    err = float((A @ O - Y).pow(2).mean())
    return (O[:-1], O[-1], err) if const else (O, O.new_zeros(Y.shape[1]), err)


def _pool(W, b):
    """Every unit of a bank in both orientations.

    relu(wz + b) and relu(-wz - b) sit at the same knee and are not
    interchangeable: a down-facing unit equals its up-facing twin plus a linear
    term, so which way a bank's units face decides what linear parts it can
    cancel.  A shrinking bank has to be free to change its mind about that --
    gradient descent cannot, since flipping a unit means taking its input weight
    through zero, where the unit is dead and there is no gradient to follow."""
    return torch.cat([W, -W]), torch.cat([b, -b])


def pick_units(H, Y, q, const=False, cap=400):
    """Which q units of a pool to keep, judged by the function they can realise
    together rather than one at a time.

    Dropping the unit with the smallest output weight is the obvious rule and
    the wrong one: the units of a trained bank overlap, so what a subset is
    worth depends on what the survivors can be refitted to do once it is gone.
    Try the subsets, refit each, keep the best.  Exhaustively while that is
    cheap, otherwise by dropping one unit at a time -- always the one whose
    absence the rest can best cover for."""
    import itertools, math
    P = H.shape[1]
    if math.comb(P, q) <= cap:
        cands = [list(s) for s in itertools.combinations(range(P), q)]
        best = min((_fit(H[:, s], Y, const) + (s,) for s in cands),
                   key=lambda r: r[2])
        return best[3], best[0], best[1], best[2]
    live, out = list(range(P)), None
    while len(live) > q:
        out = min((_fit(H[:, [j for j in live if j != i]], Y, const) + (i,)
                   for i in live), key=lambda r: r[2])
        live = [j for j in live if j != out[3]]
    return live, out[0], out[1], out[2]


def _uncentre(cfg, p, C, f, c):
    """Put back, elsewhere, the constant the refit took off an FFN's outputs.

    A constant on the key axis is a constant added to every query and every key,
    which is what the query bias already is.  A constant on the value axis is
    added to every position's attention output, since the weights sum to one,
    and so shows up as a fixed offset on the answer axis -- which is where the
    code table's origin lives.  Moving the code moves what the first FFN reads,
    so its knees move with it, and the answer axis moves under the second FFN,
    so its knees move too; done together the three are exact, and the child
    starts on the parent's function with a feature that is a shape and nothing
    else.  A shape is what three ReLUs can fit; a shape on a pedestal is not.

    Where the key and the value share one axis the same constant does both jobs
    at once, and the offset it leaves on the answer axis is the shared feature's
    constant times the attention's output scale.

    Returns the shifted code table, or None if this layout has no free place to
    put the constant and it has to stay where it is.
    """
    if f != "f1" or not (cfg.get("q_fixed") and cfg.get("v_fixed")
                         and cfg.get("q_bias") and cfg["code_dim"] == 1
                         and not cfg.get("code_fix")):
        return None
    lo, n = cfg["f1_out"][0], c.numel()
    qk, vv, oa = (_solo_axis(cfg, k) for k in ("qk_in", "v_in", "o_out"))
    ra = _solo_axis(cfg, "r_in")
    if qk is None or vv is None or oa is None or oa != ra or "b_q" not in p:
        return None
    if cfg.get("o_fixed"):
        w_o = 1.0
    elif "w_o" in p and p["w_o"].numel() == 1:
        w_o = float(p["w_o"].reshape(-1)[0])
    else:
        return None
    ck = float(c[qk - lo]) if lo <= qk < lo + n else 0.0
    cv = float(c[vv - lo]) if lo <= vv < lo + n else 0.0
    p["b_q"] = p["b_q"] + ck
    d = w_o * cv                                  # the offset reaching the answer
    if d:
        C = C + d                                     # the code's origin moves
        p["f1_b"] = p["f1_b"] - 2 * d * p["f1_w"].reshape(-1)     # z moves twice
        if "f2_b" in p:                            # the answer axis moves once
            w2 = p.get("f2_w", torch.ones_like(p["f2_b"])).reshape(-1)
            p["f2_b"] = p["f2_b"] - d * w2
    return C


def shrink_ffns(pcfg, pparams, ccfg):
    """Bring the parent's FFN banks down to the child's unit counts.

    This happens before any gauge absorption, so what is being matched is the
    parent's own function; the child then retrains from it like every other
    rung.  Returns a parent config and params whose banks are already the
    child's size, so the rest of the remap has nothing left to drop."""
    need = [f for f in ("f1", "f2")
            if pcfg.get(f) and ccfg.get(f) and ccfg[f] < pcfg[f]]
    if not need:
        return pcfg, pparams
    X = _ffn_inputs(pcfg, pparams)
    cfg, p = dict(pcfg), dict(pparams)
    C = code_of(pcfg, pparams)
    for f in need:
        x = X.get(f"{f}_in")
        if x is None:
            continue
        W, b, O = _bank(pcfg, pparams, f)
        Y = torch.relu(x @ W.T + b) @ O
        # A bank whose read weights are not learned cannot be turned round: the
        # child would have to store the sign, which is the parameter the config
        # just said it does not have.
        Wp, bp = _pool(W, b) if f"{f}_w" in pparams else (W, b)
        H = torch.relu(x @ Wp.T + bp)
        for const in (True, False):
            S, O2, c, err = pick_units(H, Y, ccfg[f], const=const)
            q = dict(p, **{f"{f}_w": Wp[S], f"{f}_b": bp[S]})
            Cn = _uncentre(pcfg, q, C, f, c) if const else C
            if Cn is not None:
                C, p = Cn, q
                p[f"{f}_o"] = O2
                break
        cfg[f] = ccfg[f]
        if f"{f}_w" not in pparams:
            p.pop(f"{f}_w")            # the child reads the axis as it stands
    p = {k: v.float() if torch.is_tensor(v) else v for k, v in p.items()}
    p["code_p"] = C.reshape(-1).float()
    cfg["code_fix"] = 0
    return cfg, p


def keep_units(params, ffn, n):
    """Which hidden units of an FFN to carry over when it shrinks: the ones
    whose output weights are largest, since a unit that writes nothing to the
    residual contributes nothing regardless of how it fires."""
    o = params[f"{ffn}_o"]
    return torch.argsort(o.reshape(o.shape[0], -1).pow(2).sum(1), descending=True)[:n]


def axis_map(pcfg, ccfg):
    """Where each parent residual axis lands in the child, by role.

    Axes are named by what reads them -- answer, key, value -- so a layout that
    gives the key and the value an axis each maps onto the layout that shares
    one between them, and the two writes are added rather than one being thrown
    away.  That is exact exactly when the two features do not overlap, which is
    the case worth warm-starting from: the key marks the nines, the value marks
    the places that carry out, and no place is both.
    """
    m = {i: i for i in range(min(pcfg["d"], ccfg["d"]))}
    for key in ("r_in", "qk_in", "v_in", "o_out"):
        pa, ca = _solo_axis(pcfg, key), _solo_axis(ccfg, key)
        if pa is not None and ca is not None:
            m[pa] = ca
    return m


def remap(pcfg, pparams, ccfg):
    """Parent params -> child params, honouring axis ranges and layer widths."""
    pcfg, pparams = shrink_ffns(pcfg, pparams, ccfg)
    params, C = absorb(pcfg, pparams, ccfg)
    dmax = max(pcfg["d"], ccfg["d"])
    amap = axis_map(pcfg, ccfg)
    keep = {f: keep_units(params, f, min(pcfg[f], ccfg[f]))
            for f in ("f1", "f2") if pcfg[f] and ccfg[f] and pcfg[f] != ccfg[f]}
    out = {}
    for name, v in params.items():
        if name == "code_p":
            if ccfg["code_dim"] == pcfg["code_dim"]:
                out[name] = code_to_p(ccfg, C).float().contiguous()
            continue
        kind, key, hid = SPEC.get(name, (None, None, None))
        v = v.clone()
        if kind is not None and key in pcfg and key in ccfg:
            ax = AXIS[name]
            tr = v.dim() > 1
            v = v.transpose(ax, -1) if tr else v
            lo, hi = pcfg[key]
            full = v.new_zeros(v.shape[:-1] + (dmax,))
            full[..., lo:hi] = v[..., :hi - lo]
            clo, chi = ccfg[key]
            out_v = v.new_zeros(v.shape[:-1] + (chi - clo,))
            for j in range(dmax):
                t = amap.get(j, j) - clo
                if 0 <= t < chi - clo:
                    if kind == "write":
                        out_v[..., t] += full[..., j]
                    elif lo <= j < hi:
                        out_v[..., t] = full[..., j]
            v = out_v.transpose(ax, -1).contiguous() if tr else out_v.contiguous()
        if hid is not None and hid in keep:
            n = keep[hid].numel()
            v = v[keep[hid]]
            if ccfg[hid] > n:
                v = torch.cat([v, v.new_zeros((ccfg[hid] - n,) + tuple(v.shape[1:]))], 0)
        out[name] = v.float()
    return out


def load(cfg, params, strict=True):
    """Build an Adder and fill it from a param dict; report what was missing."""
    from model_src import Adder
    m = Adder(cfg)
    sd = m.state_dict()
    miss = []
    for k in sd:
        v = params.get(k)
        if v is not None and v.numel() == sd[k].numel():
            sd[k] = v.float().reshape(sd[k].shape)
        else:
            miss.append(k)
    if miss and strict:
        raise KeyError(f"cannot fill {miss}")
    m.load_state_dict(sd)
    return m, miss


def check(pcfg, pparams, ccfg, cparams, n=4096, seed=3):
    """How closely the remapped child reproduces the parent's predictions."""
    pm, _ = load(pcfg, pparams)
    cm, miss = load(ccfg, cparams, strict=False)
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 10, (n, pcfg["T"], 2), generator=g)
    with torch.no_grad():
        a, b = pm(x)[:, 1:].argmax(-1), cm(x)[:, 1:].argmax(-1)
    return float((a == b).all(-1).float().mean()), miss
