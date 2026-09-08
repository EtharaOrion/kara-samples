"""Take a trained parent block down to the twelve-parameter shipped form.

Two kinds of move are used and they are kept strictly apart:

  exact       a re-parameterisation that provably leaves the read-out arg-max
              unchanged: absorbing the query scale into the keys, absorbing the
              value bias into the residual bias, dropping the read-out
              temperature, flipping a clamp unit (clamp(z) = 1 - clamp(1-z)),
              and the two gauge freedoms of this block -- a translation and a
              scale of the code.  Every one is checked numerically in float64.

  substituted a constant of the architecture is replaced by a round chosen
              number (bank gain, key contrast, recency slope, residual bias).
              This does change the function, so it is only an *initialisation*:
              `finetune.py` then retrains the twelve free values under exactly
              those constants, and `certify.py` proves the result is exact over
              the whole input domain rather than on a sample.

Projecting a two-channel parent onto one channel is neither of those: it is a
re-init, so it too is followed by real retraining (`project.py`).
"""
import torch

import ens

SLOPE, KEY, LAM, CARRY = 8.0, 400.0, -12.0, 1.0


def clone(p):
    return {k: v.clone() for k, v in p.items()}


def fwd1(p, ta, tb):
    """Forward a single (non-ensemble) member."""
    return ens.forward({k: v[None] for k, v in p.items()}, ta, tb)[0]


def _agree(p, q, ta, tb, tag):
    la, lb = fwd1(p, ta, tb), fwd1(q, ta, tb)
    return dict(step=tag,
                argmax_same=bool((la.argmax(-1) == lb.argmax(-1)).all()),
                max_logit_drift=float((la - lb).abs().max()))


# ---------------------------------------------------------------- exact moves
def exact_rewrites(p, ta, tb):
    p, log = clone(p), []

    def mark(prev, tag):
        log.append(_agree(prev, p, ta, tb, tag))

    q = clone(p)                                   # 1. query scale -> keys
    p["key_w"] = p["key_w"] * p["q"]
    p["q"] = torch.ones_like(p["q"])
    mark(q, "absorb query scale into keys")

    q = clone(p)                                   # 2. constant value -> rb
    p["rb"] = p["rb"] + p["val_b"] * (p["carry_w"] + p["fold_w"])
    p["val_b"] = torch.zeros_like(p["val_b"])
    mark(q, "absorb value bias into residual bias")

    q = clone(p)                                   # 3. positive logit scale
    p["ls_log"] = torch.zeros_like(p["ls_log"])
    mark(q, "drop read-out temperature (arg-max invariant)")

    q = clone(p)                                   # 4. translation gauge
    t = -p["code"][0].clone()
    p["code"] = p["code"] + t
    p["rb"] = p["rb"] - t
    p["knee"] = p["knee"] - 2.0 * (p["bank_w"] @ t)
    mark(q, "translate so code[0] = 0")

    q = clone(p)                                   # 5. scale gauge
    assert abs(float(p["carry_w"][0])) > 1e-6, "carry write is degenerate"
    c = CARRY / float(p["carry_w"][0])
    for k in ("code", "carry_w", "fold_w", "rb"):
        p[k] = p[k] * c
    p["bank_w"] = p["bank_w"] / c
    mark(q, f"scale so the carry write is {CARRY} (arg-max invariant)")

    q = clone(p)                                   # 6. clamp(z) = 1 - clamp(1-z)
    for u in range(p["bank_w"].shape[0]):
        if float(p["bank_w"][u, 0]) < 0:
            old_val = float(p["val_w"][u])
            p["knee"][u] = 1.0 - p["knee"][u]
            p["bank_w"][u] = -p["bank_w"][u]
            p["key_w"][u] = -p["key_w"][u]
            p["val_w"][u] = -p["val_w"][u]
            p["rb"] = p["rb"] + old_val * (p["carry_w"] + p["fold_w"])
    mark(q, "flip bank units so both rise with the digit sum")

    q = clone(p)                                   # 7. canonical unit order
    theta = -p["knee"] / p["bank_w"][:, 0]
    order = torch.argsort(theta)
    for k in ("bank_w", "knee", "key_w", "val_w"):
        p[k] = p[k][order]
    mark(q, "order bank units by knee position")

    q = clone(p)                                   # 8. orient the value stream
    if float(p["val_w"][0] + p["val_w"][1]) < 0:
        p["val_w"] = -p["val_w"]
        p["carry_w"] = -p["carry_w"]
        p["fold_w"] = -p["fold_w"]
        # carry write is no longer CARRY; re-spend the scale gauge
        c = CARRY / float(p["carry_w"][0])
        for k in ("code", "carry_w", "fold_w", "rb"):
            p[k] = p[k] * c
        p["bank_w"] = p["bank_w"] / c
    mark(q, "orient the value stream so a carry writes positively")
    return p, log


def diagnose(p):
    """What the parent actually learned, in the units the shipped form uses."""
    theta = (-p["knee"] / p["bank_w"][:, 0]).tolist()
    step = float(p["code"][2, 0] - p["code"][1, 0])
    lin = float((p["code"][:, 0] - torch.arange(10, dtype=p["code"].dtype,
                                                device=p["code"].device)
                 * p["code"][1, 0]).abs().max())
    return dict(knee_positions=theta,
                code=[round(float(v), 4) for v in p["code"][:, 0]],
                code_step=step, code_nonlinearity=lin,
                key_w=[round(float(v), 3) for v in p["key_w"]],
                val_w=[round(float(v), 4) for v in p["val_w"]],
                lam=float(p["lam"][0]), rb=float(p["rb"][0]),
                carry_w=float(p["carry_w"][0]), fold_w=float(p["fold_w"][0]))


# -------------------------------------------------------- constant substitution
def substitute(p, ta, tb):
    """Replace the architecture's constants by the round numbers the shipped
    class hard-codes.  Knee positions are held fixed while the gain changes."""
    p, log = clone(p), []

    def mark(prev, tag):
        log.append(_agree(prev, p, ta, tb, tag))

    q = clone(p)                                   # bank gain, knees held fixed
    theta = -p["knee"] / p["bank_w"][:, 0]
    p["bank_w"] = torch.full_like(p["bank_w"], SLOPE)
    p["knee"] = -SLOPE * theta
    mark(q, f"bank gain -> {SLOPE} (knees held)")

    q = clone(p)                                   # value read-out -> (0, 1)
    # A transparent place has key -KEY and is never attended, so only the value
    # at absorbing places (g = 0,0 -> 0) and at generating places (g = 1,1 ->
    # val_w0 + val_w1) is ever read.  Scaling that into the two writes leaves
    # the attended values untouched.
    s = float(p["val_w"][0] + p["val_w"][1])
    assert s > 0, "generating places must carry a positive value"
    p["carry_w"] = p["carry_w"] * s
    p["fold_w"] = p["fold_w"] * s
    p["val_w"] = torch.tensor([0.0, 1.0], dtype=p["val_w"].dtype,
                              device=p["val_w"].device)
    mark(q, "value read-out -> (0,1)")

    q = clone(p)                                   # key read-out -> (-KEY, KEY)
    p["key_w"] = torch.tensor([-KEY, KEY], dtype=p["key_w"].dtype,
                              device=p["key_w"].device)
    mark(q, f"key read-out -> (-{KEY:.0f},{KEY:.0f})")

    q = clone(p)                                   # recency slope
    p["lam"] = torch.full_like(p["lam"], LAM)
    mark(q, f"recency slope -> {LAM}")

    q = clone(p)                                   # residual bias
    p["rb"] = torch.zeros_like(p["rb"])
    mark(q, "residual bias -> 0")

    q = clone(p)                                   # restore the scale gauge
    c = CARRY / float(p["carry_w"][0])
    theta = -p["knee"] / p["bank_w"][:, 0] * c
    for k in ("code", "carry_w", "fold_w", "rb"):
        p[k] = p[k] * c
    p["knee"] = -SLOPE * theta
    mark(q, f"re-scale so the carry write is {CARRY}")
    return p, log


def to_shipped(p):
    """The twelve numbers the shipped module holds, after checking the form."""
    d, dev = p["code"].dtype, p["code"].device
    assert tuple(p["code"].shape) == (10, 1)
    assert abs(float(p["code"][0])) < 1e-12
    assert torch.allclose(p["bank_w"], torch.full_like(p["bank_w"], SLOPE))
    assert torch.allclose(p["key_w"], torch.tensor([-KEY, KEY], dtype=d, device=dev))
    assert torch.allclose(p["val_w"], torch.tensor([0.0, 1.0], dtype=d, device=dev))
    assert abs(float(p["carry_w"][0]) - CARRY) < 1e-12
    assert abs(float(p["lam"][0]) - LAM) < 1e-12
    assert float(p["rb"].abs().max()) == 0.0
    assert abs(float(p["q"][0]) - 1.0) < 1e-12 and float(p["val_b"][0]) == 0.0
    assert float(p["ls_log"][0]) == 0.0
    return dict(code=[float(v) for v in p["code"][1:, 0]],
                knee=[float(v) for v in p["knee"]],
                fold=[float(p["fold_w"][0])])


def from_shipped(w, device="cpu", dtype=torch.float64):
    """Inverse of `to_shipped`: an ens-style member in the shipped form."""
    t = lambda v, s: torch.tensor(v, dtype=dtype, device=device).reshape(s)
    return dict(code=t([0.0] + list(w["code"]), (10, 1)),
                bank_w=t([SLOPE, SLOPE], (2, 1)),
                knee=t(list(w["knee"]), (2,)),
                key_w=t([-KEY, KEY], (2,)),
                val_w=t([0.0, 1.0], (2,)),
                val_b=t([0.0], (1,)),
                q=t([1.0], (1,)),
                lam=t([LAM], (1,)),
                carry_w=t([CARRY], (1,)),
                fold_w=t(list(w["fold"]), (1,)),
                rb=t([0.0], (1,)),
                ls_log=t([0.0], (1,)))
