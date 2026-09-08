"""Stage 3: put a trained C=1 member into the shipped form and fine-tune it there.

Two kinds of move are applied to the trained C=1 model.

(a) Exact rewrites that leave the function unchanged.  There are three:
      - clamp-unit flip, using clamp(1-t,0,1) = 1 - clamp(t,0,1), so a unit can
        be replaced by its complement with (Wb,bb) -> (-Wb, 1-bb), kw -> -kw
        (a constant added to every key cancels inside the softmax) and
        vw -> -vw, vb -> vb + vw;
      - unit reordering;
      - the residual-scale gauge: scaling code, and with it every other
        residual-space quantity, leaves the read-out argmax unchanged.  We
        spend it to make the carry-in write-back uA exactly 1.
    Together these put every member in one canonical pose:
        bank gate g = (0,0) on absorb (a+b<=8), (1,0) on transparent (a+b==9),
                      (1,1) on generate (a+b>=10).

(b) Substitution of architectural constants that are not learned facts about
    addition -- the bank slope, the key/value read-off of the gate, the query
    scale, the distance-bias slope and the read-out temperature.  These are set
    to round hand-chosen numbers, held as buffers, and are NOT fitted.
    Substituting them perturbs the model, so afterwards the twelve remaining
    free values are re-trained by gradient descent inside the exact shipped
    architecture.  The shipped weights are therefore the direct output of
    training on the shipped model, not a rewritten copy of something else.

Free after this stage: code[1..9] (9), bank knees bb[0..1] (2), fold uB (1) = 12.
"""

import argparse
import torch

import arch
import lab

# Architectural constants of the shipped model (buffers, never fitted).
BANK_W = 8.0        # clamp bank slope
KEY_W = 400.0       # gate -> key read-off, +-KEY_W
Q = 1.0             # query scale
LAM = -12.0         # relative-distance bias slope
LS = 1.0            # read-out temperature (argmax is invariant to it)
CARRY_W = 1.0       # carry-in write-back (this is the residual-scale gauge)

PAIRS_A = torch.arange(10)[:, None].expand(10, 10).reshape(-1)
PAIRS_B = torch.arange(10)[None, :].expand(10, 10).reshape(-1)


def gate_by_class(p):
    """Bank gate averaged over each carry class, plus a saturation measure."""
    dev = p["code"].device
    aa, bb_ = PAIRS_A.to(dev), PAIRS_B.to(dev)
    x = p["code"][:, aa] + p["code"][:, bb_]                     # [E,100,C]
    e = torch.einsum("euc,epc->epu", p["Wb"], x) + p["bb"][:, None, :]
    g = e.clamp(0.0, 1.0)                                        # [E,100,U]
    s = aa + bb_
    cls = torch.where(s <= 8, 0, torch.where(s == 9, 1, 2))
    pat = torch.stack([g[:, cls == c].mean(1) for c in range(3)], 1)     # [E,3,U]
    slack = torch.stack([torch.minimum(g[:, cls == c], 1 - g[:, cls == c]).amax(1)
                         for c in range(3)], 1)                          # [E,3,U]
    return pat, slack


def apply_flip(p, flip):
    """flip: bool tensor [U].  Exact function-preserving complement of units."""
    q = {k: v.clone() for k, v in p.items()}
    f = flip.to(p["Wb"].device)
    sgn = torch.where(f, -1.0, 1.0)
    q["Wb"] = p["Wb"] * sgn[None, :, None]
    q["bb"] = torch.where(f[None, :], 1.0 - p["bb"], p["bb"])
    q["kw"] = p["kw"] * sgn[None, :]
    q["vw"] = p["vw"] * sgn[None, :]
    q["vb"] = p["vb"] + (p["vw"] * f[None, :]).sum(-1)
    return q


def apply_perm(p, perm):
    q = {k: v.clone() for k, v in p.items()}
    for k in ("Wb", "bb", "kw", "vw"):
        q[k] = p[k][:, perm]
    return q


def canonicalize(p):
    """Search the 8 flip/permutation poses and pick, per member, the one whose
    gate pattern is closest to (0,0)/(1,0)/(1,1)."""
    dev = p["code"].device
    target = torch.tensor([[0., 0.], [1., 0.], [1., 1.]], device=dev)
    cands = []
    for perm in ([0, 1], [1, 0]):
        for f0 in (False, True):
            for f1 in (False, True):
                c = apply_perm(p, torch.tensor(perm, device=dev))
                c = apply_flip(c, torch.tensor([f0, f1]))
                pat, _ = gate_by_class(c)
                err = ((pat - target) ** 2).sum((1, 2))
                cands.append((c, err))
    errs = torch.stack([e for _, e in cands], 1)          # [E, 8]
    pick = errs.argmin(1)
    out = {}
    for k in p:
        stack = torch.stack([c[k] for c, _ in cands], 1)   # [E,8,...]
        idx = pick.reshape(-1, 1, *([1] * (p[k].dim() - 1))).expand(
            -1, 1, *p[k].shape[1:])
        out[k] = stack.gather(1, idx).squeeze(1)
    return out, errs.gather(1, pick[:, None]).squeeze(1)


def value_norm(p):
    """Read the gate off as value 0 on absorb / 1 on generate.

    In the canonical pose absorb has gate (0,0) so its value is vb, and generate
    has gate (1,1) so its value is vw0+vw1+vb.  Rescaling the value by that span
    and folding the span into uA, uB leaves the head outputs equivalent up to
    one lost constant K = vb*(uA+uB) subtracted from the residual -- which the
    code translation below puts back, because a correct model must satisfy
    code[0] + K = 0 (see NOTES.md).
    """
    q = {k: v.clone() for k, v in p.items()}
    span = p["vw"].sum(-1)                                   # v_generate - v_absorb
    q["vw"] = torch.stack([torch.zeros_like(span), torch.ones_like(span)], 1)
    q["vb"] = torch.zeros_like(p["vb"])
    q["uA"] = p["uA"] * span[:, None]
    q["uB"] = p["uB"] * span[:, None]
    return q, p["vb"] * (p["uA"][:, 0] + p["uB"][:, 0])


def translate(p):
    """Shift the code so code[0] == 0.  A place token embeds as code[a]+code[b],
    so the residual moves by -2*code[0] and the bank bias has to follow."""
    q = {k: v.clone() for k, v in p.items()}
    c0 = p["code"][:, 0:1, :]                                   # [E,1,C]
    q["code"] = p["code"] - c0
    q["bb"] = p["bb"] + 2.0 * torch.einsum("euc,ec->eu", p["Wb"], c0[:, 0, :])
    return q


def rescale(p):
    """Residual-scale gauge: scale the residual stream so uA == CARRY_W.
    Exact: code, the bank input weight, and both write-backs scale together."""
    alpha = CARRY_W / p["uA"][:, 0]                    # [E]
    q = {k: v.clone() for k, v in p.items()}
    q["code"] = p["code"] * alpha[:, None, None]
    q["Wb"] = p["Wb"] / alpha[:, None, None]
    q["uA"] = p["uA"] * alpha[:, None]
    q["uB"] = p["uB"] * alpha[:, None]
    return q


def set_bank_slope(p):
    """Substitute the clamp slope, holding each knee where the model put it."""
    q = {k: v.clone() for k, v in p.items()}
    x_mid = (0.5 - p["bb"]) / p["Wb"][:, :, 0]        # Wb*x + bb = 0.5
    q["Wb"] = torch.full_like(p["Wb"], BANK_W)
    q["bb"] = 0.5 - BANK_W * x_mid
    return q


def sharpen(p):
    """Substitute the attention constants: unit query scale, a +-KEY_W key
    read-off of the gate, a fixed distance-bias slope, unit read-out
    temperature.  The learned key already has the form q*kw = (-c, +c); this
    only replaces c and the recency slope with round constants."""
    q = {k: v.clone() for k, v in p.items()}
    o = torch.ones_like(p["q"])
    q["kw"] = torch.stack([-KEY_W * o, KEY_W * o], 1)
    q["q"] = Q * o
    q["lam"] = LAM * o
    q["ls"] = LS * o
    return q


FREE = ("code", "bb", "uB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/c1.pt")
    ap.add_argument("--out", default="ckpt/ship.pt")
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=0.0015)
    ap.add_argument("--m_target", type=float, default=0.42)
    ap.add_argument("--sat_w", type=float, default=0.5)
    ap.add_argument("--sat_target", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    dev = "cuda"
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    d = torch.load(a.ckpt, map_location=dev, weights_only=False)
    p = {k: v.to(dev) for k, v in d["params"].items()}
    E = p["code"].shape[0]

    acc, _ = lab.evaluate(p, 8, dev, g, n_batch=4, B=512, chunk=128)
    print(f"loaded {E} C=1 members, {int((acc>=0.9999).sum())} exact", flush=True)

    def report(p, label):
        a_, _ = lab.evaluate(p, 8, dev, g, n_batch=4, B=512, chunk=128)
        print(f"  {label:<44s} exact members {int((a_>=0.9999).sum()):4d}/{len(a_)}  "
              f"best {float(a_.max()):.4f}  median {float(a_.median()):.4f}", flush=True)
        return a_

    p, pose_err = canonicalize(p)
    print(f"pose error range {float(pose_err.min()):.2e}..{float(pose_err.max()):.2e}",
          flush=True)
    acc1 = report(p, "canonicalize [exact]")
    good = (acc1 >= 0.9999) & (pose_err < 1e-3)
    idx = torch.nonzero(good).squeeze(1)
    p = {k: v[idx] for k, v in p.items()}
    print(f"keeping {len(idx)} canonical + exact members", flush=True)

    p, K = value_norm(p)
    print("  lost constant K = vb*(uA+uB):",
          [round(float(x), 3) for x in K[:5]], flush=True)
    print("  -code[0]                    :",
          [round(float(-x), 3) for x in p['code'][:5, 0, 0]], flush=True)
    print("  uA after value norm (must be > 0):",
          [round(float(x), 3) for x in p['uA'][:5, 0]], flush=True)
    report(p, "value read-off -> 0/1 [drops constant K]")
    p = translate(p)
    report(p, "translate code[0] -> 0 [restores K]")
    p = rescale(p)
    report(p, "scale gauge uA -> 1 [exact]")
    p = set_bank_slope(p)
    report(p, "bank slope -> 8 [substitution]")
    p = sharpen(p)
    acc3 = report(p, "attention constants [substitution]")

    # fine-tune the twelve free values inside the exact shipped architecture
    reps = {k: v.repeat_interleave(a.copies, 0).clone() for k, v in p.items()}
    for k in FREE:
        reps[k] += a.sigma * torch.randn(reps[k].shape, generator=g, device=dev) \
            * reps[k].abs().mean()
    for k in FREE:
        reps[k][:, 0:1] = reps[k][:, 0:1] if k != "code" else 0.0
    gm = {"code": torch.ones_like(reps["code"])}
    gm["code"][:, 0, :] = 0.0        # code[0] stays pinned at 0

    print(f"polishing {reps['code'].shape[0]} members in the shipped form, "
          f"free = {FREE}, objective = read-out margin", flush=True)
    reps, best, _ = lab.train(reps, list(FREE), a.steps, dev, g, lr=a.lr, B=512,
                              pct_start=0.05, tag="ship", eval_chunk=128, grad_mask=gm,
                              objective="margin", m_target=a.m_target, score_w=0.05,
                              eval_every=250, log_every=1000, eval_batches=6,
                              sat_w=a.sat_w, sat_target=a.sat_target)
    acc, worst = lab.evaluate(best, 8, dev, g, n_batch=24, B=1024, chunk=128)
    sat = lab.sat_slack(best).amin((1, 2))
    order = torch.argsort(acc + 0.05 * worst.clamp(-5, 5)
                          + 0.05 * sat.clamp(-5, a.sat_target), descending=True)
    print("top saturation slack:", [round(float(sat[i]), 3) for i in order[:10]],
          flush=True)
    print("top shipped-form members:",
          [(round(float(acc[i]), 5), round(float(worst[i]), 3)) for i in order[:10]],
          flush=True)
    print("members >= 0.9999:", int((acc >= 0.9999).sum()), flush=True)
    keep = order[:64]
    torch.save({"params": {k: v[keep].cpu() for k, v in best.items()},
                "acc": acc[keep].cpu(), "worst": worst[keep].cpu(),
                "sat": sat[keep].cpu(), "C": 1, "U": 2}, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
