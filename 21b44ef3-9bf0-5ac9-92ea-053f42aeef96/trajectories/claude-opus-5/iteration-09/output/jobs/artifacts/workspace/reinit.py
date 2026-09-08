"""Produce a starting point for the next rung of the ladder.

Unlike `xform.py`, this does NOT preserve the function: it moves the model to a
different point of weight space that further *training* then has to make work.
Nothing about the answer is computed here -- the only operations are a change of
variables on the digit code and the removal of one bank unit.

Why it exists.  The bank has one unit with a zero input weight whose only job is
to supply a constant offset to every key and value.  That offset is exactly what
lets the digit code sit at a non-zero intercept, so removing the unit and moving
the code's intercept to zero have to happen together.  Annealing either one
alone fails: the optimiser keeps the offset alive in the unit's other weights
until the constraint bites, then loses it all at once.  Doing both by hand and
retraining walks around that.

The code translation is a change of variables: shifting every code entry by d
shifts each token embedding by 2d, which the bank biases absorb exactly, so the
keys, the values and the attention are untouched.  What the translation does
change is the read-out, and repairing that is the job of the training run that
consumes this checkpoint.
"""
import argparse
import torch

from model_src import DigitAdder, default_cfg
from data import Sampler


def merge_knees(cfg, st, tol):
    """Fuse two bank units that have converged on the same knee.

    A unit contributes ``key_w[u] * relu(W1[u]*x + b1[u])``, so above its knee at
    ``-b1[u]/W1[u]`` it adds a ramp of slope ``key_w[u]*W1[u]``.  Two units whose
    knees coincide are one ramp written twice; keeping one of them and giving it
    the summed slope is the same function wherever the knees really do coincide,
    and training afterwards absorbs the difference where they do not.
    """
    w = st["W1"][0, :, 0]
    b = st["b1"][0]
    kw = st["key_w"][0]
    U = cfg["U"]
    knee = [-(b[u] / w[u]).item() for u in range(U)]
    step = (st["code_free"][0, -1, 0] - st["code_free"][0, 0, 0]).item() / (
        st["code_free"].shape[1] - 1)
    best, bd = None, 1e9
    for i in range(U):
        for j in range(i + 1, U):
            d = abs(knee[i] - knee[j]) / abs(step)
            if d < bd:
                best, bd = (i, j), d
    assert bd < tol, f"closest knees are {bd:.3f} code steps apart (tol {tol})"
    i, j = best
    print(f"merging units {i} and {j}: knees {knee[i]:+.5f} / {knee[j]:+.5f} "
          f"({bd:.4f} code steps apart), slopes "
          f"{(kw[i]*w[i]).item():+.3f} / {(kw[j]*w[j]).item():+.3f}")
    st["key_w"][:, i] = (st["key_w"][:, i] * st["W1"][:, i, 0]
                         + st["key_w"][:, j] * st["W1"][:, j, 0]) / st["W1"][:, i, 0]
    keep = [u for u in range(U) if u != j]
    for k in ("W1", "b1", "key_w", "val_w"):
        if k in st:
            st[k] = st[k][:, keep]
    cfg["U"] = len(keep)
    return cfg, st


def absorb_wo_shared(cfg, st):
    """Put the attention write scale on 1 by scaling the shared key/value.

    The attention output is an average of the values, so scaling every key and
    value by c scales the output by c -- exactly, if the softmax weights did not
    move.  They do move, because the key is also the score: the distribution
    gets sharper by a factor c.  For c > 1 that is the harmless direction (the
    head was already close to a hard selection), which is why this is worth
    doing by hand rather than annealing w_o down while the key scale, which has
    to grow to compensate, is still where training left it.
    """
    assert cfg["kv_share"] and cfg["wo"] == "free" and cfg["C"] == 1
    c = st["w_o"][:, 0].clone()
    assert (c > 0).all(), "w_o must be positive to keep the attention sharper"
    st["key_w"] = st["key_w"] * c[:, None]
    st["w_o2"] = st["w_o2"] / c[:, None]
    st.pop("w_o")
    cfg["wo"], cfg["wo_val"] = "fix", 1.0
    print(f"absorbed w_o = {c[0].item():.4f} into the shared key/value")
    return cfg, st


def finish(cfg, st, args):
    cfg["E"] = 1
    m = DigitAdder(cfg)
    sd = m.state_dict()
    for k, v in st.items():
        sd[k] = v.reshape(sd[k].shape)
    m.load_state_dict(sd)
    n_par = sum(p.numel() for p in m.parameters())
    smp = Sampler(args.dev)
    m = m.to(args.dev).double()
    accs, per_digit = [], 0.0
    with torch.no_grad():
        for pl in (8, 5, 11):
            a, b, t = smp.batch(4096, n=pl, split="val")
            lg = m(a, b)[:, :, 1:, :]
            accs.append((lg.argmax(-1) == t[None]).all(-1).float().mean().item())
            per_digit = (lg.argmax(-1) == t[None]).float().mean().item()
    print(f"parameters {n_par}   accuracy at this starting point "
          f"{[round(x, 4) for x in accs]}  (per-digit {per_digit:.4f})")
    torch.save({"cfg": cfg, "state": st, "n_par": n_par, "from": args.ckpt}, args.out)
    print(f"saved {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--op", default="drop_const",
                    choices=("drop_const", "merge_knees", "absorb_wo"))
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max knee separation (in code steps) to merge")
    ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = default_cfg(dict(ck["cfg"]))
    assert cfg["C"] == 1
    st = {k: v[args.member: args.member + 1].clone() for k, v in ck["state"].items()}
    if args.op == "merge_knees":
        cfg, st = merge_knees(cfg, st, args.tol)
        finish(cfg, st, args)
        return
    if args.op == "absorb_wo":
        cfg, st = absorb_wo_shared(cfg, st)
        finish(cfg, st, args)
        return
    assert cfg["f_in"] == "free" and not cfg["pin_code"]

    # the unit whose input weight is (near) zero is the constant one
    w = st["W1"][:, :, 0]
    u = int(w.abs().argmin(dim=1)[0])
    assert w[0, u].abs() < 1e-2 * w.abs().max(), f"unit {u} is not constant"
    keep = [i for i in range(cfg["U"]) if i != u]

    # translate the code to intercept zero, absorbed by the bank biases
    d = -st["code_free"][:, 0, 0].clone()
    st["code_free"][:, :, 0] += d[:, None]
    st["b1"] = st["b1"] - 2.0 * d[:, None] * st["W1"][:, :, 0]
    print(f"translated code by {d.item():+.4f}; dropped constant unit {u}")

    for k in ("W1", "b1", "key_w", "val_w"):
        if k in st:
            st[k] = st[k][:, keep]
    st["code_free"] = st["code_free"][:, 1:]        # digit 0 becomes a pinned 0
    cfg["U"] = len(keep)
    cfg["pin_code"] = ((0, 0.0),)
    cfg["E"] = 1

    m = DigitAdder(cfg)
    sd = m.state_dict()
    for k, v in st.items():
        sd[k] = v.reshape(sd[k].shape)
    m.load_state_dict(sd)
    n_par = sum(p.numel() for p in m.parameters())

    smp = Sampler(args.dev)
    m = m.to(args.dev).double()
    accs = []
    with torch.no_grad():
        for pl in (8, 5, 11):
            a, b, t = smp.batch(4096, n=pl, split="val")
            lg = m(a, b)[:, :, 1:, :]
            accs.append((lg.argmax(-1) == t[None]).all(-1).float().mean().item())
            per_digit = (lg.argmax(-1) == t[None]).float().mean().item()
    print(f"parameters {n_par}   accuracy at this starting point "
          f"{[round(x, 4) for x in accs]}  (per-digit {per_digit:.4f})")
    torch.save({"cfg": cfg, "state": st, "n_par": n_par, "from": args.ckpt}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
