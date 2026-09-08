"""Re-point a kv="share" checkpoint at the kv="one" architecture, as a warm start.

With kv="one" the attention key and the carry value are the *same* scalar times
the same axis z, so two quantities that the "share" model sets independently
become one:

    key gap  k_abs - k_gen = w * z_abs        (compared against the distance
                                               bias |lam|, so it must be small)
    carry    v_abs - v_gen = w * z_abs        (one code step, so it must be beta)

They are now literally the same number, which forces  k_abs - k_gen = -beta.
That is a real constraint, not a gauge: the code scale gauge is already spent on
pinning code[pin_row] = 1.  Everything else has to move to pay for it -- bank 1
must place its knees so that z_abs is far smaller than, and on the *opposite*
side of zero from, the transparent notch z_trans, because

    notch = w * z_trans = -beta * (z_trans / z_abs)

still has to clear 2 * |lam| * (longest carry chain) with room to spare.  The
"share" model sits at z_trans/z_abs = +9; the tied version needs a few hundred,
negative.  There is no continuous path from one to the other that keeps the
model working (z_abs passes through 0, where the carry vanishes), so this is a
warm start plus a seed lottery rather than an anneal: the shared scalar starts
at the parent's key scale, and train.py's --sigma scatters the bank-1 knees over
both signs of z_abs so that some members land in the reachable basin.

Nothing here is tuned toward a computed target; the diagnostics are printed so
that the size of the gap the training has to close is visible.
"""
import argparse
import torch
import lib
import train

DEV = "cuda"


def levels(p, cfg):
    """(z_abs, z_trans, z_gen, beta) per member, read off the code and bank 1."""
    code = lib.full_code(p, cfg)[..., 0]                      # (E,10)
    beta = code[:, 1] - code[:, 0]
    o = p["o_free"]
    if cfg["o_pin"]:
        o = torch.cat([torch.ones(o.shape[0], 1, device=o.device, dtype=o.dtype), o], 1)
    out = []
    for s in (4, 9, 14):                                      # absorb / transparent / generate
        x = 2 * code[:, 0] + beta * s
        h = torch.relu(x[:, None] * float(cfg["f1_sign"]) + p["b1"])
        out.append((h * o).sum(1))
    return out[0], out[1], out[2], beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    cfg = lib.default_cfg(**ck["cfg"])
    assert cfg["kv"] == "share" and not cfg["y_bias"], "fix the shift gauge first"
    p = {n: t.to(DEV).double() for n, t in ck["params"].items()}

    za, zt, zg, beta = levels(p, cfg)
    print("parent, per member:")
    for e in range(za.shape[0]):
        print("  b_q=%9.2f w_v=%8.3f  beta=%+.5f  z_abs=%+.3e z_trans=%+.3e"
              "  ratio=%+8.1f  need<%+8.1f"
              % (p["b_q"][e], p["w_v"][e, 0], beta[e], za[e] - zg[e], zt[e] - zg[e],
                 (zt[e] - zg[e]) / (za[e] - zg[e]),
                 30.0 / beta[e]))

    p1, cfg1 = dict(p), dict(cfg)
    # the shared scalar starts at the key scale: the notch is the quantity with
    # the tight lower bound, and it is the key side that already meets it.
    p1["w_v"] = p1.pop("b_q").reshape(-1, 1).repeat(1, cfg["C"]).contiguous()
    cfg1["kv"] = "one"

    ev_u = train.make_eval(8, 8192, DEV, 12345)
    ev_c = train.make_eval(8, 8192, DEV, 999, chain=True)

    def sc(pp, cc):
        return torch.minimum(train.evaluate(pp, cc, ev_u), train.evaluate(pp, cc, ev_c))
    print("held-out exact-match  parent:", [round(float(v), 5) for v in sc(p, cfg)])
    print("                  warm start:", [round(float(v), 5) for v in sc(p1, cfg1)])
    torch.save(dict(cfg=cfg1, params={n: t.float().cpu() for n, t in p1.items()},
                    scores=[]), args.out)
    print("saved", args.out, lib.n_params(cfg1), "params", cfg1)


if __name__ == "__main__":
    main()
