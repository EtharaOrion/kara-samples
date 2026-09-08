"""Re-express bank 2 in bank 1's orientation, as a warm start for a tie run.

Bank 2's fold is a difference of ReLUs that switches on *above* a threshold in
the residual, while bank 1's units switch on *below* one.  Tying the two banks
needs them to agree, and

    relu(r - k) = relu(k - r) + (r - k)

turns one into the other at the cost of a term linear in r; when the fold's
read-out coefficients sum to zero -- which they must, or the fold would not
saturate -- the linear parts cancel and only a constant is left, which is what
`y_bias` is for.

This is *not* one of xform.py's exact rewrites: the trained coefficients sum to
zero only to within training noise, so the reoriented model is a starting point
for `train.py --shrink tie`, and the accuracy it starts from is printed here so
that the size of that residual is visible rather than assumed.
"""
import argparse
import itertools
import torch
import lib
import train

DEV = "cuda"


def reorient(p, cfg):
    assert cfg["C"] == 1 and cfg["f2_in"] == "sign" and not cfg["tie"]
    p, cfg = {n: t.clone() for n, t in p.items()}, dict(cfg)
    s2 = float(cfg["f2_sign"])
    assert s2 == -float(cfg["f1_sign"]), "bank 2 already reads in bank 1's orientation"
    # unit i is relu(s2 * (r - k_i)); the reoriented unit is relu(-s2 * (r - k_i)),
    # and relu(u) - relu(-u) = u leaves  old - new = s2 * (r * sum(p) - sum(p*k))
    k = -p["b2"] / s2                                  # knee of unit i, in r
    by = -s2 * torch.einsum("eu,eu->e", p["p_out"][..., 0], k)
    p["b2"] = -p["b2"]
    p["by"] = p.get("by", torch.zeros_like(by)) + by
    cfg["f2_sign"] = float(cfg["f1_sign"])
    cfg["y_bias"] = True
    return p, cfg


def pad_to_tie(p, cfg):
    """Widen bank 2 to bank 1's width, matching each fold unit to the bank-1 unit
    whose knee is nearest (best total, over the few permutations) and filling the unused slots with bank-1's own bias and
    a zero read-out row.  A permutation plus zero rows, so the function is
    unchanged; it just puts the model in the shape `--shrink tie` anneals.
    """
    assert cfg["U2"] <= cfg["U"] and cfg["f1_sign"] == cfg["f2_sign"]
    p, cfg = {n: t.clone() for n, t in p.items()}, dict(cfg)
    E, U, U2 = p["b1"].shape[0], cfg["U"], cfg["U2"]
    b2 = p["b1"].clone()                                  # unused slots sit on bank 1
    po = torch.zeros(E, U, cfg["C"], dtype=p["p_out"].dtype, device=p["p_out"].device)
    for e in range(E):
        best = min(itertools.permutations(range(U), U2),
                   key=lambda a: sum(abs(float(p["b1"][e, i] - p["b2"][e, j]))
                                     for j, i in enumerate(a)))
        for j, i in enumerate(best):
            b2[e, i], po[e, i] = p["b2"][e, j], p["p_out"][e, j]
    p["b2"], p["p_out"] = b2, po
    cfg["U2"] = U
    return p, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tie_pad", action="store_true",
                    help="also widen bank 2 to bank 1's width, ready for --shrink tie")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    cfg = lib.default_cfg(**ck["cfg"])
    p = {n: t.to(DEV).double() for n, t in ck["params"].items()}
    p1, cfg1 = reorient(p, cfg)
    if args.tie_pad:
        p1, cfg1 = pad_to_tie(p1, cfg1)

    ev_u, ev_c = train.make_eval(8, 8192, DEV, 12345), train.make_eval(8, 8192, DEV, 999, chain=True)
    def sc(pp, cc):
        return torch.minimum(train.evaluate(pp, cc, ev_u), train.evaluate(pp, cc, ev_c))
    a0, a1 = sc(p, cfg), sc(p1, cfg1)
    print("residual sum of fold read-out (exactly 0 would make this exact):",
          [round(float(v), 6) for v in p["p_out"].sum(1)[:, 0]])
    print("held-out exact-match  before:", [round(float(v), 5) for v in a0])
    print("                       after:", [round(float(v), 5) for v in a1])
    torch.save(dict(cfg=cfg1, params={n: t.float().cpu() for n, t in p1.items()},
                    scores=[float(v) for v in a1],
                    uniform=ck.get("uniform", []), chain=ck.get("chain", [])), args.out)
    print("saved", args.out, cfg1)


if __name__ == "__main__":
    main()
