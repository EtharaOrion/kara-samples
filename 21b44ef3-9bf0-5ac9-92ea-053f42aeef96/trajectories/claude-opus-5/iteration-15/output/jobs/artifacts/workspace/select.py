"""Certify every member of a shipped-form checkpoint and rank them.

Ranking is: certified first, then bank saturation slack, then the worst
read-out margin.  A member whose bank sits on top of a reachable digit sum is
rejected outright rather than patched -- it is a member that did not find the
mechanism cleanly, and there are plenty that did.
"""
import argparse, json
import torch

import certify, ens, reduce as red


def member_report(p, n=8):
    """p: a single float64 member already in the shipped form."""
    w = dict(code=p["code"][:, 0], knee=p["knee"], fold=p["fold_w"],
             bank_w=p["bank_w"][:, 0], key_w=p["key_w"], val_w=p["val_w"],
             carry_w=p["carry_w"], lam=p["lam"])
    r1 = certify.step1_bank(w)
    r2 = certify.step2_attention(w, n)
    r3 = certify.step3_readout(w, r2["drift"])
    ok = r1["saturated"] and r1["classes_match_arithmetic"] and r3["proved_exact"]
    return ok, r1, r2, r3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--top", type=int, default=64)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    pr = ck["params"]
    acc, order = ck["acc"], ck["order"]
    rows = []
    for i in range(min(args.top, len(order))):
        m = int(order[i])
        if float(acc[m]) < 0.9999:
            break
        p = ens.single(pr, m)
        try:
            red.to_shipped(p)
        except AssertionError as e:
            continue
        ok, r1, r2, r3 = member_report(p, args.n)
        rows.append(dict(member=m, certified=ok, slack=r1["slack"],
                         classes_ok=r1["classes_match_arithmetic"],
                         margin=r3["worst_margin"], drift=r2["drift"],
                         safety=r3["safety_factor"],
                         code_gap=r3["code_gap"]))
    # Slack past ~0.5 buys nothing -- the clamp is saturated either way and the
    # aux term drives every member to about the same place -- so cap it and let
    # the read-out margin, which is what the certificate actually spends, decide.
    rows.sort(key=lambda r: (r["certified"], min(r["slack"], 0.5), r["margin"]),
              reverse=True)
    n_ok = sum(r["certified"] for r in rows)
    print(f"{len(rows)} members checked, {n_ok} certified exact on n={args.n}")
    for r in rows[:10]:
        print(f"  member {r['member']:5d} certified={r['certified']} "
              f"slack={r['slack']:+.4f} margin={r['margin']:.4f} "
              f"gap={r['code_gap']:.4f} safety={r['safety']:.0f}x")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
    if rows and rows[0]["certified"]:
        print(f"BEST {rows[0]['member']}")


if __name__ == "__main__":
    main()
