"""Walk a trained one-channel member down to the twelve shipped numbers.

Prints the accuracy after every move so the effect of each is visible, and
writes the surviving candidates to a json file for `finetune.py`.
"""
import argparse, json, os
import torch

import data, ens, reduce as red

PLACES = (8, 5, 11, 3)


def measure(p, dev, N=16384, places=PLACES, seed=31337):
    """Held-out exact-match accuracy of one member at several widths."""
    g = torch.Generator(device=dev).manual_seed(seed)
    e1 = {k: v[None].float() for k, v in p.items()}
    out = {}
    for n in places:
        ta, tb, tg, keep = data.heldout(N, n, g, dev)
        a, tot = ens.exact_acc(e1, ta, tb, tg, keep)
        out[n] = (float(a[0]), tot)
    return out


def worst(m):
    return min(v[0] for v in m.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    pr = {k: v.to(dev) for k, v in ck["params"].items()}
    acc, order = ck["acc"].to(dev), ck["order"].to(dev)

    g = torch.Generator(device=dev).manual_seed(5150)
    probe = [data.sample(2048, n, g, dev)[:2] for n in PLACES]

    keep = []
    for i in range(min(args.top, len(order))):
        m = int(order[i])
        if float(acc[m]) < 0.999:
            break
        p = ens.single(pr, m)
        pa = measure(p, dev)
        log = []
        for (ta, tb) in probe[:1]:
            p_ex, l = red.exact_rewrites(p, ta, tb)
            log += l
        # re-check the exact moves on every width
        ok_exact = all(e["argmax_same"] for e in log)
        for (ta, tb) in probe[1:]:
            log.append(red._agree(p, p_ex, ta, tb, "exact chain (other widths)"))
            ok_exact &= log[-1]["argmax_same"]
        pb = measure(p_ex, dev)
        diag = red.diagnose(p_ex)

        p_sub, l2 = red.substitute(p_ex, probe[0][0], probe[0][1])
        pc = measure(p_sub, dev)
        try:
            w = red.to_shipped(p_sub)
        except AssertionError as e:
            print(f"member {m}: shipped-form check failed: {e}")
            continue

        line = (f"member {m:5d}  parent {worst(pa):.6f} -> exact {worst(pb):.6f} "
                f"-> substituted {worst(pc):.6f}   exact-moves-clean={ok_exact}")
        print(line)
        if args.verbose:
            print("   knees", [round(t, 4) for t in diag["knee_positions"]],
                  " step", round(diag["code_step"], 4),
                  " nonlin", round(diag["code_nonlinearity"], 4),
                  " key", diag["key_w"], " val", diag["val_w"],
                  " lam", round(diag["lam"], 3), " rb", round(diag["rb"], 4),
                  " fold", round(diag["fold_w"], 4))
        keep.append(dict(member=m, weights=w, diag=diag,
                         acc_parent=pa, acc_exact=pb, acc_sub=pc,
                         exact_moves=log, exact_moves_clean=ok_exact,
                         sub_moves=l2))

    keep.sort(key=lambda r: -worst(r["acc_sub"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(keep, f, indent=1)
    print(f"wrote {len(keep)} candidates to {args.out}")
    if keep:
        print("best after substitution:", worst(keep[0]["acc_sub"]),
              "member", keep[0]["member"])


if __name__ == "__main__":
    main()
