"""Pick which trained member to ship.

Runs the exhaustive whole-domain certificate on each candidate and ranks by the
worst-case read-out margin.  All candidates that certify are exactly correct on
every 8-digit input; the margin says how much numerical room each one has, so
this picks the most robust of the exact ones rather than the first found.
"""

import argparse
import torch

from adder import DigitPairAdder
from verify import certify


def n_classes(m):
    dig = torch.arange(10)
    code = m.code().double()
    x = code[dig][:, None] + code[dig][None, :]
    u = torch.clamp(x[..., None] * m.bank_w.double() + m.bank_bias.double(), 0.0, 1.0)
    kv = torch.stack([(u * m.key_w.double()).sum(-1).reshape(-1),
                      (u * m.val_w.double()).sum(-1).reshape(-1)], -1)
    return torch.unique(kv, dim=0).shape[0]


def member(ck, i, cls=DigitPairAdder):
    m = cls()
    with torch.no_grad():
        for k, v in ck["params"].items():
            getattr(m, k).copy_(v[i].reshape(getattr(m, k).shape))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    acc = ck["acc"]
    rows = []
    for i in range(min(args.top, acc.numel())):
        m = member(ck, i)
        # A member whose bank does not saturate has many key/value classes and
        # the class enumeration is ncls**n; it cannot certify anyway, so skip.
        ncls = n_classes(m)
        if ncls > 4:
            print(f"member {i:3d} heldout {acc[i]:.6f} not certifiable: "
                  f"{ncls} key/value classes (bank does not saturate)")
            continue
        c = certify(m, args.n)
        rows.append((i, float(acc[i]), c))
        print(f"member {i:3d} heldout {acc[i]:.6f} wrong {c['wrong']:8d} "
              f"margin {c['min_margin']:+.4f} gap {c['min_code_gap']:.4f} "
              f"classes {c['classes']} sat {c['saturated']} role_ok {c['role_ok']}")
    ok = [r for r in rows if r[2]["wrong"] == 0 and r[2]["saturated"] and r[2]["role_ok"]]
    if not ok:
        print("no member certifies exact over the whole domain")
        return
    best = max(ok, key=lambda r: r[2]["min_margin"])
    print(f"\n{len(ok)}/{len(rows)} certify exact on all 10^{2*args.n} inputs; "
          f"best is member {best[0]} with worst-case margin {best[2]['min_margin']:.4f}")


if __name__ == "__main__":
    main()
