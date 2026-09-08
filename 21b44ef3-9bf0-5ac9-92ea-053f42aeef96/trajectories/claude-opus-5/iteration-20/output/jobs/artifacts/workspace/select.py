"""Rank ensemble members for shipping.

Accuracy alone is a poor ranking: thousands of members reach 1.0 on any sample
while having a gate bank that is not fully saturated, which is exactly the
property that makes the whole-domain certificate possible.  Rank by
(exact, saturated, worst read-out margin) instead.
"""

import argparse

import torch

import data
import ens
import train


@torch.no_grad()
def report(path, places=(3, 5, 8, 12), batch=1024, top=20, device="cuda"):
    blob = torch.load(path, map_location=device)
    params = {k: v.to(device) for k, v in blob["params"].items()}
    e = params["code_free"].shape[0]
    gen = torch.Generator(device=device).manual_seed(4242)
    sets = []
    for n in places:
        da, db = data.sample(batch, n, device, gen, held_out=True, p_msb_nonzero=1.0)
        sets.append(data.tokens_and_targets(da, db))
    exact, margin = train.evaluate(params, sets, chunk=1024)
    sat = train.saturation(params)
    res = train.code_residual(params)

    key = (exact >= 1.0).float() * 100 + (sat > 0).float() * 10 + margin.clamp(-1, 5)
    order = torch.argsort(key, descending=True)[:top]
    rows = []
    for i in order.tolist():
        rows.append(dict(member=i, exact=float(exact[i]), sat=float(sat[i]),
                         margin=float(margin[i]), code_res=float(res[i])))
    summary = dict(members=e,
                   exact=int((exact >= 1.0).sum()),
                   exact_and_saturated=int(((exact >= 1.0) & (sat > 0)).sum()))
    return summary, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    s, rows = report(a.ckpt, top=a.top)
    print(s)
    for r in rows:
        print("  member %6d  exact %.5f  sat %+.4f  margin %+.4f  code_res %.4f"
              % (r["member"], r["exact"], r["sat"], r["margin"], r["code_res"]))
