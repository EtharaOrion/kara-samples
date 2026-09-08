"""Collect the members that won phase 1 out of one or more lottery runs.

Phase 1 is a lottery: the great majority of random restarts settle into a code
that is not an arithmetic progression and can never add.  Pooling lets several
smaller runs stand in for one huge one, and lets a later run's winners join the
earlier ones.
"""

import argparse

import torch

import scorer


def collect(paths, thresh, device="cuda"):
    keep = {}
    rows = []
    for path in paths:
        blob = torch.load(path, map_location=device)
        p = {k: v.to(device) for k, v in blob["params"].items()}
        p.pop("proto", None)
        code = scorer.code_table(p)
        res, step = scorer.residual(code)
        # The pinned entry code[1]=1 is excluded from the fit, so a code that has
        # collapsed toward zero scores a perfect residual while being useless.
        # Require it to lie on the same line as the rest.
        onstep = (code[:, 1] - step).abs() / step.abs().clamp(min=1e-6) < 0.5
        idx = ((res < thresh) & onstep).nonzero().flatten()
        if idx.numel() == 0:
            continue
        for k, v in p.items():
            keep.setdefault(k, []).append(v[idx])
        for i in idx.tolist():
            rows.append((path.split("/")[-1], i, float(res[i]), float(step[i])))
    return {k: torch.cat(v, dim=0) for k, v in keep.items()}, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=0.08)
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    params, rows = collect(a.ckpts, a.thresh)
    for src, i, res, st in rows:
        print("  %-30s member %6d  code_res %.4f  step %+.3f" % (src, i, res, st))
    n = params["code_free"].shape[0] if params else 0
    torch.save({"params": {k: v.cpu() for k, v in params.items()}}, a.out)
    print("pooled %d parents -> %s" % (n, a.out))
