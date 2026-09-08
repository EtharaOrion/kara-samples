"""How linear is each member's code?

code[a] + code[b] must depend on a+b alone, which forces code[d] = step*d.  The
pinned entry code[1]=1 sits inside a tolerance band rather than fixing the step
exactly, so fit the step on the other nine entries and report the residual in
units of that step.
"""

import sys

import torch

import ens


def residual(code):
    d = torch.arange(10.0, device=code.device)
    keep = torch.ones(10, dtype=torch.bool, device=code.device)
    keep[1] = False
    step = (code[:, keep] * d[keep]).sum(1) / (d[keep] ** 2).sum()
    res = (code - step[:, None] * d).abs()
    return res[:, keep].amax(1) / step.abs().clamp(min=1e-6), step


def code_table(params):
    e = params["code_free"].shape[0]
    head = torch.tensor([ens.CONST["code0"], ens.CONST["code1"]],
                        device=params["code_free"].device).expand(e, 2)
    return torch.cat([head, params["code_free"]], dim=1)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        blob = torch.load(path, map_location="cuda")
        p = blob["params"]
        p = {k: v.cuda() for k, v in p.items()}
        code = code_table(p)
        res, step = residual(code)
        e = code.shape[0]
        k1 = p["knee"][:, 1] / step
        good = res < 0.08
        print("%-42s E=%d  linear(<0.08): %5d (%.3f%%)  "
              "linear&knee1_in(9,10): %4d  best_res %.4f  step[median|linear] %s"
              % (path.split("/")[-1], e, int(good.sum()), 100.0 * float(good.float().mean()),
                 int((good & (k1 > 9) & (k1 < 10)).sum()), float(res.min()),
                 round(float(step[good].median()), 3) if int(good.sum()) else "-"))
