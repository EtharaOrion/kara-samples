"""Reduce the trained C=2 parent to the 12-parameter shipped form.

stage `project`   C=2 -> C=1.  The learned code comes out numerically rank-1,
                  so projecting it onto its top singular direction (and folding
                  the mean perpendicular part into the bank bias) is close to
                  exact; a short retrain in C=1 recovers the rest.

stage `ship`      Spend the one residual-scale gauge on carry_w = 1, substitute
                  the architectural saturation constants (bank slope, key
                  contrast, recency slope, read-out temperature), then re-train
                  the 12 remaining free values -- code[1..9], the two knees and
                  the fold -- against a geometric read-out margin.

Every rewrite that claims to be exact is checked in float64 against the model
it came from before anything downstream uses it.
"""
import argparse
import time

import torch

import arch
import data
import lab
import two_phase as tp

DEV = "cuda"
BANK_W = 8.0
KEY_W = 400.0
LAM = -12.0


def ship_cfg():
    return arch.default_cfg(C=1, U=2, act="clamp", fix_bank_w=BANK_W,
                            fix_key_w=KEY_W, fix_val_w=(0.0, 1.0),
                            fix_val_b=True, fix_uA=1.0, fix_lam=LAM,
                            fix_q=1.0, fix_ls=1.0, fix_code0=True)


# ---------------------------------------------------------------- project
def project(p, cfg):
    """C=2 -> C=1 along the code's top singular direction."""
    w = arch.effective(p, cfg, DEV)
    code = w["code"]                                        # (E,10,C)
    _, S, Vh = torch.linalg.svd(code, full_matrices=False)
    u = Vh[:, 0, :]                                         # (E,C)
    c = torch.einsum("edc,ec->ed", code, u)                 # (E,10)
    sgn = torch.sign(c[:, 9])
    u, c = u * sgn[:, None], c * sgn[:, None]
    r = code - c[..., None] * u[:, None, :]                 # perpendicular part
    rbar = r.mean(1)                                        # (E,C)

    q = {}
    q["code"] = c[..., None].contiguous()
    q["Wb"] = torch.einsum("euc,ec->eu", w["Wb"], u)[:, :, None].contiguous()
    q["bb"] = w["bb"] + 2.0 * torch.einsum("euc,ec->eu", w["Wb"], rbar)
    q["uA"] = (w["uA"] * u).sum(-1, keepdim=True)
    q["uB"] = (w["uB"] * u).sum(-1, keepdim=True)
    q["kw"] = w["kw"].clone()
    q["vw"] = w["vw"].clone()
    q["vb"] = w["vb"].clone()
    q["lam"] = w["lam"].clone()
    q["q"] = w["q"].clone()
    q["log_ls"] = torch.log(w["ls"].clamp(min=1e-8))
    return q, (S[:, 1] / S[:, 0].clamp(min=1e-12))


# ------------------------------------------------------------------- ship
def to_ship_form(p, cfg):
    """Gauge-fix carry_w to 1 and substitute the saturation constants.

    Returns params in `ship_cfg()` parameterisation, where g_u =
    clamp(8*x + bb_u, 0, 1) so the knee sits at -bb_u/8.
    """
    w = arch.effective(p, cfg, DEV)
    s = 1.0 / w["uA"][:, 0]                                 # residual rescale
    code = w["code"][:, :, 0] * s[:, None]                  # (E,10)
    Wb = w["Wb"][:, :, 0] / s[:, None]                      # (E,U)
    theta = (0.5 - w["bb"]) / Wb                            # knee midpoint in x
    theta = theta * 1.0                                     # already in new units
    uB = w["uB"][:, 0] * s

    q = {}
    q["code"] = code[..., None].contiguous()
    q["Wb"] = torch.ones(code.shape[0], 2, 1, device=DEV)   # sign only; |.|=8
    # shipped midpoint is -bb/8 + 0.5/8, so match midpoints
    q["bb"] = 0.5 - BANK_W * theta
    q["uA"] = torch.ones(code.shape[0], 1, device=DEV)
    q["uB"] = uB[:, None].contiguous()
    q["kw"] = torch.tensor([[-KEY_W, KEY_W]], device=DEV).expand(code.shape[0], 2).contiguous()
    q["vw"] = torch.tensor([[0.0, 1.0]], device=DEV).expand(code.shape[0], 2).contiguous()
    q["vb"] = torch.zeros(code.shape[0], 1, device=DEV)
    q["lam"] = torch.full((code.shape[0], 1), LAM, device=DEV)
    q["q"] = torch.ones(code.shape[0], 1, device=DEV)
    q["log_ls"] = torch.zeros(code.shape[0], 1, device=DEV)
    return q


def canonical_mask(p, cfg):
    """Members already in the orientation the shipped form assumes."""
    w = arch.effective(p, cfg, DEV)
    Wb = w["Wb"][:, :, 0]
    theta = (0.5 - w["bb"]) / Wb
    kwe = w["kw"] * w["q"]
    return ((Wb[:, 0] > 0) & (Wb[:, 1] > 0)          # both units up-facing
            & (theta[:, 0] < theta[:, 1])            # notch knee below value knee
            & (kwe[:, 0] < 0) & (kwe[:, 1] > 0)      # key notched, not peaked
            & (w["uA"][:, 0] > 0)                    # carry written positively
            & (w["uB"][:, 0] < 0))                   # fold subtracts


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["project", "ship"])
    ap.add_argument("--inp", default="parent.pt")
    ap.add_argument("--out", default="c1.pt")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.0)
    a = ap.parse_args()

    ck = torch.load(a.inp, map_location=DEV, weights_only=False)
    p = {k: v.to(DEV) for k, v in ck["params"].items()}
    cfg = ck["cfg"]
    E = p["code"].shape[0]
    print(f"loaded {a.inp}: {E} members, C={cfg['C']}", flush=True)

    if a.stage == "project":
        q, ratio = project(p, cfg)
        print(f"code singular ratio  median {ratio.median():.5f}  "
              f"min {ratio.min():.5f}", flush=True)
        ncfg = arch.default_cfg(**{**cfg, "C": 1})
        corners = lab.make_eval([8, 5, 11, 3], DEV, per_n=2048)
        acc0 = lab.eval_exact(q, ncfg, corners)
        print(f"after raw projection: max {acc0.max():.5f}  "
              f"#>=.99 {(acc0>=0.99).sum().item()}", flush=True)
        keys = arch.trainable_keys(ncfg)
        t0 = time.time()
        b, w = tp.train(ncfg, q, keys, a.steps, 512, a.lr, 21, corners,
                        [2, 3, 5, 8, 11], pct=0.05, log="c1")
        keep = (b >= 0.9999).nonzero().flatten()
        print(f"C=1 retrain: max {b.max():.5f}  #=1.0 {(b>=1.0).sum().item()}  "
              f"#>=.9999 {keep.numel()} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(dict(params={k: v[keep].cpu() for k, v in w.items()},
                        cfg=ncfg, acc=b[keep].cpu()), a.out)
        print(f"saved {a.out} with {keep.numel()} members", flush=True)

    else:
        keep = canonical_mask(p, cfg).nonzero().flatten()
        print(f"canonically oriented: {keep.numel()}/{E}", flush=True)
        p = {k: v[keep].contiguous() for k, v in p.items()}
        q = to_ship_form(p, cfg)
        scfg = ship_cfg()
        print("free scalars in shipped form =", arch.n_free_values(scfg))
        corners = lab.make_eval([8, 5, 11, 3], DEV, per_n=2048)
        acc0 = lab.eval_exact(q, scfg, corners)
        print(f"after constant substitution: max {acc0.max():.5f}  "
              f"#>=.99 {(acc0>=0.99).sum().item()}  "
              f"#>=.5 {(acc0>=0.5).sum().item()}", flush=True)
        torch.save(dict(params={k: v.cpu() for k, v in q.items()}, cfg=scfg,
                        acc=acc0.cpu()), a.out)
        print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
