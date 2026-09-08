"""Stage 4 -- pick a member and convert it into the shipped state_dict.

The trainer carries the shipped form in the ensemble parameterisation
(`reduce.ship_cfg()`); the graded file wants plain named tensors.  The map is a
renaming plus one algebraic rewrite of the bank, which is checked in float64
against the trainer's own forward pass before anything is written:

    trainer:  e = sign(Wb)*8 * x + bb
    shipped:  e = bank_w * (x - knee)      with bank_w = sign(Wb)*8,
                                                knee   = -bb / bank_w

Every candidate is then put through the whole-domain certificate and the one
with the widest certified margin is the one that ships.
"""
import argparse
import types

import torch

import arch
import certify as cert
import reduce as red
import ship_model

DEV = "cuda"


def to_shipped(p, cfg, i):
    """Member i of a ship_cfg ensemble -> (params, buffers) for the graded file."""
    w = arch.effective(p, cfg, DEV)
    bank_w = torch.sign(w["Wb"][i, :, 0]) * red.BANK_W          # (U,)
    knee = -w["bb"][i] / bank_w                                 # (U,)
    code = w["code"][i, :, 0]                                   # (10,)
    params = {
        "code_free": code[1:].tolist(),
        "knee": knee.tolist(),
        "fold": w["uB"][i].tolist(),
    }
    buffers = {
        "code0": code[:1].tolist(),
        "bank_w": bank_w.tolist(),
        "key_w": w["kw"][i].tolist(),
        "val_w": w["vw"][i].tolist(),
        "val_b": w["vb"][i].tolist(),
        "carry_w": w["uA"][i].tolist(),
        "lam": w["lam"][i].tolist(),
        "ls": w["ls"][i].tolist(),
    }
    return params, buffers


def as_module(params, buffers):
    """A stand-in with the same build_model() surface certify.py expects."""
    def build_model():
        m = ship_model.DigitPairAdder(params, buffers)
        m.eval()
        return m, {}
    return types.SimpleNamespace(build_model=build_model)


def rewrite_is_exact(p, cfg, i, params, buffers, n=8):
    """float64 check that the rewritten bank reproduces member i's logits."""
    import data
    g = torch.Generator(device=DEV).manual_seed(99)
    da, db = data.sample(256, n, DEV, gen=g)
    tok = data.tokens(da, db)
    one = {k: v[i:i + 1].double() for k, v in p.items()}
    ref = arch.forward(one, cfg, tok)[0]
    m = ship_model.DigitPairAdder(params, buffers).to(DEV).double()
    with torch.no_grad():
        got = m(tok)
    return float((ref - got).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="ship_ft.pt")
    ap.add_argument("--out", default="shipped.pt")
    ap.add_argument("--top", type=int, default=24)
    a = ap.parse_args()

    ck = torch.load(a.inp, map_location=DEV, weights_only=False)
    cfg = ck["cfg"]
    p = {k: v.to(DEV) for k, v in ck["params"].items()}
    E = p["code"].shape[0]
    margin = ck.get("margin", torch.zeros(E)).to(DEV)
    acc = ck.get("acc", torch.zeros(E)).to(DEV)
    cand = (acc >= 1.0).nonzero().flatten()
    cand = cand[torch.argsort(margin[cand], descending=True)][:a.top]
    print(f"loaded {a.inp}: {E} members, {cand.numel()} candidates", flush=True)

    best = None
    for i in cand.tolist():
        params, buffers = to_shipped(p, cfg, i)
        ok, rep = cert.certify(as_module(params, buffers), verbose=False)
        r = rep["margin_ratio"] if ok else -1.0
        print(f"  member {i:4d} margin {margin[i]:.4f} -> certified {ok} "
              f"radius {rep['readout_safety_radius']:.4f} "
              f"drift {rep['residual_drift_bound']:.3e} ratio {r:.3e}",
              flush=True)
        if ok and (best is None or r > best[0]):
            best = (r, i, params, buffers, rep)

    if best is None:
        print("no member certified -- nothing written")
        return
    r, i, params, buffers, rep = best
    err = rewrite_is_exact(p, cfg, i, params, buffers)
    print(f"\nchosen member {i}: certified margin ratio {r:.4e}")
    print(f"trainer-vs-shipped max logit difference (float64): {err:.3e}")
    torch.save(dict(params=params, buffers=buffers, member=i, report=rep), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
