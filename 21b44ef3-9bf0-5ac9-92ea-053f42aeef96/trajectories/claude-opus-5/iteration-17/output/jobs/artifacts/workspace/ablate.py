"""Is the self-attention doing real work?

Two things are checked on the shipped weights:

  1. the attention map varies with the input (count distinct argmax routings);
  2. deleting the CONTENT part of the key -- leaving only the fixed relative
     position bias, i.e. turning the attention into a pattern that no longer
     depends on the input -- destroys the model.

Measured on carry-heavy inputs.  On uniform digits the immediate predecessor is
usually the right place to attend to anyway, so a position-only pattern scores
misleadingly well there; both distributions are reported.
"""
import argparse

import torch

import arch
import build
import data
import reduce as red

DEV = "cuda"


def ship_to_arch(sd):
    """Shipped state_dict -> the ensemble parameterisation (E=1)."""
    code = torch.cat([sd["code0"], sd["code_free"]]).to(DEV)
    return {
        "code": code[None, :, None].clone(),
        "Wb": torch.ones(1, 2, 1, device=DEV),
        "bb": (-sd["bank_w"].to(DEV) * sd["knee"].to(DEV))[None].clone(),
        "kw": torch.zeros(1, 2, device=DEV),
        "vw": torch.zeros(1, 2, device=DEV),
        "vb": torch.zeros(1, 1, device=DEV),
        "uA": sd["carry_w"].to(DEV)[None].clone(),
        "uB": sd["fold"].to(DEV)[None].clone(),
        "lam": sd["lam"].to(DEV)[None].clone(),
        "q": torch.ones(1, 1, device=DEV),
        "log_ls": torch.zeros(1, 1, device=DEV),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    a = ap.parse_args()
    mod = build.load_fresh(a.path, "ablate_mod")
    model, _ = mod.build_model()
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    p = ship_to_arch(sd)
    cfg = red.ship_cfg()

    g = torch.Generator(device=DEV).manual_seed(4242)
    sets = {
        "uniform 8-digit": data.uniform_pairs(16384, 8, DEV, gen=g),
        "carry-heavy    ": data.sample(16384, 8, DEV, regimes=(0.9,), gen=g),
        "mixed          ": data.sample(16384, 8, DEV, gen=g),
    }
    for name, (da, db) in sets.items():
        tok, tgt = data.tokens(da, db), data.targets(da, db)
        valid = tgt >= 0
        out = []
        for kc in (True, False):
            lg, parts = arch.forward(p, cfg, tok, return_parts=True,
                                     key_content=kc)
            pred = lg.argmax(-1)[0]
            em = (((pred == tgt) | ~valid).all(-1)).float().mean().item()
            out.append(em)
            if kc:
                am = parts["aA"][0].argmax(-1)                  # (B,P)
                n_pat = len({tuple(r) for r in am.tolist()})
                frac = (am != (torch.arange(tok.shape[1], device=DEV) - 1)
                        .clamp(min=0)).any(-1).float().mean().item()
        print(f"{name}: shipped {out[0]:.4f} | key content removed "
              f"{out[1]:.4f} | distinct attention routings {n_pat}/16384 | "
              f"differs from attend-to-previous on {frac:.3f} of inputs")


if __name__ == "__main__":
    main()
