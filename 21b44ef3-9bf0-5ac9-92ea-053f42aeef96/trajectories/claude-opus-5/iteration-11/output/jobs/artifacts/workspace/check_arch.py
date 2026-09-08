"""Sanity check: hand-set the 11-parameter target weights and confirm the
architecture can express exact addition.  This validates the *architecture*
only -- the shipped weights come from training, not from here."""

import torch
import adder
from data import Sampler

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = adder.default_cfg(free_theta_neg=False, free_alpha=False, free_e1=False,
                        free_ls=False, free_kw=False, free_lam=False,
                        free_rb=False, code_fix=1)
print("n_params:", adder.n_params(cfg))

p = {
    "code": torch.arange(10, dtype=torch.float32, device=device)[None],
    "rb": torch.zeros(1, device=device),
    "theta": torch.tensor([9.0], device=device),
    "theta_neg": torch.tensor([9.0], device=device),
    "alpha": torch.tensor([2.0], device=device),
    "e1": torch.tensor([1.0], device=device),
    "e2": torch.tensor([-10.0], device=device),
    "ls": torch.tensor([1.0], device=device),
    "kw": torch.tensor([2000.0], device=device),
    "lam": torch.tensor([-8.0], device=device),
}

sam = Sampler(device)
for n in (8, 3, 14):
    geo = adder.geometry(n + 2, device)
    tot = ok = 0
    for spec in (None, 0.4, 0.9):
        a, b = sam.sample_digits(4096, n, spec)
        ab, tgt, mask = sam.pack(a, b)
        logits = adder.forward(p, cfg, ab, geo)
        pred = logits.argmax(-1)[0]
        good = ((pred == tgt) | (mask == 0)).all(-1)
        tot += good.numel()
        ok += int(good.sum())
    print(f"n={n:3d}  exact {ok}/{tot}")
