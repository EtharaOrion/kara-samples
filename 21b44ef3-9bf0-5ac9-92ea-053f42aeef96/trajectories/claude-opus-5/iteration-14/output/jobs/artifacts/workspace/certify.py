"""Whole-domain certificate for a reduced member.

The model's output at a place depends only on (a_i, b_i, carry_in, carry_out), and
carry_in / carry_out are produced by attention over a key/value stream that is a
function of a_i+b_i alone.  So if we prove

  (1) the clamp bank is exactly saturated for all 100 digit pairs, and assigns
      them to absorb / transparent / generate exactly as a+b <= 8 / == 9 / >= 10,
  (2) the value is exactly 0 on absorb+transparent and 1 on generate,
  (3) each attention head puts all but eps of its mass on the intended key, with a
      bound on eps that holds for every input of width <= Pmax, and
  (4) for every (a, b, carry_in) the read-out argmax is the right digit even when
      the attention output is perturbed by eps,

then the model is exact on the entire 8-digit domain (and on any width <= Pmax-2).
All arithmetic here is float64; float32 execution noise is measured separately.
"""
import math
import torch

import arch


def _cls(a, b):
    s = a + b
    return 0 if s <= 8 else (1 if s == 9 else 2)


def certify(p, cfg, Pmax=26, verbose=True):
    """p: single-member dict of float tensors (no ensemble axis).  Returns a report."""
    p64 = {k: v.detach().to("cpu", torch.float64).unsqueeze(0) for k, v in p.items()}
    C, U = cfg["C"], cfg["U"]
    assert C == 1, "certificate is written for the scalar residual form"
    code = arch.full_code(p64, cfg)[0, :, 0]                      # (10,)
    Bw = p64["Bw"][0, 0]                                          # (U,)
    bb = p64["bb"][0]
    kw = p64["kw"][0]
    vw = p64["vw"][0]
    vb = p64["vb"][0] if "vb" in p64 else torch.zeros((), dtype=torch.float64)
    rb = p64["rb"][0, 0] if "rb" in p64 else torch.zeros((), dtype=torch.float64)
    q = p64["q"][0]
    lam = p64["lam"][0]
    w1 = p64["w1"][0, 0]
    w2 = p64["w2"][0, 0]

    rep = {"ok": True, "fail": []}

    def bad(msg):
        rep["ok"] = False
        rep["fail"].append(msg)

    # ---------------------------------------------------------------- (1) bank
    A = torch.arange(10)
    x = code[A][:, None] + code[A][None, :]                       # (10,10)
    z = x[:, :, None] * Bw[None, None, :] + bb[None, None, :]     # (10,10,U)
    u = z.clamp(0.0, 1.0)
    # exact saturation: every z is <= 0 or >= 1
    interior = ((z > 0.0) & (z < 1.0))
    slack = torch.where(z <= 0.0, -z, z - 1.0)                    # >= 0 iff saturated
    rep["bank_saturation_slack"] = float(slack.min())
    if bool(interior.any()):
        bad(f"bank not saturated for {int(interior.sum())} of {z.numel()} (pair,unit) cells")

    cls = torch.tensor([[_cls(int(a), int(b)) for b in A] for a in A])
    rep["classes"] = {}
    # ------------------------------------------------------- (2) key and value
    k = (u * kw).sum(-1)
    v = (u * vw).sum(-1) + vb
    kv = {}
    for c in (0, 1, 2):
        m = cls == c
        kv[c] = (k[m].min().item(), k[m].max().item(), v[m].min().item(), v[m].max().item())
        if k[m].max() - k[m].min() > 1e-12:
            bad(f"key not constant within class {c}: spread {float(k[m].max()-k[m].min()):.3e}")
        if v[m].max() - v[m].min() > 1e-12:
            bad(f"value not constant within class {c}: spread {float(v[m].max()-v[m].min()):.3e}")
    rep["class_key_value"] = kv
    v0, v1, v2 = kv[0][2], kv[1][2], kv[2][2]
    k0, k1, k2 = kv[0][0], kv[1][0], kv[2][0]
    if not (abs(v0 - v1) < 1e-12 and abs(v0 - v1) < 1e-12):
        bad(f"absorb/transparent values differ: {v0} vs {v1}")
    vgap = v2 - v0
    rep["value_absorb"], rep["value_generate"] = v0, v2
    if vgap <= 0:
        bad(f"generate value not above absorb value ({v2} vs {v0})")
    kgap = min(k0, k2) - k1                       # how far the transparent key is below
    rep["key_notch_gap"] = kgap
    if kgap <= 0:
        bad(f"transparent key is not the lowest ({k1} vs {k0},{k2})")
    if abs(k0 - k2) > 1e-12:
        bad(f"absorb and generate keys differ: {k0} vs {k2}")

    # -------------------------------------------------- (3) attention leakage
    lm = float(lam)
    if lm >= 0:
        bad(f"lam must be negative (recency); got {lm}")
        lm = -1e-9
    same = math.exp(lm) / (1.0 - math.exp(lm))            # other non-transparent keys
    tr_exp = float(q) * (-kgap) + abs(lm) * (Pmax - 1)
    trans = Pmax * math.exp(tr_exp)                       # transparent keys
    S = same + trans
    eps = S / (1.0 + S) * abs(vgap)                       # |c - ideal| bound (value units)
    rep["leak_same_class"], rep["leak_transparent"], rep["eps_value"] = same, trans, eps

    # ------------------------------------------------------------ (4) read-out
    half = 0.5 * code * code
    worst = float("inf")
    worst_case = None
    pert = eps * (abs(float(w1)) + abs(float(w2)))
    for a in range(10):
        for b in range(10):
            c = int(cls[a, b])
            for cin in (0, 1):
                cout = 1 if c == 2 else (0 if c == 0 else cin)
                tgt = (a + b + cin) % 10
                base = float(x[a, b]) + float(w1) * (v0 + cin * vgap) \
                       + float(w2) * (v0 + cout * vgap) + float(rb)
                # w1,w2 multiply values in {v0, v0+vgap}; carry-in/out semantics above
                for o in (base - pert, base + pert):
                    sc = o * code - half
                    m = float(sc[tgt] - sc[torch.arange(10) != tgt].max())
                    if m < worst:
                        worst, worst_case = m, (a, b, cin, cout, tgt)
    rep["readout_margin"] = worst
    rep["readout_worst_case"] = worst_case
    rep["readout_perturbation"] = pert
    rep["code_gap"] = float((code.sort().values[1:] - code.sort().values[:-1]).min())
    if worst <= 0:
        bad(f"read-out margin non-positive: {worst:.4e} at {worst_case}")

    if verbose:
        print("--- certificate ---")
        for kk in ("bank_saturation_slack", "class_key_value", "value_absorb",
                   "value_generate", "key_notch_gap", "leak_same_class",
                   "leak_transparent", "eps_value", "readout_perturbation",
                   "readout_margin", "readout_worst_case", "code_gap"):
            print(f"  {kk:24s} {rep[kk]}")
        print(f"  PASS = {rep['ok']}" + ("" if rep["ok"] else f"  fails={rep['fail']}"))
    return rep
