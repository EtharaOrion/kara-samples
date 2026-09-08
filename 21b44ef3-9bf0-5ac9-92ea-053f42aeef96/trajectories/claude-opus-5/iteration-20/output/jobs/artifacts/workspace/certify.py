"""Whole-domain certificate for the graded file, in float64.

If the gate bank is saturated at every one of the 100 digit pairs, then the key
and value at a place depend on the pair only through its class -- so the model's
behaviour on all 10^16 operand pairs is determined by the 3^8 class patterns.
This enumerates every pattern, computes the exact attention, and checks the
read-out for every digit pair consistent with each pattern.  No sampling.
"""

import argparse
import itertools

import torch

import verify

CLASS_NAMES = {(0, 0): "absorb", (1, 0): "transparent", (1, 1): "generate",
               (0, 1): "inverted"}


def weights(model):
    return {
        "code": torch.cat([model.code01, model.code_free]).double(),
        "knee": model.knee.double(),
        "carry_w": model.carry_w.double(),
        "fold": model.fold.double(),
        "bank_w": model.bank_w.double(),
        "key_w": model.key_w.double(),
        "lam": model.lam.double(),
    }


def gate_classes(w):
    """Per-pair gate values and the worst saturation slack over all 100 pairs."""
    code = w["code"]
    x = code[:, None] + code[None, :]                       # (10,10) = code[a]+code[b]
    t = w["bank_w"] * (x[..., None] - w["knee"])            # (10,10,2)
    slack = torch.maximum(-t, t - 1.0).amin().item()
    gates = (t >= 0.5).long()                               # exact once saturated
    return x, gates, slack


def attention(keys, lam):
    """Exact float64 attention for one key row: returns h_in, h_out weights."""
    p = keys.shape[-1]
    pos = torch.arange(p, dtype=torch.float64)
    delta = pos[:, None] - pos[None, :]
    logits = keys[..., None, :] + lam * delta
    blocked = torch.finfo(torch.float64).min
    earlier = (delta > 0) | ((pos[:, None] == 0) & (pos[None, :] == 0))
    upto = delta >= 0
    w_in = torch.softmax(torch.where(earlier, logits, blocked), dim=-1)
    w_out = torch.softmax(torch.where(upto, logits, blocked), dim=-1)
    return w_in, w_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--places", type=int, default=8)
    a = ap.parse_args()

    mod = verify.load(a.path)
    model, _ = mod.build_model()
    w = weights(model)
    x, gates, slack = gate_classes(w)

    print("code:", [round(float(v), 6) for v in w["code"]])
    print("knee:", [round(float(v), 6) for v in w["knee"]],
          " carry_w:", round(float(w["carry_w"]), 6),
          " fold:", round(float(w["fold"]), 6))
    print(f"gate saturation slack over all 100 pairs: {slack:.6f}  "
          f"({'SATURATED' if slack > 0 else 'NOT SATURATED -- certificate invalid'})")
    if slack <= 0:
        return 1

    # Which class does each digit-sum land in?
    seen = {}
    for da in range(10):
        for db in range(10):
            seen.setdefault((int(gates[da, db, 0]), int(gates[da, db, 1])), []).append(da + db)
    for g, sums in sorted(seen.items()):
        print(f"   gate {g} -> {CLASS_NAMES.get(g,'?'):11s} digit sums "
              f"{sorted(set(sums))}")

    classes = sorted(seen.keys())
    pad_class = (int(gates[0, 0, 0]), int(gates[0, 0, 1]))
    key_of = {g: float(w["key_w"]) * (g[1] - g[0]) for g in classes}
    val_of = {g: float(g[1]) for g in classes}

    n = a.places
    patterns = torch.tensor(list(itertools.product(range(len(classes)), repeat=n)))
    npat = patterns.shape[0]
    full = torch.cat([torch.zeros(npat, 1, dtype=torch.long),
                      patterns,
                      torch.zeros(npat, 1, dtype=torch.long)], dim=1)
    full[:, 0] = classes.index(pad_class)
    full[:, -1] = classes.index(pad_class)

    key_tab = torch.tensor([key_of[c] for c in classes], dtype=torch.float64)
    val_tab = torch.tensor([val_of[c] for c in classes], dtype=torch.float64)
    keys = key_tab[full]
    vals = val_tab[full]
    w_in, w_out = attention(keys, float(w["lam"]))
    h_in = torch.einsum("pij,pj->pi", w_in, vals)
    h_out = torch.einsum("pij,pj->pi", w_out, vals)

    # True carry into each place, from the class pattern.
    is_t = torch.tensor([c == (1, 0) for c in classes])[full]
    is_g = torch.tensor([c == (1, 1) for c in classes])[full]
    carry = torch.zeros(npat, n + 2, dtype=torch.long)
    for i in range(1, n + 2):
        carry[:, i] = torch.where(is_t[:, i - 1], carry[:, i - 1],
                                  is_g[:, i - 1].long())
    leak_in = (h_in - carry.double()).abs().max().item()
    print(f"max |carry-in head - true carry| over all {npat} patterns: {leak_in:.3e}")

    code = w["code"]
    worst_margin = float("inf")
    wrong = 0
    checks = 0
    for ci, cls in enumerate(classes):
        pairs = [(da, db) for da in range(10) for db in range(10)
                 if (int(gates[da, db, 0]), int(gates[da, db, 1])) == cls]
        if not pairs:
            continue
        pa = torch.tensor([p[0] for p in pairs])
        pb = torch.tensor([p[1] for p in pairs])
        xv = x[pa, pb]                                          # (K,)
        sel = full[:, 1:] == ci                                 # (npat, n+1) places 1..n+1
        idx = sel.nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        hs = h_in[:, 1:][idx[:, 0], idx[:, 1]]
        ho = h_out[:, 1:][idx[:, 0], idx[:, 1]]
        cin = carry[:, 1:][idx[:, 0], idx[:, 1]]
        r = xv[None, :] + w["carry_w"] * hs[:, None] + w["fold"] * ho[:, None]
        want = (pa[None, :] + pb[None, :] + cin[:, None]) % 10
        dist = (r[..., None] - code).abs()
        good = dist.gather(-1, want[..., None])
        other = dist.scatter(-1, want[..., None], float("inf")).amin(-1, keepdim=True)
        gap = (other - good).squeeze(-1)
        worst_margin = min(worst_margin, float(gap.min()))
        wrong += int((gap <= 0).sum())
        checks += gap.numel()

    gaps = (code[1:] - code[:-1]).abs()
    print(f"\nexhaustive check: {checks} (pattern, place, digit-pair) cases, "
          f"{wrong} wrong")
    print(f"worst read-out margin: {worst_margin:.6f}   "
          f"min prototype spacing: {float(gaps.min()):.6f}")
    print("CERTIFICATE:", "PASS -- exact on every 8-digit operand pair" if wrong == 0
          else "FAIL")
    return 0 if wrong == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
