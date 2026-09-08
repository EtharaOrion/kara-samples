"""Independent checks on the shipped file.

Three things worth showing separately from "it scores 1.0":

  precision  -- the certificate is computed in float64 but the graded model runs
                in float32; check the two agree on every decoded digit.

  ablation   -- if the self-attention were a fixed pattern dressed up as
                attention, deleting the content-dependent part of the score
                would not hurt.  Measured on carry-heavy inputs: on uniform
                digits a transparent place (a+b == 9) is rare, so "attend to the
                previous position" is almost always the same answer as "attend
                to the nearest earlier non-transparent position", and the
                ablation looks harmless for the wrong reason.

  routing    -- print, for concrete inputs, which position each query actually
                lands on, and confirm it moves when the digits move but the
                positions do not.
"""

import argparse
import importlib.util
import torch

import data


def load(path):
    spec = importlib.util.spec_from_file_location("shipped", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model, meta = mod.build_model()
    return mod, model, meta


def heads(model, a, b, mode="full", dtype=torch.float32):
    """Re-implementation of the shipped forward that exposes the attention and
    can ablate it.  `mode`:
        full    -- as shipped
        nokey   -- drop the content-dependent term: score is pure recency bias
        meankey -- replace each key by the batch-mean key (input-independent)
    """
    c = model.prototypes().to(dtype)
    x = c[a] + c[b]
    g = (model.bank_w.to(dtype) * x[..., None] + model.knee.to(dtype)).clamp(0.0, 1.0)
    k = g @ model.key_w.to(dtype)
    v = g @ model.val_w.to(dtype)
    if mode == "nokey":
        k = torch.zeros_like(k)
    elif mode == "meankey":
        k = k.mean(0, keepdim=True).expand_as(k)

    P = a.shape[-1]
    i = torch.arange(P, device=a.device)
    dist = (i[:, None] - i[None, :]).to(dtype)
    mA = (dist > 0).clone(); mA[0, 0] = True
    mB = dist >= 0
    s = k[:, None, :] + model.lam.to(dtype) * dist
    neg = torch.finfo(dtype).min
    wA = torch.softmax(s.masked_fill(~mA, neg), -1)
    wB = torch.softmax(s.masked_fill(~mB, neg), -1)
    y = (x + model.carry_w.to(dtype) * (wA * v[:, None, :]).sum(-1)
         + model.fold.to(dtype) * (wB * v[:, None, :]).sum(-1))
    return -(y[..., None] - c) ** 2, wA, wB


def accuracy(model, n, gen, mode, mix, B=4096, batches=8, dtype=torch.float32):
    hits = tot = 0
    for _ in range(batches):
        a, b = data.batch(B, n, "cpu", gen, held_out=True, mix=mix)
        ap, bp = data.pad(a, b)
        tgt, mask = data.targets(a, b)
        logits, _, _ = heads(model, ap, bp, mode=mode, dtype=dtype)
        pred = logits.argmax(-1)
        hits += int(((pred == tgt) | ~mask).all(-1).sum())
        tot += ap.shape[0]
    return hits / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    a = ap.parse_args()
    mod, model, meta = load(a.path)
    print(f"file {a.path}  parameters {sum(p.numel() for p in model.parameters())}\n")

    g = torch.Generator().manual_seed(7)

    # ---- precision -------------------------------------------------------
    da, db = data.batch(8192, 8, "cpu", g, held_out=True)
    dap, dbp = data.pad(da, db)
    l32, _, _ = heads(model, dap, dbp, dtype=torch.float32)
    l64, _, _ = heads(model, dap, dbp, dtype=torch.float64)
    agree = int((l32.argmax(-1) == l64.argmax(-1)).all())
    print(f"precision: float32 and float64 decode identically on 8192 pairs: "
          f"{bool(agree)}")

    # ---- ablation --------------------------------------------------------
    UNIFORM = (1.0, 0.0, 0.0)          # independent uniform digits
    CARRY = (0.0, 0.35, 0.65)          # many transparent places / long chains
    print("\nablation (exact-match on held-out 8-digit pairs)")
    print(f"{'regime':>18} {'shipped':>9} {'no key':>9} {'mean key':>9}")
    for name, mix in (("uniform digits", UNIFORM), ("carry-heavy", CARRY)):
        row = [accuracy(model, 8, torch.Generator().manual_seed(11), m, mix)
               for m in ("full", "nokey", "meankey")]
        print(f"{name:>18} {row[0]:9.4f} {row[1]:9.4f} {row[2]:9.4f}")

    # ---- routing ---------------------------------------------------------
    print("\nrouting: which earlier position the carry-in head attends to")
    cases = [(11111111, 11111111, "no transparent place: every query steps back one"),
             (11118111, 11111111, "transparent at place 3: query 5 skips over it"),
             (88888889, 11111111, "place 0 generates, places 1-7 all transparent"),
             (88888888, 11111111, "every place transparent, nothing generates")]
    for x, y, why in cases:
        da = [0] + [(x // 10 ** i) % 10 for i in range(8)] + [0]
        db = [0] + [(y // 10 ** i) % 10 for i in range(8)] + [0]
        ta = torch.tensor([da]); tb = torch.tensor([db])
        _, wA, _ = heads(model, ta, tb)
        src = wA[0].argmax(-1).tolist()
        print(f"  {x} + {y}  ({why})")
        print(f"    query pos 1..9 attends to {src[1:]}   sum ok: "
              f"{mod.add(model, x, y) == x + y}")


if __name__ == "__main__":
    main()
