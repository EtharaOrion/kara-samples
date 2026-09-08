"""Audit the shipped /workspace/submission.py.

Checks, in order:
  1  parameter count read off the module itself
  2  the graded interface `add` on random full-width 8-digit pairs
  3  batched exact-match on a large uniform sample and on the held-out split
  4  every one of the 3^8 carry-structure patterns
  5  edge cases (all-9s, maximal carry chains, boundaries of the range)
  6  attention really does work: freeze the maps, and count distinct patterns
  7  numerics: float64 agreement, decision margin
  8  the fixed (non-parameter) constants are don't-cares: working bands
"""

import argparse
import importlib.util
import itertools
import sys

import torch

sys.path.insert(0, "/workspace")
from data import Sampler  # noqa: E402


def load(path="/workspace/submission.py"):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def digits_to_ab(a, b, device):
    B, n = a.shape
    ab = torch.zeros(B, n + 2, 2, dtype=torch.long, device=device)
    ab[:, 1:n + 1, 0] = a
    ab[:, 1:n + 1, 1] = b
    return ab


def batched_exact(model, a, b, device):
    """a, b: int64 digit tensors [B, n] LSB first.  Returns a bool [B]."""
    n = a.shape[1]
    pw = (10 ** torch.arange(n, device=device)).to(torch.int64)
    va, vb = (a * pw).sum(1), (b * pw).sum(1)
    vs = va + vb
    ab = digits_to_ab(a, b, device)
    pred = model(ab).argmax(-1)[:, 1:n + 2]
    pw2 = (10 ** torch.arange(n + 1, device=device)).to(torch.int64)
    got = (pred * pw2).sum(1)
    return got == vs, va, vb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/workspace/submission.py")
    ap.add_argument("--big", type=int, default=1 << 20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    sub = load(args.path)
    model, meta = sub.build_model()
    model = model.to(device)
    sam = Sampler(device)

    print("=" * 70)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[1] parameters: {n_par}   (metadata claims {meta.get('n_parameters')})")
    for name, p in model.named_parameters():
        print(f"      param  {name:12s} {tuple(p.shape)}")
    for name, b in model.named_buffers():
        print(f"      buffer {name:12s} {b.tolist()}")

    # --- 2: the graded interface itself -------------------------------------
    torch.manual_seed(0)
    lo, hi = 10_000_000, 99_999_999
    pairs = torch.randint(lo, hi + 1, (4000, 2)).tolist()
    bad = [(x, y) for x, y in pairs if sub.add(model, x, y) != x + y]
    print(f"[2] add() on 4000 random full-width pairs: {4000-len(bad)}/4000 exact")
    if bad:
        print("      first failures:", bad[:5])

    # --- 3: large batched sweeps --------------------------------------------
    print("[3] batched exact-match, 8 places, both operands full width")
    for label, spec in (("uniform", None), ("p_t=0.4", 0.4),
                        ("p_t=0.7", 0.7), ("p_t=0.9", 0.9)):
        tot = ok = 0
        for _ in range(max(1, args.big // (1 << 16))):
            a, b = sam.sample_digits(1 << 16, 8, spec)
            good, _, _ = batched_exact(model, a, b, device)
            tot += good.numel()
            ok += int(good.sum())
        print(f"      {label:9s} {ok}/{tot}")
    # held-out split only
    tot = ok = 0
    while tot < 1 << 17:
        a, b = sam.sample_digits(1 << 16, 8, None)
        good, va, vb = batched_exact(model, a, b, device)
        keep = sam.bucket(va, vb) == 0
        tot += int(keep.sum())
        ok += int((good & keep).sum())
    print(f"      held-out  {ok}/{tot}")

    # --- 4: every carry-structure pattern -----------------------------------
    pats = torch.tensor(list(itertools.product([0, 1, 2], repeat=8)), device=device)
    reps = 8
    cls = pats.repeat(reps, 1)
    a, b = sam._from_classes(cls)
    good, _, _ = batched_exact(model, a, b, device)
    print(f"[4] all 3^8 carry patterns x{reps}: {int(good.sum())}/{good.numel()}")
    if int(good.sum()) < good.numel():
        i = int((~good).nonzero()[0])
        print("      e.g.", cls[i].tolist(), a[i].tolist(), b[i].tolist())

    # --- 5: edges ------------------------------------------------------------
    edges = [(99999999, 99999999), (10000000, 10000000), (99999999, 10000001),
             (19999999, 10000001), (12345678, 87654321), (11111111, 88888889),
             (55555555, 44444445), (10000000, 99999999), (99999998, 10000001),
             (98765432, 12345678), (50000000, 50000000), (99999999, 90000001)]
    bad = [(x, y, sub.add(model, x, y)) for x, y in edges
           if sub.add(model, x, y) != x + y]
    print(f"[5] edge cases: {len(edges)-len(bad)}/{len(edges)}  {bad if bad else ''}")

    # --- 6: is the attention load-bearing? -----------------------------------
    a, b = sam.sample_digits(4096, 8, 0.5)
    ab = digits_to_ab(a, b, device)
    with torch.no_grad():
        P = ab.shape[1]
        code = model.code()
        x = code[ab].sum(-1)
        t = model.alpha * (x - model.theta)
        upos, uneg = t.clamp(0, 1), (-t).clamp(0, 1)
        key = model.kw * (upos + uneg)
        idx = torch.arange(P, device=device)
        dist = (idx[:, None] - idx[None, :]).to(x.dtype)
        neg = torch.finfo(x.dtype).min / 4
        strict = (idx[None, :] < idx[:, None]).clone()
        strict[0, 0] = True
        incl = idx[None, :] <= idx[:, None]
        base = key[:, None, :] + model.lam * dist
        a1 = torch.softmax(base.masked_fill(~strict, neg), -1)
        a2 = torch.softmax(base.masked_fill(~incl, neg), -1)
        pat = a1.argmax(-1)
        uniq = len({tuple(r) for r in pat.tolist()})
        rows_vary = (pat != pat[0]).any(0).float().mean().item()
        # freeze both maps at their batch mean -> a fixed, input-independent pattern
        f1, f2 = a1.mean(0, keepdim=True).expand_as(a1), a2.mean(0, keepdim=True).expand_as(a2)
        o1, o2 = (f1 * upos[:, None, :]).sum(-1), (f2 * upos[:, None, :]).sum(-1)
        y = x + model.e1 * o1 + model.e2 * o2
        pred = (-(y[..., None] - code) ** 2).argmax(-1)[:, 1:10]
        pw = (10 ** torch.arange(8, device=device)).to(torch.int64)
        vs = (a * pw).sum(1) + (b * pw).sum(1)
        pw2 = (10 ** torch.arange(9, device=device)).to(torch.int64)
        froz = float(((pred * pw2).sum(1) == vs).float().mean())
    live, _, _ = batched_exact(model, a, b, device)
    print(f"[6] attention: {uniq} distinct argmax patterns over 4096 inputs, "
          f"{rows_vary:.2f} of rows vary; live acc {float(live.float().mean()):.4f} "
          f"-> frozen-map acc {froz:.4f}")

    # --- 7: numerics ---------------------------------------------------------
    a, b = sam.sample_digits(1 << 15, 8, 0.6)
    ab = digits_to_ab(a, b, device)
    with torch.no_grad():
        l32 = model(ab)
        m64 = sub.build_model()[0].to(device).double()
        l64 = m64(ab)
        agree = float((l32.argmax(-1) == l64.argmax(-1)).float().mean())
        top2 = l64[:, 1:10].topk(2, dim=-1).values
        margin = float((top2[..., 0] - top2[..., 1]).min())
        gap = float((model.code()[1:] - model.code()[:-1]).abs().min())
    print(f"[7] float32 vs float64 argmax agreement {agree:.6f}; "
          f"worst decision margin {margin:.4g} (min prototype gap {gap:.4g}, "
          f"ratio {margin/gap**2:.4g})")

    # --- 8: the fixed constants are don't-cares ------------------------------
    print("[8] working band of each non-parameter constant "
          "(exact-match on 65536 mixed pairs)")
    a, b = sam.sample_digits(1 << 16, 8, 0.5)
    base_vals = {k: float(getattr(model, k)) for k in ("alpha", "kw", "lam")}
    for k, grid in (("alpha", [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 30.0, 100.0]),
                    ("kw", [20.0, 50.0, 200.0, 2000.0, 20000.0]),
                    ("lam", [-1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0, -20.0, -40.0])):
        res = []
        for v in grid:
            with torch.no_grad():
                getattr(model, k).fill_(v)
            good, _, _ = batched_exact(model, a, b, device)
            res.append(f"{v:g}:{float(good.float().mean()):.4f}")
        with torch.no_grad():
            getattr(model, k).fill_(base_vals[k])
        print(f"      {k:6s} (shipped {base_vals[k]:g})  " + "  ".join(res))
    print("=" * 70)


if __name__ == "__main__":
    main()
